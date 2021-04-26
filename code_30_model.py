# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:51:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""


#import numpy as np

#from torch.utils.data import DataLoader
#import torchtext

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
#import dgl.nn.pytorch as dglnn
import dgl.function as fn



def _init_input_modules(g, ntype, textset, hidden_dims):

    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2  #各种的电影类型genre，2维数据
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1 #电影年year，1维数据
            m = nn.Embedding(
                data.max() + 2, hidden_dims, padding_idx=-1) #max=80，共81个再加个padding索引-1
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m
            print("print(column):",column)

    if textset is not None:#电影标题
        for column, field in textset.fields.items():
            if field.vocab.vectors is not None:
                module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
            else:
                print("wrong! please use vectors first!!!")

    return module_dict

class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])

        self.emb.weight.data.copy_(field.vocab.vectors)
        self.emb.weight.requires_grad = False
        
        '''
        pt13会报错
        RuntimeError: you can only change requires_grad flags of leaf variables.
        '''
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
    def forward(self, x, length):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1) / length.unsqueeze(1).float()
#        x = self.emb(x).detach().sum(1) / length.unsqueeze(1).float()
#        x = x.detach() #add
        
        return self.proj(x)



class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """
    def __init__(self, full_graph, ntype, textset, hidden_dims):
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(full_graph, ntype, textset, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.inputs[feature]
            if isinstance(module,  BagOfWordsPretrained):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + '__len']
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)



class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()
        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)
    def forward(self, g, h, weights):
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))  #{'m': edges.data['h']},{'h': torch.sum(nodes.mailbox['m'], dim=1)}
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            return z

        
        

class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = SAGENet(hidden_dims, n_layers)
        
        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

        
    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}
    
    def scorer(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score  
    
    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        # return h_item_dst + self.sage(blocks, h_item)
        
        z =  h_item_dst + self.sage(blocks, h_item)
        z_norm = z.norm(2, 1, keepdim=True)
        z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
        z = z / z_norm
        return z
    
