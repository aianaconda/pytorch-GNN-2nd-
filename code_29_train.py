# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:09:03 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import tqdm

import code_31_sampler as sampler_module
from code_30_model import *




#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#读取数据集
dataset_path = './data.pkl'
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
g = dataset['train-graph']
item_texts = dataset['item-texts']

# 设置节点属性值
g.nodes['user'].data['id'] = torch.arange(g.number_of_nodes('user'))
g.nodes['movie'].data['id'] = torch.arange(g.number_of_nodes('movie'))



#加载词向量，用于解析电影标题
fields = {}
examples = []

titlefield = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
fields = [('title', titlefield)]
for i in range(g.number_of_nodes('movie')):
    example = torchtext.data.Example.fromlist( [item_texts['title'][i] ],  fields)
    examples.append(example)    
textset = torchtext.data.Dataset(examples, fields)

titlefield.build_vocab(getattr(textset, 'title'), vectors = "glove.6B.100d")#将样本数据转为词向量
#    titlefield.build_vocab(getattr(textset, 'title'))#将样本数据转为词向量


num_layers =2
hidden_dims = 32

#构建带采样器的输入数据集
neighbor_sampler = sampler_module.NeighborSampler(   g, 'user', 'movie',num_layers)

collator = sampler_module.PinSAGECollator(neighbor_sampler, g, 'movie', textset)

batch_size = 32
#正负样本对采样器
batch_sampler = sampler_module.ItemToItemBatchSampler(
    g, 'user', 'movie', batch_size)

#训练集数据加载器
dataloader = DataLoader( batch_sampler,collate_fn=collator.collate_train)

#测试集数据加载器
dataloader_test = DataLoader( torch.arange(g.number_of_nodes('movie')),
    batch_size=batch_size, collate_fn=collator.collate_test)

dataloader_it = iter(dataloader)
#################################################################


#######################################################################

# Model
model = PinSAGEModel(g, 'movie', textset, hidden_dims, num_layers).to(device)
# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=0.001,  weight_decay=5e-4)

# For each batch of head-tail-negative triplets...
num_epochs = 2
batches_per_epoch = 5000

for epoch_id in range(num_epochs):
    model.train()
    for batch_id in tqdm.trange(batches_per_epoch):
        pos_graph, neg_graph, blocks = next(dataloader_it)
        # Copy to GPU
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)

        loss = model(pos_graph, neg_graph, blocks).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch_id,loss)


model.eval()
with torch.no_grad():
    h_item_batches = []
    for blocks in dataloader_test:
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        h_item_batches.append(model.get_repr(blocks))
    h_item = torch.cat(h_item_batches, 0) 


# len(h_item)      h_item.shape  #[3706, 32]
# len(item_texts['title'])


# test_matrix= dataset['test-matrix'].tocsr()
# val_matrix = dataset['val-matrix'].tocsr()

# n_users = g.number_of_nodes('user')#6040

graph_slice = g.edge_type_subgraph(['watched'])
latest_interactions = dgl.sampling.select_topk(graph_slice, 1, 'timestamp', edge_dir='out')

user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
# each user should have at least one "latest" interaction
# assert torch.equal(user, torch.arange(n_users))


user_batch = user[:batch_size]
latest_item_batch = latest_items[user_batch].to(device=h_item.device)
dist = h_item[latest_item_batch] @ h_item.t()

# dist.shape   #[32, 3706]

#删除已经看过的电影
for i, u in enumerate(user_batch.tolist()):
    interacted_items = g.successors(u, etype='watched')
    dist[i, interacted_items] = -np.inf


scores, re_index = dist.cpu().topk(5, 1)#推荐个数5
# scores = scores.numpy()

for i in range(batch_size):
    uid = user_batch[i].numpy()
    movieid = latest_item_batch[i].cpu().numpy()
    moviestr = item_texts['title']
    print("用户:",uid," 最后一次观看的电影:",movieid,moviestr[ movieid ])
    print("推荐的电影标题:",moviestr[re_index[i].numpy()])
    print("推荐的电影分数:",scores[i].numpy())
    print("推荐的电影 id:",re_index[i].numpy())    


   

