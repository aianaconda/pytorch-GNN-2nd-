# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:22:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

import torch
import dgl
import numpy as np
import scipy.sparse as ssp

# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True)
    # remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g

def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    print(len(val_src))
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix

def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / \
        (values.max(0, keepdims=True) - values.min(0, keepdims=True))