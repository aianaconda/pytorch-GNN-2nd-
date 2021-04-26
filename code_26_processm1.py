# -*- coding: utf-8 -*-

"""
Created on Sat Apr 11 08:03:20 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import os
import re
import pickle
import pandas as pd
import torch
from code_26_builder import PandasGraphBuilder   
from code_27_data_utils import *

directory = './ml-1m'
output_path = './data.pkl'

# Load data
users = []
with open(os.path.join(directory, 'users.dat'), encoding='latin1') as f:
    for l in f:
        id_, gender, age, occupation, zip_ = l.strip().split('::')
        users.append({'user_id': int(id_), })
users = pd.DataFrame(users).astype('category')



movies = []
with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
    for l in f:
        id_, title, genres = l.strip().split('::')
        genres_set = set(genres.split('|'))

        # extract year
        assert re.match(r'.*\([0-9]{4}\)$', title)
        year = title[-5:-1]
        title = title[:-6].strip()

        data = {'movie_id': int(id_), 'title': title, 'year': year}
        for g in genres_set:
            data[g] = True
        movies.append(data)
movies = pd.DataFrame(movies).astype({'year': 'category'})




ratings = []
with open(os.path.join(directory, 'ratings.dat'), encoding='latin1') as f:
    for l in f:
        user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
        ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'timestamp': timestamp,
            })
ratings = pd.DataFrame(ratings)

# Filter the users and items that never appear in the rating table.
distinct_users_in_ratings = ratings['user_id'].unique()
distinct_movies_in_ratings = ratings['movie_id'].unique()
users = users[users['user_id'].isin(distinct_users_in_ratings)]
movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]

# Group the movie features into genres (a vector), year (a category), title (a string)
genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
movies_categorical = movies.drop('title', axis=1)



# Build graph
graph_builder = PandasGraphBuilder()
graph_builder.add_entities(users, 'user_id', 'user') #加节点
graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')#加边
graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')

g = graph_builder.build()

graph_builder.edges_per_relation #边关系
graph_builder.num_nodes_per_type #节点数


# Assign features.
# Note that variable-sized features such as texts or images are handled elsewhere.
#转成张量数组

g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values) #转为索引向量Categories (81, object): [1919, 1920, 1921, 1922, ..., 1997, 1998, 1999, 2000]
g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)

g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)


def train_test_split_by_time(g, column, etype, itype):
    n_edges = g.number_of_edges(etype)
    with g.local_scope():
        def splits(edges):
            num_edges, count = edges.data['train_mask'].shape
            # print(num_edges, count)
            # print(edges.data['rating'])
            # print(edges.data['timestamp'].shape)

            # sort by timestamp
            _, sorted_idx = edges.data[column].sort(1)

            train_mask = edges.data['train_mask']
            val_mask = edges.data['val_mask']
            test_mask = edges.data['test_mask']

            x = torch.arange(num_edges)

            # If one user has more than one interactions, select the latest one for test.
            if count > 1:
                train_mask[x, sorted_idx[:, -1]] = False
                test_mask[x, sorted_idx[:, -1]] = True
            # If one user has more than two interactions, select the second latest one for validation.
            if count > 2:
                train_mask[x, sorted_idx[:, -2]] = False
                val_mask[x, sorted_idx[:, -2]] = True
            return {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}

        g.edges[etype].data['train_mask'] = torch.ones(n_edges, dtype=torch.bool)
        g.edges[etype].data['val_mask'] = torch.zeros(n_edges, dtype=torch.bool)
        g.edges[etype].data['test_mask'] = torch.zeros(n_edges, dtype=torch.bool)
#        g.nodes[itype].data['count'] = g.in_degrees(etype=etype)
        g.group_apply_edges('src', splits, etype=etype)

        train_indices = g.filter_edges(lambda edges: edges.data['train_mask'], etype=etype)
        val_indices = g.filter_edges(lambda edges: edges.data['val_mask'], etype=etype)
        test_indices = g.filter_edges(lambda edges: edges.data['test_mask'], etype=etype)

    return train_indices, val_indices, test_indices
# Train-validation-test split
# This is a little bit tricky as we want to select the last interaction for test, and the
# second-to-last interaction for validation.
train_indices, val_indices, test_indices = train_test_split_by_time(g, 'timestamp', 'watched', 'movie')

# Build the graph with training interactions only.
train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')

# Build the user-item sparse matrix for validation and test set.
val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

# print(val_matrix.shape)
# print(val_matrix)
## Build title set
movie_textual_dataset = {'title': movies['title'].values}

# The model should build their own vocabulary and process the texts.  Here is one example
# of using torchtext to pad and numericalize a batch of strings.
#     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
#     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
#     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
#     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
#     token_ids, lengths = field.process([examples[0].title, examples[1].title])

## Dump the graph and the datasets

dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': movie_textual_dataset,
}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
 
#with open(output_path, 'rb') as f:
#    dataset = pickle.load(f)