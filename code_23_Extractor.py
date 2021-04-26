# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:53:47 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np#引入基础库


from code_22_FinetuneResNet150 import (device,load_dir,data_transform,dataset_path,
                                       OwnDataset,DataLoader,ResNet)

from code_22_FinetuneResNet150 import (test,val_loader)


secondmodepth = './CUB150Res101_2.pth'
ResNet.load_state_dict(torch.load( secondmodepth,map_location=device))	#加载本地模型,map_location=device
#test(ResNet, device, val_loader ) #验证模型是否准确  




#重组模型，使其输出视觉特征
ResNet = nn.Sequential(*list(ResNet.children())[:-1])
ResNet = ResNet.to(device)
ResNet.eval()


tfile_labels,classes = load_dir(dataset_path) #载入所有数据
filenames, labels=zip(*tfile_labels)


#为每个类的所有图片计算视觉特征，每个类一个json文件，形成特征矩阵
cur_root = r'./CUBfeature/'
os.makedirs(cur_root, exist_ok=True)#创建目录，用于存放视觉特征

#定义数据集
input_dataset=OwnDataset(filenames, labels,list(range(len(labels))),data_transform['val'])
input_loader=DataLoader(dataset=input_dataset, batch_size=1, shuffle=False)

def savefeature(classdir,filename,obj): #定义函数保存特征
    os.makedirs(classdir, exist_ok=True)#创建子目录
    cur_url=os.path.join(classdir,filename)#定义文件名称
    json.dump(obj,open(cur_url,"w"))#保存json文件
    print("%s has finished ..."%(classdir))
     

def avgfeature(all_features):#计算类别的平均特征
    #对类别中所有图片的特征求平均值
    avg_features = np.sum(np.asarray(all_features),axis = 0)/len(all_features)
    avg_features = torch.tensor(avg_features)
    avg_features = F.normalize(avg_features, dim=0)#对平均特征归一化
    avg_features = avg_features.numpy()#形状为(2048,)
    return avg_features


all_features=[]
target_VC=[]    #用于保存训练集中每个类的视觉特征
test_VC=[]#用于保存测试集中每个类的视觉特征
classsplit = 150 #训练集与测试集的分割标记

i = -1
for batch_idx, data in enumerate(input_loader):#遍历所有数据
    x,y= data
    x=x.to(device)
    if i == -1:
        i = y

    
    if i!=y and len(all_features)>0:
        classdir= os.path.join(cur_root,classes[i]) #定义第i类的路径
        obj={}
        obj["features"]=all_features
#        print("all_features",len(all_features),i,y)
        savefeature(classdir,"ResNet101.json",obj) #保存第i类所有图片的视觉特征
        avg_features = avgfeature(all_features)    #计算第i类的平均视觉特征
        
        if i<classsplit:                            #根据分割标记区分训练集与测试集
            target_VC.append(avg_features.tolist())
#            print("target_VC",len(target_VC),i)
        else:
            test_VC.append(avg_features.tolist())
        i= y   
        
        all_features=[]
    
    with torch.no_grad():
        features = ResNet(x)        #计算图片的视觉特征
        
#    f = features.to("cpu")
#    f = f.detach().numpy()#(1, 2048, 1, 1)
#        fea_vec = f[0].reshape(-1)

    fea_vec= features.cpu().detach().numpy().reshape(-1)#获取结果，形状为(1, 2048, 1, 1)
    
    #归一化
    fea_vec = torch.tensor(fea_vec)
    fea_vec = F.normalize(fea_vec, dim=0)
    fea_vec = fea_vec.numpy()#(2048,)
    all_features.append(fea_vec.tolist()) 
    
#保存最后一类的视觉特征    
classdir= os.path.join(cur_root,classes[i]) 
obj={}
obj["features"]=all_features
savefeature(classdir,"ResNet101.json",obj)
#计算最后一类的平均特征
avg_features = avgfeature(all_features)#形状为(2048,)
test_VC.append(avg_features.tolist())
#保存训练集和测试集中每个类的平均视觉特征
obj={}
obj["train"]=target_VC #前150类，每个类的平均特征。（每类一条数据）
obj["test"]=test_VC    #后50类特征
savefeature(r'./CUBVCfeature/',"ResNet101VC.json",obj)

#用聚类的方法，计算测试集中，所有图片视觉特征的中心点
alltestfeatures = []
for ii,target in enumerate(classes[classsplit:]): #遍历未知类的特征数据
    cur=os.path.join(cur_root,target)
    fea_name=""
    url=os.path.join(cur,"ResNet101.json")
    js = json.load(open(url, "r"))
    cur_features=js["features"]
    alltestfeatures =alltestfeatures+cur_features
      
        
        

from sklearn.cluster import KMeans
#from sklearn.cluster import SpectralClustering

def KM(features):
    clf = KMeans(n_clusters=len(classes)-classsplit, 
                 n_init=50, max_iter=100000, init="k-means++")

    print("Start Cluster ...")
    s=clf.fit(features)
    print("Finish Cluster ...")

    obj={}
    obj["VC"]=clf.cluster_centers_.tolist()

    print('Start writing ...')
    savefeature(r'./CUBVCfeature/',"ResNet101VC_testCenter.json",obj)
    print("Finish writing ...")
    
KM(alltestfeatures)

















