"""
Created on Fri Feb 14 10:22:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import os
import numpy as np
from datetime import datetime
import sys
from functools import partial
from matplotlib import pyplot as plt

import torchvision
import torch.nn as nn
import torch
import torch.utils.data as tordata
from ranger import *

from code_05_DataLoader import load_data



print("torch v:",torch.__version__, " cuda v:",torch.version.cuda)


pathstr = 'perdata'
label_train_num = 70#训练集的个数。剩下是测试集



##########################################本地
import platform
sysstr = platform.system()
if(sysstr =="Windows"):
    print("Windows")
    pathstr = "D:\样本\图片\gait\CASIA\GaitDatasetB-silh\perdata"
    label_train_num = 10  #训练集的个数。剩下是测试集
    batch_size = (3, 6)
    frame_num = 8
    hidden_dim =  64 
else:
    print("linux")
    pathstr = 'perdata'
#    label_train_num = 70#训练集的个数。剩下是测试集
#    batch_size = (4, 16)
#    frame_num = 30
#    hidden_dim =  256 
#####################################

    
dataconf= {
    'dataset_path': pathstr,
    'imgresize': '64',
    'label_train_num': label_train_num,   #训练集的个数。剩下是测试集
    'label_shuffle': True,
}
 
print("加载训练数据...")  
train_source, test_source = load_data(**dataconf) #一次全载入
print("训练数据集长度：", len(train_source)) # label_num* type10* view11

'''
#显示数据集里的标签
train_label_set = set(train_source.data_label)
print("数据集里的标签:", train_label_set) #人{4, 9, 10, 11, 12, 16, 23, 25, 30, 32}   
    
#获取一条数据 
dataimg, matedata,labelimg = train_source.__getitem__(4)
print("图片样本数据形状：", dataimg.shape," 数据的元信息：", matedata," 数据标签索引：",labelimg)

plt.imshow(dataimg[0])                        # 显示图片
plt.axis('off')                           # 不显示坐标轴
plt.show()



def imshow(img):
    print("图片形状：",np.shape(img))
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

imshow(torchvision.utils.make_grid(torch.from_numpy(dataimg[-10:]).unsqueeze(1),nrow=10))
'''

###############################
print("___")

from code_05_DataLoader import TripletSampler,collate_fn_for_train

#batch_size = (4, 8)
#frame_num = 32

num_workers = torch.cuda.device_count()
print( "cuda.device_count",num_workers )
if num_workers<=1: #一块或没有GPU，则使用主线程处理
    num_workers =0
print( "num_workers",num_workers )   

#采样器
triplet_sampler = TripletSampler(train_source, batch_size)
#采样器的聚合函数
collate_train = partial(collate_fn_for_train, frame_num=frame_num)
#定义数据加载器 ：每次迭代，按照采样器的索引去train_source中取出数据
train_loader = tordata.DataLoader( dataset=train_source,
    batch_sampler=triplet_sampler, collate_fn=collate_train,
    num_workers=num_workers)  

#用数据加载器 获取数据
batch_data, batch_meta, batch_label = next(iter(train_loader))

print(len(batch_data),batch_data.shape )#(18, 8, 64, 44)
print(batch_label)
print("___")

    

from code_07_gaitset import GaitSetNet,TripletLoss,np2var
#hidden_dim =  256 
encoder = GaitSetNet(hidden_dim,frame_num).float()
encoder = nn.DataParallel(encoder)
encoder.cuda()
encoder.train()

optimizer = Ranger(encoder.parameters(),lr=0.004)

TripletLossmode = 'full'
triplet_loss = TripletLoss( int( np.prod( batch_size) ), TripletLossmode ,margin=0.2)
triplet_loss = nn.DataParallel(triplet_loss)
triplet_loss.cuda()


ckp = 'checkpoint'
os.makedirs(ckp, exist_ok=True)
save_name = '_'.join(map(str,[hidden_dim,int(np.prod( batch_size  )),
                           frame_num,'full'])) 
      
ckpfiles= sorted(os.listdir(ckp) )
if len(ckpfiles)>1:
    modecpk =os.path.join(ckp, ckpfiles[-2]  )  
    optcpk = os.path.join(ckp, ckpfiles[-1]  )  
    encoder.module.load_state_dict(torch.load(modecpk))#加载模型文件
    optimizer.load_state_dict(torch.load(optcpk))
    print("load cpk !!! ",modecpk)
  
    
hard_loss_metric = []
full_loss_metric = []
full_loss_num = []
dist_list = []
mean_dist = 0.01
restore_iter = 0
total_iter=10000
lastloss = 65535
trainloss = []

_time1 = datetime.now()
for batch_data, batch_meta, batch_label in train_loader:
    restore_iter += 1
    optimizer.zero_grad()

    batch_data =np2var(batch_data).float()#torch.cuda.DoubleTensor变为torch.cuda.FloatTensor

    feature = encoder(batch_data)

    #将标签转为id
    target_label = np2var(np.array(batch_label)).long()#len=32

    triplet_feature = feature.permute(1, 0, 2).contiguous()#[62, 32, 256]
    triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)#[62, 32]


    (full_loss_metric_, hard_loss_metric_, mean_dist_, full_loss_num_
     )= triplet_loss(triplet_feature, triplet_label)

    
    if triplet_loss.module.hard_or_full == 'full':
        loss = full_loss_metric_.mean()
    else:
        loss = hard_loss_metric_.mean()
    
    trainloss.append(loss.data.cpu().numpy())

    
    
    hard_loss_metric.append(hard_loss_metric_.mean().data.cpu().numpy())
    full_loss_metric.append(full_loss_metric_.mean().data.cpu().numpy())
    full_loss_num.append(full_loss_num_.mean().data.cpu().numpy())
    dist_list.append(mean_dist_.mean().data.cpu().numpy())    



    if  loss> 1e-9:
        loss.backward()
        optimizer.step()
    else:
        print("loss is very small",loss )

    if restore_iter % 1000 == 0:
        print("restore_iter 1000 time:",datetime.now() - _time1)
        _time1 = datetime.now()

    if restore_iter % 100 == 0:

        print('iter {}:'.format(restore_iter), end='')
        print(', hard_loss_metric={0:.8f}'.format(np.mean(hard_loss_metric)), end='')
        print(', full_loss_metric={0:.8f}'.format(np.mean(full_loss_metric)), end='')
        print(', full_loss_num={0:.8f}'.format(np.mean(full_loss_num)), end='')
     
        print(', mean_dist={0:.8f}'.format(np.mean(dist_list)), end='')
        print(', lr=%f' % optimizer.param_groups[0]['lr'], end='')
        print(', hard or full=%r' % TripletLossmode )
        
                
        if lastloss>np.mean(trainloss):        #保存模型
            print("lastloss:", lastloss," loss:",np.mean(trainloss),"need save!")
            lastloss = np.mean(trainloss)
            modecpk = os.path.join(ckp, 
                    '{}-{:0>5}-encoder.pt'.format( save_name, restore_iter))
            optcpk = os.path.join(ckp,
                    '{}-{:0>5}-optimizer.pt'.format(save_name, restore_iter))
            torch.save(encoder.module.state_dict(),modecpk)
            torch.save(optimizer.state_dict(),optcpk)
        else:
            print("lastloss:", lastloss," loss:",np.mean(trainloss),"don't save")
       
        print("__________________")
        
        sys.stdout.flush()
        hard_loss_metric.clear()
        full_loss_metric.clear()
        full_loss_num.clear()
        dist_list.clear()
        trainloss.clear()


    if restore_iter == total_iter:
        break













