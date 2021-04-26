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
from functools import partial
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as tordata
from code_05_DataLoader import load_data,collate_fn_for_test
from code_07_gaitset import GaitSetNet,np2var

print("torch v:",torch.__version__, " cuda v:",torch.version.cuda)







print("Initializing data source...")
####################################################
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
    label_train_num = 70#训练集的个数。剩下是测试集
    batch_size = (8, 16)
    frame_num = 30
    hidden_dim =  256 
 ###########################################################
    
    
pathstr = 'perdata'
label_train_num = 70#训练集的个数。剩下是测试集
batch_size = (8, 16)
frame_num = 30
hidden_dim =  256    
    


num_workers = torch.cuda.device_count()
print( "cuda.device_count",num_workers )
if num_workers<=1: #一块或没有GPU，则使用主线程处理
    num_workers =0
print( "num_workers",num_workers )   
  
dataconf= {
    'dataset_path': pathstr,
    'imgresize': '64',
    'label_train_num': label_train_num,   #训练集的个数。剩下是测试集
    'label_shuffle': True,
}
train_source, test_source = load_data(**dataconf) #一次全载入
print( len(test_source.data_seq_dir)) # label_num* type10* view11


#采样器
sampler_batch_size =32
#采样器的聚合函数
collate_train = partial(collate_fn_for_test, frame_num=frame_num)
#定义数据加载器 ：每次迭代，按照采样器的索引去test_source中取出数据
test_loader = tordata.DataLoader(
            dataset=test_source,
            batch_size=sampler_batch_size,
            sampler=tordata.sampler.SequentialSampler(test_source),
            collate_fn=collate_train,
            num_workers=num_workers)



encoder = GaitSetNet(hidden_dim,frame_num).float()
encoder = nn.DataParallel(encoder)
encoder.cuda()
encoder.eval()



ckp = 'checkpoint'
save_name = '_'.join(map(str,[hidden_dim,int(np.prod( batch_size  )),
                           frame_num,'full'])) 
      
ckpfiles= sorted(os.listdir(ckp) )
if len(ckpfiles)>1:
    modecpk =os.path.join(ckp, ckpfiles[-2]  )  
    encoder.module.load_state_dict(torch.load(modecpk))#加载模型文件
    print("load cpk !!! ",modecpk)
else:
    print("No  cpk!!!")



def cuda_dist(x, y):#计算距离
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

#计算多角度准确率
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result

def evaluation(data):
    feature, meta, label = data
    view, seq_type = [],[]
    for i in meta:
        view.append(i[2] )
        seq_type.append(i[1])
        
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)


    probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
    gallery_seq = [['nm-01', 'nm-02', 'nm-03', 'nm-04']]

    num_rank = 5
    acc = np.zeros([len(probe_seq), view_num, view_num, num_rank])
    for (p, probe_s) in enumerate(probe_seq):
        for gallery_s in gallery_seq:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_s) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_s) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]
                    
                    if len(probe_x)>0 and len(gallery_x)>0:

                        dist = cuda_dist(probe_x, gallery_x)
                        idx = dist.sort(1)[1].cpu().numpy()#返回排序后的索引，（【0】是排序后的值）
                        rank_data = np.round(
                            np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                   0) * 100 / dist.shape[0], 2)
                        acc[p, v1, v2, 0:len(rank_data)] = rank_data

    return acc

print('test_loader',len(test_loader))
time = datetime.now()
print('开始评估模型...')
feature_list = list()
view_list = list()
seq_type_list = list()
label_list = list()
batch_meta_list = []


with torch.no_grad():
    for i, x in  tqdm (enumerate(test_loader)):
        batch_data, batch_meta, batch_label = x
        batch_data =np2var(batch_data).float()#[2, 212, 64, 44]
    
        feature = encoder(batch_data)#[4, 62, 64]
        
        feature_list.append(feature.view(feature.shape[0], -1).data.cpu().numpy())#sampler_batch_size 个特征  
        batch_meta_list += batch_meta
        label_list += batch_label




test = (np.concatenate(feature_list, 0), batch_meta_list, label_list)
acc = evaluation(test)
print('评估完成. 耗时:', datetime.now() - time)



for i in range(1):
    print('===Rank-%d 准确率===' % (i + 1))
    print('携带包裹: %.3f,\t普通: %.3f,\t穿大衣: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i])))


for i in range(1):
    print('===Rank-%d 准确率(除去自身的行走条件)===' % (i + 1))
    print('携带包裹: %.3f,\t普通: %.3f,\t穿大衣: %.3f' % (
        de_diag(acc[0, :, :, i]),
        de_diag(acc[1, :, :, i]),
        de_diag(acc[2, :, :, i])))


np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print('===Rank-%d 的每个角度准确率 (除去自身的行走条件)===' % (i + 1))
    print('携带包裹:', de_diag(acc[0, :, :, i], True))
    print('普通:', de_diag(acc[1, :, :, i], True))
    print('穿大衣:', de_diag(acc[2, :, :, i], True))



