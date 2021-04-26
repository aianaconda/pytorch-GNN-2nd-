# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:28:13 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import json
import time
import os
import numpy as np#引入基础库

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv

from code_25_Wasserstein import SinkhornDistance

class GCN(nn.Module):

    def __init__(self,  in_channels, out_channels, hidden_layers,device):
        super(GCN,self).__init__()
        self.m_layers = nn.ModuleList()
        last_c = in_channels
        #定义隐藏层
        for cout in hidden_layers:
            self.m_layers.append(
                    GraphConv(last_c, cout,activation=nn.LeakyReLU(negative_slope=0.2)))
            last_c = cout
            
        self.m_layers.append( GraphConv(last_c, out_channels))

    def forward(self, g,inputs):
        h = inputs
        for layer in self.m_layers:#隐藏层
            h = layer(g, h)
        return F.normalize(h)




#####  L2+Chamfer-Distance

def CDVSc(a,b,device,n,m):


    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes

    #### Start Calculating CD Loss

    CD_loss=None

    A=a[n-m:]
    B=b[n-m:]

    A=A.cpu()
    B=B.cpu()

#    for x in A:
#        for y in B:
#            dis=((x-y)**2).sum()


    for x in A:
        MINI=None
        for y in B:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI

    for x in B:
        MINI=None
        for y in A:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI


    CD_loss=CD_loss.to(device)
    #######

    lamda=0.0003
    lamda=0.0001

    tot_loss=L2_loss+CD_loss*lamda
    return tot_loss

#####
from scipy.optimize import linear_sum_assignment
def BMVSc(a,b,device,n,m):

    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes


    A=a[n-m:]
    B=b[n-m:]

    DIS=torch.zeros((m,m))


    DIS=DIS.to(device)

    for A_id,x in enumerate(A): #生成俩俩距离矩阵
        for B_id,y in enumerate(B):
            dis=((x-y)**2).sum()
            DIS[A_id,B_id]=dis

    matching_loss=0

    cost=DIS.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost) #最优匹配

    for i,x in enumerate(row_ind):
        matching_loss+=DIS[row_ind[i],col_ind[i]]
    
    lamda=0.0001
    tot_loss=L2_loss+matching_loss*lamda

    return tot_loss


def WDVSc(a,b,device,n,m,no_use_VSC=True):
    WD=SinkhornDistance(0.01,1000,None,"mean")
    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes
    A = a[n - m:]
    B = b[n - m:]
    A=A.cpu()
    B=B.cpu()
    if no_use_VSC:
        WD_loss=0.
        P=None
        C=None
    else:
        WD_loss,P,C=WD(A,B)
        WD_loss = WD_loss.to(device)
    lamda=0.001
    tot_loss=L2_loss+WD_loss*lamda
    return tot_loss,P,C

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ParsingAtt(lines):#312列依次转成浮点
    line=lines.strip().split()
    cur=[]
    for x in line:
        y=float(x)
        y=y/100.0
        if y<0.0:
            y=0.0
        cur.append(y)
    return cur

def ParsingClass(lines):#类名，取第二项
    line=lines.strip().split()
    return line[1]

def get_attributes(url,function=None):
    data=[]
    with open(url,"r") as f:
        for lines in f:
            cur =function(lines)
            data.append(cur)
    return data

input_dim=312
classNum=200
unseenclassnum=50
data_path=r'D:\样本\图片\Caltech-UCSD Birds-200-2011\Caltech-UCSD Birds-200-2011'
attributes_url = os.path.join(data_path,"CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
all_class_url = os.path.join(data_path, "CUB_200_2011/classes.txt")

att = get_attributes(attributes_url,ParsingAtt) #获得属性
classname = get_attributes(all_class_url,ParsingClass)#获得类名

word_vectors=torch.tensor(att).to(device)
word_vectors = F.normalize(word_vectors) ## Normalize


vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC.json") #可见类的VC中心文件json file
#保存可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
VC=obj["train"] #获得可见类的中心点

# Obtain the approximated VC of unseen class
vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC_testCenter.json") #可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
test_center=obj["VC"]


VC = VC+test_center   #源域类别中心点和目的域类别聚类中心点

#VC = VC+obj["test"]   #源域类别中心点和目的域类别聚类中心点


VC=torch.tensor(VC)
VC=VC.to(device)
VC=F.normalize(VC)


output_dim=2048
hidden_layers=[2048,2048]


import dgl
G = dgl.DGLGraph()
G.add_nodes(classNum) #生成DGL节点
#G.add_edges([(u, u) for u in range(classNum)]) 
G.add_edges(G.nodes(), G.nodes())
Net = GCN( input_dim, output_dim, hidden_layers,device).to(device)


print('word vectors:', word_vectors.shape)
print('VC vectors:', VC.shape)



#####Parameters
lr=0.0001
wd=0.0005
max_epoch=5000
####

optimizer = torch.optim.Adam(Net.parameters(), lr=lr, weight_decay=wd)
step_optim_scheduler=lr_scheduler.StepLR(optimizer,step_size=4000,gamma=0.1)


method='WDVSc'
#method='VCL'
#method='BMVSc'
#pos=0
for epoch in range(max_epoch + 1):

    s=time.time()
    Net.train()
    step_optim_scheduler.step(epoch)

#    syn_vc = Net(word_vectors) #调用模型，根据属性生成特征
    syn_vc = Net(G,word_vectors) #调用模型，根据属性生成特征

    if method=='VCL': #视觉中心学习，源域训练， 属性特征，与视觉特征MSE，
        loss,_,_=WDVSc(syn_vc,VC,device,classNum,unseenclassnum)  ## Here we have set [--no_use_VSC] to True
    if method=='CDVSc':
        loss=CDVSc(syn_vc,VC,device,classNum,unseenclassnum)
    if method=='BMVSc':
        loss=BMVSc(syn_vc, VC, device,classNum,unseenclassnum)
    if method=='WDVSc':
        loss,_,_=WDVSc(syn_vc,VC,device,classNum,unseenclassnum,no_use_VSC=False)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    e=time.time()
    print("Epoch %d Loss is %.5f Cost Time %.3f mins"%(epoch,loss.item(),(e-s)/60))
#### Training

Net.eval()
output_vectors = Net(G,word_vectors) #调用模型，根据属性生成特征
output_vectors = output_vectors.detach()
        

np.save("Pred_Center.npy",output_vectors.cpu().numpy())



def NN_search(x,center):
    ret=""
    MINI=-1
    for c in center.keys():
        tmp=np.sum((x-center[c])*(x-center[c]))#L2_dis
#        print(c,tmp)
        if MINI==-1:
            MINI=tmp
            ret=c
        if tmp<MINI:
            MINI=tmp
            ret=c
    return ret




centernpy = np.load("Pred_Center.npy")

center=dict(zip(classname,centernpy))#全部中心点

subcenter = dict(zip(classname[-50:],centernpy[-50:]))#

vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC.json") #可见类的VC中心文件json file
#保存可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
VC=obj["train"] #获得可见类的中心点
VCunknown = obj["test"]
allVC = VC+VCunknown #视觉中心点
vccenter = dict(zip(classname,allVC))#全部中心点

cur_root = r'./CUBfeature/'
allacc = []

#for target in classname[:classNum-unseenclassnum]: #遍历未知类的特征数据
for target in classname[classNum-unseenclassnum:]: #遍历未知类的特征数据
    cur=os.path.join(cur_root,target)
    fea_name=""
    url=os.path.join(cur,"ResNet101.json")
    js = json.load(open(url, "r"))
    cur_features=js["features"]

    correct=0
    for fea_vec in cur_features:  #### Test the image features of each class
        fea_vec=np.array(fea_vec)
#        ans=NN_search(fea_vec,center)  # Find the nearest neighbour in the feature space
        ans=NN_search(fea_vec,subcenter)
#        ans=NN_search(fea_vec,vccenter)
        
        if ans==target:
            correct+=1

    allacc.append( correct * 1.0 / len(cur_features) )
    print( target,correct)

print("The final MCA result is %.5f"%(sum(allacc)/len(allacc)))



#测试类别中心点与模型输出的中心点比较
for i,fea_vec in enumerate(VCunknown):  #
    fea_vec=np.array(fea_vec)
    ans=NN_search(fea_vec,center)  # 
    if classname[150+i]!=ans:
        print(classname[150+i],ans)    

#聚类效果
result = {}
for i,fea_vec in enumerate(test_center):  #### Test the image features of each class
    fea_vec=np.array(fea_vec)
    ans=NN_search(fea_vec,vccenter)  # Find the nearest neighbour in the feature space
    classindex = int(ans.split('.')[0])
    if classindex<=150:
        print("聚类错误的类别",i,ans)
    if classindex not in result.keys():
        result[classindex]=i
    else:
        print("聚类重复的类别",i,result[classindex],ans)
for i in range(150,200):
    if i+1 not in result.keys():
        print("聚类失败的类别：",classname[i])
        
    
        