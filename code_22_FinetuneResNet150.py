# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:07:40 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import glob
import os
import numpy as np#引入基础库
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader

import torchvision.models as models
import torchvision.transforms as transforms

######################################################################
def load_dir(directory,labstart=0,classend=None):#获取所有directory中的所有图片和标签
    #返回path指定的文件夹所包含的文件或文件夹的名字列表
    strlabels = os.listdir(directory)
    #对标签进行排序，以便训练和验证按照相同的顺序进行
    strlabels.sort()
    #######################################################
    if classend is not None:
        strlabels = strlabels[0:classend]
    #########################################################    
    #创建文件标签列表
    file_labels = []
    for i,label in enumerate(strlabels):
        jpg_names = glob.glob(os.path.join(directory, label, "*.jpg"))
        #加入列表
        file_labels.extend(zip( jpg_names,[i+labstart]*len(jpg_names))  )

    return file_labels,strlabels


def load_data(dataset_path):      #定义函数加载文件名称和标签
    sub_dir= sorted(os.listdir(dataset_path) )#跳过子文件夹
    start =1 #none：0
    tfile_labels,tstrlabels=[],['none']
    for i in sub_dir:
        directory = os.path.join(dataset_path, i)
        if os.path.isdir(directory )==False: #只处理目录中的数据
            print(directory)
            continue
        file_labels,strlabels = load_dir(directory ,labstart = start )
        tfile_labels.extend(file_labels)
        tstrlabels.extend(strlabels)
        start  = len(strlabels)
    #理解为解压缩，把数据路径和标签解压缩出来
    filenames, labels=zip(*tfile_labels)
    return filenames, labels,tstrlabels



def default_loader(path):
    return Image.open(path).convert('RGB')

class OwnDataset(Dataset):
    def __init__(self,img_dir, labels, indexlist= None, transform=transforms.ToTensor(),
                 loader=default_loader,cache=True):
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.loader=loader
        self.cache = cache 								#缓存标志
        if indexlist is None:
            self.indexlist = list(range(len(self.img_dir)))
        else:
            self.indexlist = indexlist
        self.data = [None] * len(self.indexlist) 		#存放样本图片
    
    def __getitem__(self, idx):
        if self.data[idx] is None:						#第一次加载
            data = self.loader(self.img_dir[self.indexlist[idx]])
            if self.transform:
                data = self.transform(data)
        else:
            data = self.data[idx]
        if self.cache: 									#保存到缓存里
            self.data[idx] = data
        return data,  self.labels[self.indexlist[idx]]

    def __len__(self):
        return len(self.indexlist)


data_transform = {                      #定义数据的预处理方法
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
def Reduction_img(tensor,mean,std):#还原图片
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])#扩展维度后计算

    
#dataset_path = r'./data/'
#filenames, labels,classes = load_data(dataset_path)   

#dataset_path = r'./data/images' 
dataset_path = r"D:\样本\图片\Caltech-UCSD Birds-200-2011\Caltech-UCSD Birds-200-2011\CUB_200_2011\images"
   
tfile_labels,classes = load_dir(dataset_path,classend = 150) 
filenames, labels=zip(*tfile_labels)
#####################################################


#打乱数组顺序
np.random.seed(0)
label_shuffle_index = np.random.permutation( len(labels)  ) 
label_train_num = (len(labels)//10) *8 
train_list =  label_shuffle_index[0:label_train_num]  
test_list =   label_shuffle_index[label_train_num: ] 
print("label_train_num________________",label_train_num,len(labels),len(classes))

train_dataset=OwnDataset(filenames, labels,train_list,data_transform['train'])
val_dataset=OwnDataset(filenames, labels,test_list,data_transform['val'])


train_loader =DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


#
#sample = iter(train_loader)
#images, labels = sample.next()
#print('样本形状：',np.shape(images))
#print('标签个数：',len(classes))
#
#mulimgs = torchvision.utils.make_grid(images[:10],nrow=10)
#Reduction_img(mulimgs,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#_img= ToPILImage()( mulimgs )
#plt.axis('off')
#plt.imshow(_img)
#plt.show()
#print(','.join('%5s' % classes[labels[j]] for j in range(len(images[:10]))))


############################################

#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_ResNet(classes,pretrained=True,loadfile = None):
    ResNet=models.resnet101(pretrained)# 这里自动下载官方的预训练模型
    if loadfile!= None:
        ResNet.load_state_dict(torch.load( loadfile))	#加载本地模型 
        
    # 将所有的参数层进行冻结
    for param in ResNet.parameters():
        param.requires_grad = False
    # 这里打印下全连接层的信息
#    print(ResNet.fc)
    x = ResNet.fc.in_features #获取到fc层的输入
    ResNet.fc = nn.Linear(x, len(classes)) # 定义一个新的FC层
#    print(ResNet.fc) # 最后再打印一下新的模型
    return ResNet

ResNet=get_ResNet(classes)
ResNet.to(device)

criterion = nn.CrossEntropyLoss()
#指定新加的fc层的学习率
optimizer = torch.optim.Adam([ {'params':ResNet.fc.parameters()}], lr=0.001)


def train(model,device, train_loader, epoch,optimizer):
    model.train()
    allloss = []
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat= model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        allloss.append(loss.item())
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,np.mean(allloss)  ))

def test(model, device, val_loader):
    model.eval()
    test_loss = []
    correct = []
    with torch.no_grad():
        for i,data in enumerate(val_loader):          
            x,y= data
            x=x.to(device)
            y=y.to(device)
            y_hat = model(x)
            test_loss.append( criterion(y_hat, y).item()) # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct.append( pred.eq(y.view_as(pred)).sum().item()/pred.shape[0] )
    print('\nTest set——{}: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
         len(correct),np.mean(test_loss), np.mean(correct)*100 ))

if __name__ == '__main__':

    firstmodepth = './CUB150Res101_1.pth'
    
    
    if os.path.exists(firstmodepth) ==False:
        print("_____训练最后一层________")  
        for epoch in range(1, 2):
            train(ResNet,device, train_loader,epoch,optimizer )
            test(ResNet, device, val_loader )
        # 保存模型
        torch.save(ResNet.state_dict(), firstmodepth)       
    
    
    
    secondmodepth = './CUB150Res101_2.pth'
    optimizer2=optim.SGD(ResNet.parameters(),lr=0.001,momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer2, step_size=2, gamma=0.9)
    
    for param in ResNet.parameters():
        param.requires_grad = True
    
    if os.path.exists(secondmodepth) :
        ResNet.load_state_dict(torch.load( secondmodepth))	#加载本地模型
    else:
        ResNet.load_state_dict(torch.load(firstmodepth))	#加载本地模型    
    print("_____全部训练________")    
    for epoch in range(1, 100):
        train(ResNet,device, train_loader,epoch,optimizer2 )
        if optimizer2.state_dict()['param_groups'][0]['lr']>0.00001:
            exp_lr_scheduler.step()
            print("___lr:" ,optimizer2.state_dict()['param_groups'][0]['lr'] )
        
        test(ResNet, device, val_loader )    
    # 保存模型
    torch.save(ResNet.state_dict(), secondmodepth)  



















