# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Fri Feb  1 00:07:25 2019
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import models
from torchvision import transforms

#获取模型，如果本地缓存没有，则会自动下载
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.eval()

#在将图像数据输入网络之前，需要对图像进行预处理。
transform = transforms.Compose([
 transforms.Resize(256),#1. 将图像resize到256 x 256
 transforms.CenterCrop(224),#2. 中心裁剪成224 x 224
 transforms.ToTensor(),#3. 转换成Tensor归一化到[0,1]
 transforms.Normalize(#4. 使用均值、方差标准化
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
         )
])


#################################

def preimg(img):                                    #定义图片预处理函数
    if img.mode=='RGBA':                            #兼容RGBA图片
        ch = 4 
        print('ch',ch)
        a = np.asarray(img)[:,:,:3] 
        img = Image.fromarray(a)
    return img

#要预测的图像
img = Image.open('./horse.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()
im =preimg( img )

inputimg = transform(im).unsqueeze(0)#对输入数据进行维度扩展，成为NCHW

#显示用transform转化后的图片
tt = np.transpose(inputimg.detach().numpy()[0], (1, 2, 0))
plt.imshow(tt)
plt.show()


#################
output = model(inputimg)
print("输出结果的形状",output['out'].shape)#[1, 21, 224, 224]
#去掉批次维度，提取结果，形状为(21,224, 224)
output = torch.argmax(output['out'].squeeze(), dim=0).detach().cpu().numpy()

resultclass = set(list(output.flat))
print("所发现的分类：",resultclass)

def decode_segmap(image, nc=21): #定义函数，根据不同分类进行区域染色
  label_colors = np.array([(0, 0, 0),  #定义每个分类对应的颜色
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = np.zeros_like(image).astype(np.uint8)#初始化RGB
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):#根据预测结果进行染色
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  return np.stack([r, g, b], axis=2)#返回结果



rgb = decode_segmap(output)
img = Image.fromarray(rgb)
plt.axis('off')#显示模型的可视化结果
plt.imshow(img)












