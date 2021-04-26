# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:22:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""


from PIL import Image  						#引入基础库
import matplotlib.pyplot as plt
import json
import numpy as np

import torch								#引入PyTorch库
import torch.nn.functional as F
from torchvision import models, transforms #引入torchvision库

model = models.resnet18(pretrained=True) 	#true 代表下载
model = model.eval()

labels_path = 'imagenet_class_index.json'  #处理英文标签
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

def getone(onestr):
    return onestr.replace(',',' ')
with open('中文标签.csv','r+') as f: 		#处理中文标签				
    zh_labels =list( map(getone,list(f))  )
    print(len(zh_labels),type(zh_labels),zh_labels[:5]) #显示输出中文标签

transform = transforms.Compose([			#对图片尺寸预处理
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(						#对图片归一化预处理
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
         )
])

def preimg(img):                              		#定义图片预处理函数
    if img.mode=='RGBA':                    		#兼容RGBA图片
        ch = 4 
        print('ch',ch)
        a = np.asarray(img)[:,:,:3] 
        img = Image.fromarray(a)
    return img

im =preimg( Image.open('book.png') )				#打开图片
transformed_img = transform(im)					#调整图片尺寸

inputimg = transformed_img.unsqueeze(0)			#增加批次维度

output = model(inputimg)						#输入模型
output = F.softmax(output, dim=1)				#获取结果

# 从预测结果中取出前3名
prediction_score, pred_label_idx = torch.topk(output, 3)
prediction_score = prediction_score.detach().numpy()[0] #获取结果概率
pred_label_idx = pred_label_idx.detach().numpy()[0]		 #获取结果的标签id

predicted_label = idx_to_labels[str(pred_label_idx[0])][1]#取出标签名称
predicted_label_zh = zh_labels[pred_label_idx[0] + 1 ] #取出中文标签名称
print(' 预测结果:', predicted_label,predicted_label_zh, 
        '预测分数：', prediction_score[0])

#可视化处理，创建一个1行2列的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
fig.sca(ax1)  				#设置第一个轴是ax1
ax1.imshow(im)  				#第一个子图显示原始要预测的图片

#设置第二个子图为预测的结果，按概率取前3名
barlist = ax2.bar(range(3), [i for i in prediction_score])
barlist[0].set_color('g')  		#颜色设置为绿色

#预测结果前3名的柱状图
plt.sca(ax2)
plt.ylim([0, 1.1])

#竖直显示Top3的标签
plt.xticks(range(3), [idx_to_labels[str(i)][1][:15] for i in pred_label_idx ], rotation='vertical')
fig.subplots_adjust(bottom=0.2)  	#调整第二个子图的位置
plt.show()  							#显示图像 


