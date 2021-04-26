# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:22:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""


import os
import numpy as np
import cv2					# 引入基础模块
from PIL import Image  						#引入基础库

import torch								#引入PyTorch库
import torch.nn.functional as F
from torchvision import models, transforms #引入torchvision库

model = models.resnet18(pretrained=False) 	#true 代表下载
model.load_state_dict(torch.load( 'resnet18-5c106cde.pth'))


in_list= [] # 这里存放所有的输出
def hook(module, input, output):
    in_list.clear()
    for i in range(input[0].size(0)):#批次个数，逐个保存特征
        in_list.append(input[0][i].cpu().numpy())

model.avgpool.register_forward_hook(hook) 


def preimg(img):                              		#定义图片预处理函数
    if img.mode=='RGBA':                    		#兼容RGBA图片
        ch = 4 
        print('ch',ch)
        a = np.asarray(img)[:,:,:3] 
        img = Image.fromarray(a)
    return img
transform = transforms.Compose([			#对图片尺寸预处理
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(						#对图片归一化预处理
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
         )
])
 
photoname = 'bird.jpg' 
im =preimg( Image.open(photoname) )				#打开图片
transformed_img = transform(im)					#调整图片尺寸
inputimg = transformed_img.unsqueeze(0)			#增加批次维度


with torch.no_grad():
    output = model(inputimg)						#输入模型
output = F.softmax(output, dim=1)				#获取结果

# 从预测结果中取出前3名
_, pred_label_idx = torch.topk(output, 3)
pred_label_idx = pred_label_idx.detach().numpy()[0]		 #获取结果的标签id
preindex = pred_label_idx[0]

print(model.fc)
class_weights = list(model.fc.parameters())[0]

conv_outputs = in_list[0]#(512, 7, 7)

output_file = os.path.join('./', f"{preindex}.{photoname}")


# 在输入图上绘制热力图
def plotCMD(photoname, output_file, predictions, conv_outputs):
    img_ori = cv2.imread(photoname)  		# 读取原始测试图片
    if img_ori is None:
        raise ("no file!")
        return

    # conv_outputs的形状为[ 512，7，7]
    cam = conv_outputs.reshape(in_list[0].shape[0],-1)	#cam.shape	(512, 49)
    

    class_weights_w = class_weights[preindex,:].view(1,class_weights.shape[1])

    
    
    class_weights_w = class_weights_w.detach().numpy()
    cam = class_weights_w @  cam 			# 两个矩阵相乘cam.shape	(1, 49)
    cam = np.reshape(cam, (7, 7))  			# 矩阵变成7*7大小
    cam /= np.max(cam)  				# 归一化到[0 1]
    # 特征图变到原始图片大小
    cam = cv2.resize(cam, (img_ori.shape[1], img_ori.shape[0]))  
    # 绘制热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  
    heatmap[np.where(cam < 0.2)] = 0  		# 热力图阈值0.2
    img = heatmap * 0.5 + img_ori  			# 在原影像图上叠加热力图
    cv2.imwrite(output_file, img)  			# 保存图片
plotCMD(photoname,output_file, preindex,conv_outputs )


