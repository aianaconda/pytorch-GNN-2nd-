"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
Created on Tue Mar 19 22:24:58 2019
"""

import torchvision  
import torch#引入PyTorch库
from torch.nn import functional as F
import numpy as np
#引入解释库
from captum.attr import (IntegratedGradients,Saliency,DeepLift,
                         NoiseTunnel, visualization)



#引入本地代码库
from code_10_CNNModel import myConNet,classes,test_loader,imshow,batch_size



network = myConNet()    
    
#使用模型
network.load_state_dict(torch.load( './CNNFashingMNIST.pth'))#加载模型
dataiter = iter(test_loader)
inputs, labels = dataiter.next() #取一批次（10个）样本
print('样本形状：',np.shape(inputs))
print('样本标签：',labels)

imshow(torchvision.utils.make_grid(inputs,nrow=batch_size))
print('真实标签: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(inputs))))
outputs = network(inputs)
_, predicted = torch.max(outputs, 1)


print('预测结果: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(len(inputs))))



ind = 3  #指定分类标签
img = inputs[ind].unsqueeze(0)# 提取单张图片，形状为[1, 1, 28, 28]
img.requires_grad = True
network.eval()


saliency = Saliency(network)
grads = saliency.attribute(img, target=labels[ind].item())
grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


ig = IntegratedGradients(network)
network.zero_grad()
attr_ig, delta  = ig.attribute(img,target=labels[ind], baselines=img * 0, 
                               return_convergence_delta=True )
attr_ig = np.transpose(attr_ig.squeeze(0).cpu().detach().numpy(), (1, 2, 0))



ig = IntegratedGradients(network)
nt = NoiseTunnel(ig)
network.zero_grad()
attr_ig_nt = nt.attribute(img, target=labels[ind],baselines=img * 0, nt_type='smoothgrad_sq',
                                      n_samples=100, stdevs=0.2)
attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))



dl = DeepLift(network)
network.zero_grad()
attr_dl = dl.attribute(img,target=labels[ind], baselines=img * 0)
attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


print('Predicted:', classes[predicted[ind]], 
      ' Probability:', torch.max(F.softmax(outputs, 1)).item())


original_image = np.transpose(inputs[ind].cpu().detach().numpy() , (1, 2, 0))

#显示输入的原始图片
visualization.visualize_image_attr(None, original_image[...,0], 
                      method="original_image", title="Original Image")

#显示Saliency可解释性结果
visualization.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                          show_colorbar=True, title="Overlayed Gradient Magnitudes")

#显示IntegratedGradients可解释性结果
visualization.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                          show_colorbar=True, title="Overlayed Integrated Gradients")
#显示带有NoiseTunnel的IntegratedGradients可解释性结果
visualization.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                             outlier_perc=10, show_colorbar=True, 
                             title="Overlayed IG \n with SmoothGrad Squared")
#显示DeepLift可解释性结果
visualization.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                          title="Overlayed DeepLift")




