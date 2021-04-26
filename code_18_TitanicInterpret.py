# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:36:39 2019
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

import numpy as np

import torch
import matplotlib.pyplot as plt
from scipy import stats
from captum.attr import IntegratedGradients,LayerConductance,NeuronConductance
from code_05_Titanic import ThreelinearModel,test_features , test_labels,feature_names

 
net = ThreelinearModel()
net.load_state_dict(torch.load('models/titanic_model.pt'))
print("Model Loaded!")

#测试模型
test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
out_probs = net(test_input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))

#####################################
#####################################

#选择并使用解释算法(梯度积分)
ig = IntegratedGradients(net)


test_input_tensor.requires_grad_()#将输入张量设置为可以被求梯度
#利用梯度积分的方法，求出原数据的可解释特征
attr, delta = ig.attribute(test_input_tensor,target=1, return_convergence_delta=True)
attr = attr.detach().numpy()

############################
#可视化
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, rotation='vertical')#wrap=True,

        plt.ylabel('value')
        plt.xlabel(axis_title)
        plt.title(title)
visualize_importances(feature_names, np.mean(attr, axis=0))

#查看某一特征的样本分布
plt.hist(attr[:,1], 100)
plt.title("Distribution of Sibsp Attribution in %d"%len(test_labels))


#bin_means, bin_edges, _ = stats.binned_statistic(test_features[:,1], attr[:,1], statistic='mean', bins=6)
#bin_count, _, _ = stats.binned_statistic(test_features[:,1], attr[:,1], statistic='count', bins=6)
#
#bin_width = (bin_edges[1] - bin_edges[0])
#bin_centers = bin_edges[1:] - bin_width/2
#plt.scatter(bin_centers, bin_means, s=bin_count)
#plt.xlabel("Average Sibsp Feature Value");
#plt.ylabel("Average Attribution");


###########################
#查看每层中，哪些神经元节点发现了有用特征
cond = LayerConductance(net, net.mish1)
cond_vals = cond.attribute(test_input_tensor,target=1)
cond_vals = cond_vals.detach().numpy()
#将第一层的12个节点学习到的内容可视化
visualize_importances(range(12),np.mean(cond_vals, axis=0),title="Average Neuron Importances", axis_title="Neurons")


plt.figure()
plt.hist(cond_vals[:,7], 100);
plt.title("Neuron 7 Distribution in %d"%len(test_labels));


plt.figure()
plt.hist(cond_vals[:,10], 100);
plt.title("Neuron 10 Distribution in %d"%len(test_labels));

################################
#查看每个神经元对属性的关注度
neuron_cond = NeuronConductance(net, net.mish1)
neuron_cond_vals_10 = neuron_cond.attribute(test_input_tensor, neuron_index=10, target=1)
neuron_cond_vals_6 = neuron_cond.attribute(test_input_tensor, neuron_index=6, target=1)

visualize_importances(feature_names, neuron_cond_vals_6.mean(dim=0).detach().numpy(), title="Average Feature Importances for Neuron 6")

visualize_importances(feature_names, neuron_cond_vals_10.mean(dim=0).detach().numpy(), title="Average Feature Importances for Neuron 10")
