"""
Created on Tue Jun  2 05:42:53 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import torch
import torch.nn as nn
#出至： https://github.com/dfdazac/wassdistance/blob/master/layers.py
class SinkhornDistance(nn.Module):    
    def __init__(self, eps, max_iter ,device, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device=device

    def forward(self, x, y):

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        #初始化质量概率
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()
        u = torch.zeros_like(mu)#初始化
        v = torch.zeros_like(nu)

        if self.device is not None: #指派运算硬件
            mu=mu.to(self.device)
            nu=nu.to(self.device)
            u = u.to(self.device)
            v=v.to(self.device)

        thresh = 1e-1 #停止迭代的阀值
        
        C = self._cost_matrix(x, y)  #计算成本矩阵
        # Sinkhorn iterations
        for i in range(self.max_iter):#按照指定迭代次数计算行列归一化
            u1 = u  #保存上一步U值
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            if err.item() < thresh: #如果U值没有在更新，则结束
                break

        U, V = u, v
        #计算耦合矩阵
        pi = torch.exp(self.M(C, U, V))
        #计算最终成本
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost, pi, C

    def M(self, C, u, v):#计算指数空间的耦合矩阵
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
#        return (C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):#计算成本矩阵
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

