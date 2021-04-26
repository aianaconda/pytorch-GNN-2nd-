"""
Created on Sat Apr 11 08:03:20 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.BatchNorm = nn.BatchNorm2d(out_channels )

    def forward(self, x):
        x = self.conv(x)
        x =x *( torch.tanh(F.softplus(x)))
        return self.BatchNorm (x)



class GaitSetNet(nn.Module):
    def __init__(self, hidden_dim,frame_num):
        super(GaitSetNet, self).__init__()
        self.hidden_dim = hidden_dim

        cnls = [1,32, 64, 128]
        self.set_layer1 = BasicConv2d(cnls[0], cnls[1], 5, padding=2)
        self.set_layer2 = BasicConv2d(cnls[1], cnls[1], 3, padding=1)
        self.set_layer1_down = BasicConv2d(cnls[1], cnls[1], 2,stride = 2)
        
        self.set_layer3 = BasicConv2d(cnls[1], cnls[2], 3, padding=1)        
        self.set_layer4 = BasicConv2d(cnls[2], cnls[2], 3, padding=1)
        self.set_layer2_down = BasicConv2d(cnls[2], cnls[2], 2,stride = 2)
        self.gl_layer2_down = BasicConv2d(cnls[2], cnls[2], 2,stride = 2)
        
        self.set_layer5 = BasicConv2d(cnls[2], cnls[3], 3, padding=1)
        self.set_layer6 = BasicConv2d(cnls[3], cnls[3], 3, padding=1)

        self.gl_layer1 = BasicConv2d(cnls[1], cnls[2], 3, padding=1)
        self.gl_layer2 = BasicConv2d(cnls[2], cnls[2], 3, padding=1)
        self.gl_layer3 = BasicConv2d(cnls[2], cnls[3], 3, padding=1)
        self.gl_layer4 = BasicConv2d(cnls[3], cnls[3], 3, padding=1)
        
        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

    def frame_max(self, x,n):
        return torch.max(x.view(n,-1,x.shape[1],x.shape[2],x.shape[3]), 1)[0]

    def forward(self, xinput):#定义前向处理方法
        n= xinput.size()[0]#形状为[batch,帧数，高，宽]
        x = xinput.reshape(-1, 1, xinput.shape[-2],xinput.shape[-1])
        del xinput
        
        x = self.set_layer1(x)
        x = self.set_layer2(x)
        x = self.set_layer1_down(x)
        gl = self.gl_layer1(self.frame_max(x,n)) #将每一层的帧取最大值
        
        gl = self.gl_layer2(gl)
        gl = self.gl_layer2_down(gl)
        x = self.set_layer3(x)
        x = self.set_layer4(x)
        x = self.set_layer2_down(x)
        
        gl = self.gl_layer3(gl + self.frame_max(x,n))
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x,n)
        gl = gl + x

        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
      
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()#62 n c
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()

        return feature


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.hard_or_full = hard_or_full

    def forward(self, feature, label):#[62, 32, 256]  [62, 32])
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).view(-1)#[62, 32, 32]展开[63488]
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).view(-1)


        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)#[62] 所有的平均距离
        dist = dist.view(-1)#[62, 32, 32]展开[63488]
        # hard
        #[62, 32] 每个样本与其它同类样本中找到距离最大的那个
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)#要让间隔最小化，到0为止

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)#[62]

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)#[62, 32, 8, 1]
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)#[62, 32, 1, 24]
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)#[62, 32*8*24]

        full_loss_metric_sum = full_loss_metric.sum(1)#计算[62]中每个loss的和
        full_loss_num = (full_loss_metric != 0).sum(1).float()#计算[62]中每个loss的个数


        full_loss_metric_mean = full_loss_metric_sum / full_loss_num#计算平均值
        full_loss_metric_mean[full_loss_num == 0] = 0 #将无效值设为0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num#,loss

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)#[62, 32]
        #dist [62, 32, 32]
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist


def ts2var( x):
    return autograd.Variable(x).cuda()

def np2var(x):
    return ts2var(torch.from_numpy(x))