"""
Created on Fri Feb 14 10:22:46 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

import numpy as np
import os
import torch.utils.data as tordata
from PIL import Image
from tqdm import tqdm
import random

#定义函数加载文件夹的文件名称
def load_data(dataset_path, imgresize, label_train_num, label_shuffle ):
    
    label_str= sorted(os.listdir(dataset_path) )#以人物为标签
    #将不完整的样本忽略，只载入完整样本
    removelist = ['005','026','037','079','109','088','068','048']
    for removename in removelist:
        if removename in label_str:
            label_str.remove(removename)

    print("label_str:",label_str)
    label_index = np.arange( len(label_str) )#序列数组
    
    if label_shuffle:
        np.random.seed(0)
        label_shuffle_index = np.random.permutation( len(label_str)  ) #乱序数组 
        train_list =  label_shuffle_index[0:label_train_num]  
        test_list =   label_shuffle_index[label_train_num: ]  
    else:
        train_list =  label_index[0:label_train_num]  
        test_list =   label_index[label_train_num: ]  
        
    print(train_list,test_list) 
    #加载人物列表中的图片名称
    data_seq_dir,data_label,meta_data = load_dir(dataset_path,train_list,label_str)
    test_data_seq_dir,test_data_label,test_meta_data = load_dir(dataset_path,test_list,label_str)
    
    #将图片文件名称转化为数据集
    train_source = DataSet(data_seq_dir,data_label,meta_data,imgresize)
    #test数据不缓存
    test_source = DataSet(test_data_seq_dir,test_data_label,test_meta_data,imgresize,False)
    
    return train_source, test_source

def load_dir( dataset_path,label_index,label_str):
    data_seq_dir,data_label,meta_data= [],[],[]
    for i_label in label_index:#获取样本个体
        label_path = os.path.join(dataset_path, label_str[i_label]) #拼接目录
        for _seq_type in sorted(os.listdir(label_path)): #获取样本类型，普通条件、穿大衣、携带物品
            seq_type_path = os.path.join(label_path, _seq_type)#拼接目录
            for _view in sorted(os.listdir(seq_type_path)):#获取拍摄角度
                _seq_dir = os.path.join(seq_type_path, _view)#拼接图片目录
                if len( os.listdir(_seq_dir))>0: #有图片
                    data_seq_dir.append(_seq_dir) #图片目录
                    data_label.append( i_label ) #图片目录对应的标签
                    meta_data.append((label_str[i_label],_seq_type,_view) )
                else:
                    print("No files:",_seq_dir)
                    
    return  data_seq_dir,data_label,meta_data    

class DataSet(tordata.Dataset):
    def __init__(self, data_seq_dir,data_label,meta_data,imgresize,cache=True):
        self.data_seq_dir = data_seq_dir  #样本图片文件名
        self.data = [None] * len(self.data_seq_dir) #存放样本图片
        self.cache = cache #缓存标志

        self.meta_data = meta_data #数据的元信息
        self.data_label = np.asarray(data_label) #存放标签
        self.imgresize = int(imgresize) #载入的样本图片大小
        self.cut_padding = int(float(imgresize)/64*10) #对样本图片进行剪辑的大小


    def load_all_data(self): #加载所有数据
        for i in tqdm (range(len(self.data_seq_dir)) ):
            self.__getitem__(i)

    def __loader__(self, path): #读取指定路径的数据并剪辑
        frame_imgs = self.img2xarray( path)/ 255.0
        frame_imgs = frame_imgs[:, :, self.cut_padding:-self.cut_padding]#将宽的前10，后10去掉
        return frame_imgs

    def __getitem__(self, index):#加载指定索引数据
        if self.data[index] is None:#第一次加载
            data = self.__loader__(self.data_seq_dir[index])
        else:
            data = self.data[index]
        if self.cache: #如果需要缓存，则保存到缓存里
            self.data[index] = data
        return data, self.meta_data[index],  self.data_label[index]

    def img2xarray(self, flie_path):#读取指定路径的数据
        frame_list = []  #存放图片数据
        imgs = sorted(list(os.listdir(flie_path)))

        # 将图片读入，并放到数组里
        for _img in imgs:
            _img_path = os.path.join(flie_path, _img)
            if os.path.isfile(_img_path):
                img =np.asarray(Image.open(_img_path).resize( (self.imgresize, self.imgresize) ) )
                if len( img.shape)==3: #加载预处理后的图片   
                    frame_list.append(img[...,0])
                else:
                    frame_list.append(img)
                
        return np.asarray( frame_list,dtype=np.float ) #[帧数，高，宽]

    def __len__(self):#计算数据集长度
        return len(self.data_seq_dir)


#定义采样器
class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size#(标签数，样本数)
        
        self.label_set = list( set(dataset.data_label)) #标签集合

    def __iter__(self):
        while (True):
            sample_indices = []
            label_list = random.sample( self.label_set , self.batch_size[0]) #随机获取指定个数标签
            #从每个人的样本中，随机找出指定个数个文件夹
            for _label in label_list: #按照标签个数循环
                
                data_index = np.where(self.dataset.data_label==_label)[0]

                _index =  np.random.choice(data_index,self.batch_size[1],replace=False)
#                print( "_index",_index)
               
                sample_indices += _index.tolist()
                
            yield np.asarray(sample_indices) 

    def __len__(self):
        return len(self.dataset) #人数*类型*角度

#frame_num 每个样本采集的帧数
def collate_fn_for_train( batch,frame_num): #采样器处理函数
    batch_data,batch_label,batch_meta = [],[],[]
    batch_size = len(batch)  #batch_size
    for i in range(batch_size):
        
        batch_label.append(batch[i][2])
        batch_meta.append(batch[i][1])
        data = batch[i][0]

        if data.shape[0] < frame_num:#帧数少，随机加入几个
            multy = (frame_num-data.shape[0])//data.shape[0]+1 #复制几倍，用于少太多的请况
            choicenum = (frame_num-data.shape[0])%data.shape[0]#额外随机加入的个数
            choice_index =np.random.choice( data.shape[0] ,choicenum,replace=False)
            choice_index = list(range(0,data.shape[0]))*multy+ choice_index.tolist()
        else:    
            choice_index =np.random.choice( data.shape[0] , frame_num,replace=False)

        batch_data.append( data[choice_index]  )

        
    batch = [np.asarray(batch_data), batch_meta, batch_label] 
           
    
    return batch
def collate_fn_for_test( batch,frame_num): #接口模式，取全部的帧
    

    batch_data,batch_label,batch_meta = [],[],[]
    batch_size = len(batch)  #batch_size
    batch_frames = np.zeros(batch_size, np.int)
    for i in range(batch_size):
        
        batch_label.append(batch[i][2])
        batch_meta.append(batch[i][1])
        data = batch[i][0]
        
        if data.shape[0] < frame_num:#帧数少，随机加入几个
            print(batch_meta,data.shape[0] )
            multy = (frame_num-data.shape[0])//data.shape[0]+1
            choicenum = (frame_num-data.shape[0])%data.shape[0]
            choice_index =np.random.choice( data.shape[0] ,choicenum,replace=False)
            choice_index = list(range(0,data.shape[0]))*multy+ choice_index.tolist()
            data = np.asarray(data[choice_index])
        
        batch_frames[i] = data.shape[0]#保证所有的都大于等于frame_num

        batch_data.append( data )

    max_frame = np.max(batch_frames)
    
    batch_data = np.asarray([ np.pad(batch_data[i],((0, max_frame - batch_data[i].shape[0]), (0, 0), (0, 0)),
                               'constant', constant_values=0)
                   for i in range(batch_size)])
    
    batch = [batch_data, batch_meta, batch_label] 
    
    return batch


if __name__ == '__main__':


    pass




    