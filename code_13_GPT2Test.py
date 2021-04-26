# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:10:34 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练模型（权重）
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


#编码输入
indexed_tokens = tokenizer.encode("Who is Li Jinhong ? Li Jinhong is a")

print( tokenizer.decode(indexed_tokens))

tokens_tensor = torch.tensor([indexed_tokens])#转换为张量

# 加载预训练模型（权重）
model = GPT2LMHeadModel.from_pretrained('gpt2')

#将模型设置为评估模式
model.eval()

tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# 预测所有标记
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# 得到预测的下一词
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)


#生成一段完整的话
stopids = tokenizer.convert_tokens_to_ids(["."])[0] 
past = None
for i in range(100):
    with torch.no_grad():
        output, past = model(tokens_tensor, past=past)
    token = torch.argmax(output[..., -1, :])

    indexed_tokens += [token.tolist()]

    if stopids== token.tolist():
        break
    tokens_tensor = token.unsqueeze(0)
    
sequence = tokenizer.decode(indexed_tokens)

print(sequence)


