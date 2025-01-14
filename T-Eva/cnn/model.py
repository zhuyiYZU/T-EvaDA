import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,
                 class_num=None,
                 embed_size=None,
                 embed_dim=64,
                 kernel_num=128,
                 kernel_size_list=(3, 4, 5),
                 # 将元素置0的概率
                 dropout=0.5):
        
        super(TextCNN, self).__init__()

        # nn.Embedding(embed_size, embed_dim)的意思是创建一个词嵌入模型，
        # embed_size - 词嵌入字典大小，即一个字典里要有多少个词。
        # embed_dim - 每个词嵌入向量的大小。
        self.embedding = nn.Embedding(embed_size, embed_dim)

        # ModuleList:储存不同 module
        self.conv1d_list = nn.ModuleList([
            # embed_dim:词向量维度 kernal_num:卷积核的数量 kernel_size:卷积核大小
            nn.Conv1d(embed_dim, kernel_num, kernel_size)
                for kernel_size in kernel_size_list
        ])

        # 全连接层 kernel_num * len(kernel_size_list):输入张量的大小, class_num:输出张量的大小
        self.linear = nn.Linear(kernel_num * len(kernel_size_list), class_num)

        # 在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果。
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x.shape is (batch, word_nums)

        # after embedding x.shape is (batch, word_nums, embed_dim)
        x = self.embedding(x)
        
        # since the input of conv1d require shape: (batch, in_channels, in_length)
        # here in_channels is embed_dim, in_length is word_nums
        # we should tranpose x into shape: (batch, embed_dim, word_nums)
        # 改变序列
        x = x.transpose(1, 2)

        # after conv1d the shape become: (batch, kernel_num, out_length)
        # here out_length = word_nums - kernel_size + 1
        # relu激活函数
        x = [F.relu(conv1d(x)) for conv1d in self.conv1d_list]

        # pooling apply on 3th dimension, window size is the length of 3th dim
        # after pooling the convert to (batch, kernel_num, 1)
        # squeeze is requred to remove the 3th dimention
        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]

        # shape: (batch, kernel_num * len(kernel_size_list))
        # 在给定维度上对输入的张量序列x进行连接操作。dim:选择扩维
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        
        # shape: (batch, class_num)
        x = self.linear(x)
        
        return F.softmax(x, dim=1)  # 按照行来一行一行做归一化的
