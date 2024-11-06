import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable


# 实现Transformer的整个代码结构。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################ 嵌入层；#################################
class embeddings(nn.Module):
    def __init__(self,d_model:int,vocab:int):
        # 这里embedding层是和Transformer的主体一起训练的，所以不需要加载预训练的词向量；
        # embedding层一共有vocab个词语，每个词语的维度为d_model个，一共是vocab*d_model个数字组成的权重矩阵；
        super(embeddings,self).__init__() 
        self.embedding = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    
    def forward(self,x):
        # 当使用tied-embedding并且对应使用 xavier初始化时，需要乘上维度的平方
        # 这样是保证和position encoding的大小相对一致；
        return self.embedding(x) * math.sqrt(self.d_model)
    

############################ PE位置编码；############################
class Positional_Encoding(nn.Module):
    def __init__(self,d_model:int,dropout:float,len_position=500):
        super(Positional_Encoding,self).__init__()
        
        # 在训练阶段按概率p随即将输入的张量元素随机归零，常用的正则化器，用于防止网络过拟合;
        self.dropout = nn.Dropout(p=dropout)
        # 生成的PE维度为（len_position,d_model）,最大支持len_position长度的句子，每个位置有d_model个特征；
        self.PE = torch.zeros(len_position,d_model)
        self.PE = self.PE.to(device)
        #  从0到len_position-1构造一个向量，再把维度扩展为(len_position,1)；
        position = torch.arange(0.,len_position).unsqueeze(1)
        # 首先实现三角函数的里面部分sin(div_term)和cos(div_term)；
        # $(\frac{1}{10000^{2j/d}})=(10000^{-2j/d}) = \exp(-2j\log(10000)/d)$
        # shape of div_term: (d_model/2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model)))
        # position * div_term进行广播，维度为(len_position,d_model/2)；
        # self.PE[:,0::2]表示从0开始，步长为2，即偶数位置；self.PE[:,1::2]表示从1开始，步长为2，即奇数位置；
        self.PE[:,0::2] = torch.sin(position * div_term) # 偶数位置使用sin编码；
        self.PE[:,1::2] = torch.cos(position * div_term) # 基数位置使用cos编码；
        #插上batch这个维度；
        self.PE = self.PE.unsqueeze(0)
        # self.register_buffer("PE",self.PE)

    def forward(self,x):
        # print("DEBUG:X.SIZE____:",np.shape(x))
        # print("DEBUG:PE.SIZE____:",np.shape(self.PE))
        # 广播，等于每个样本都加上PE；
        # self.PE[:,:x.size(1)]就是保留输入长度的位置编码，多余的部分不要；
        x = x + Variable(self.PE[:, :x.size(1)]).to(x.device) 
        # x = x + self.PE[:,:x.size(1)]
        return self.dropout(x)

# 用来拷贝多个层，用于构建多层网络，方便复用
def clones(module, N):
    # deepcopy是深拷贝，拷贝对象及其子对象，完全拷贝一份新的，避免原对象和拷贝对象共享内存；
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

############################ 多头注意力即多层注意力机制。原先输入的词维度为512的将通过三个投影矩阵投影到更小的维度；############################
class MultiHeadedAttention(nn.Module): 
    def __init__(self, head_num:int, d_model, dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        #在这个实现中，每个词的维度是 d_model，和多头注意力的输出维度一样，即对应讲义中有d = m * d_k；
        self.d_k = d_model // head_num #d_k为输出模型大小的维度；
        self.head_num = head_num #多头的数目；
        self.dropout = dropout
        # 定义四个投影矩阵，分别是W_Q，W_K，W_VW_O，矩阵维度对应讲义中的$(d , (m *d_k))$
        self.Linears = clones(nn.Linear(d_model,d_model),4) 
        self.Attention = None
        self.Dropout = nn.Dropout(p=dropout)
    
    def forward(self,query,key,value,mask=None):
        nbatches = query.size(0)

        seq_q_len = query.size(1)
        seq_k_len = key.size(1)
        seq_v_len = value.size(1)

        if mask is not None:
            mask = mask.unsqueeze(1) # 给mas添加一个维度，并设置其值为1；

        # 分别进行线性变换；
        # print(np.shape(query))
        # 注意这里由于权重矩阵是nn.Linear(d_model,d_model)，因此不需要转置；
        # query * W_q; key * W_k; value * W_v;
        query = self.Linears[0](query)
        key = self.Linears[1](key)
        value = self.Linears[2](value)

        # 重塑维度为（nbatches,seq_len,head_num,d_k），讲义中我们省略了batch维度，这里加上；
        query = query.view(nbatches,seq_q_len,self.head_num,self.d_k)
        key = key.view(nbatches,seq_k_len,self.head_num,self.d_k)
        value = value.view(nbatches,seq_v_len,self.head_num,self.d_k)

        # 将与头有关的维度放在前面，方便后续注意力层进行操作；
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # 经过注意力层，返回softmax(qk/）sqrt(d)*v;
        x,self.attn = Attention(query,key,value,mask=mask,dropout=self.Dropout)
        # 转换维度
        # 使用view、transpose等操作将多头相关的维度进行交换后可能导致张量在内存中不连续。).contiguous()，确保张量在内存中是连续存储的，提高计算效率；
        x = x.transpose(1,2).contiguous() 
        # 维度重塑为（nbatches,seq_len,d_model）；
        x = x.view(nbatches,-1,self.head_num * self.d_k) 
        # W_O输出转换；
        return self.Linears[-1](x) # concat掉。 

############################ 注意力层； ############################
def Attention(query,key,value,mask=None,dropout=None):
    d_k = query.size(-1) # 获取最后一个维度；
    
    # 对其进行相乘；多头的维度保持不变，使用softmax(qk/sqrt(d))*v公式进行计算;
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # print("Shape of Mask:",np.shape(mask))
    # print("Shape of Scores:",np.shape(scores))
    if mask is not None: 
        # 如果有mask就使用，当需要制作解码器的时候使用，即mask中为0的位置，会被填充为 -1e9
        scores = scores.masked_fill_(mask == 0, -1e9) 
    p_attn = F.softmax(scores, dim=-1)  # 获取注意力分数图；

    if dropout is not None: 
        p_attn = dropout(p_attn)
    
    # 返回计算的数值和注意力分数矩阵；
    return torch.matmul(p_attn, value), p_attn 

############################ LayerNorm层； ############################
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        # features=d_model，用来给nn.Parameter生成可训练参数的维度；
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        # 经过多头注意力后，输入x的维度为（nbatches,seq_len,d_model）；
        # print("X_SHAPE_IN_LN:",np.shape(x))
        # 对最后一维，d_moedel求均值，keepdim保持输出维度相同；
        # 矩阵形状为（nbatches,seq_len,1）；
        mean = x.mean(-1,keepdim=True) 
        # eps保持标准差不为零，或者防止过小；
        std = x.std(-1,keepdim=True) + self.eps 
        result = self.a_2 * (x-mean)/std + self.b_2
        # print(result.dtype)
        return result
# 有了LayerNorm层就可以构造sublayerConnection层了：
    
############################ SubLayerConnection层； ############################
# 这里实现的是Pre-LN Transformer结构；
class SubLayerConnection(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(SubLayerConnection,self).__init__()
        self.LayerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # 采用残差结构；
        return x + self.dropout(sublayer(self.LayerNorm(x)))

############################ FFN层； ############################
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_Hidden, dropout=0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear_1 = nn.Linear(d_model,d_Hidden)
        self.linear_2 = nn.Linear(d_Hidden,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

######################## 所有的子层便构建完毕； ###########################
