# 可能用到的原始模块；
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

# 自己定义的各类函数；
import Loss
import Layers as L
import Encoder_Decoder as ED
import Dataset as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 首先，获得数据集、源词表、目标词表；
DataLoader,src_vocab,tgt_vocab = Data.BuildDataLoader(1000)

# 获得模型：
Transformer = ED.make_model(src_vocab=len(src_vocab),# 源词表大小；
                            tgt_vocab=len(tgt_vocab),# 目标词表大小；
                            N=2, # Encoder层和Decoder层各两层；
                            d_model=512,
                            d_ff=2048, # FFN层；
                            h=8) # 多头；

Transformer.to(device)

# 获得模型的优化器：
opt = Loss.get_std_opt(Transformer)

# 获得模型的损失函数；
lossfun = Loss.LabelSmoothingKLDivLoss(size=len(tgt_vocab),
                                    padding_idx=0,
                                    smoothing=0.1).to(device)

# 训练一个批次；
def run_epoch(aepoch, model, opt, lossfun):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i,batch in enumerate(DataLoader):
        batch = {key: value.to(device) for key, value in batch.items()}
        # 获得模型的输出；
        output = model.forward(batch['src'], 
                            batch['tgt'], 
                            batch['src_mask'], 
                            batch['tgt_mask'])  
        # output = F.softmax(output)
        # 整理output和真值为可以送进Loss里训练的模样；
        # 整理后output, shape:batch_size*seq_len,vocab_size
        output = output.contiguous().view(-1, output.size(-1)) 
        # 整理后真值，shape:batch_size*seq_len,没有词表。因为值是词的索引
        target = batch['tgt_y'].contiguous().view(-1)

        # 计算Loss;
        loss = lossfun(output, target)
        loss = loss / batch['ntokens'][0]

        # 反向传播；
        loss.backward()
        if opt is not None:
            opt.step()
            opt.optimizer.zero_grad()

        # 获得loss;
        mean_ntokens = torch.mean(batch['ntokens'].type(torch.float)).detach().cpu()  # Move to CPU and detach
        loss = loss.data.item() * mean_ntokens.item()  # Convert mean_ntokens to a scalar

        # total_loss += loss.detach().numpy()
        total_loss += loss
        # total_tokens += mean_ntokens.numpy()
        total_tokens += mean_ntokens.item() 
        tokens += mean_ntokens.item()

        # 输出log;
        if i % 50 == 1:
            elapsed = time.time() - start
            print ('epoch step: {}:{} Loss: {}/{}, tokens per sec: {}/{}'
                    .format(aepoch, i, loss, batch['ntokens'][0], 
                    tokens, elapsed))
            start = time.time()
            tokens = 0
            
    # 返回这一轮Loss的平均损失；
    return total_loss / total_tokens

def __main__():
    for epoch in range(10):
        print ('epoch={}, training...'.format(epoch))
        Transformer.train()
        # 设置模型进入训练模式;
        
        run_epoch(epoch, Transformer, opt, lossfun)
        # 重新构造一批数据，并执行训练;
        
        Transformer.eval()
        print ('evaluating...')
        print(run_epoch(epoch, Transformer, opt, lossfun))

    torch.save(Transformer.state_dict(),'./model.pt')

if __name__==__main__():
    __main__()