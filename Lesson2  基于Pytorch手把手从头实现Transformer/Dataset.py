import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import nltk
import jieba
import Encoder_Decoder as ED
import numpy as np

#下载后可以注释
# 如果网络问题下载不方便，可以在https://www.nltk.org/nltk_data/下载punkt和punkt_tab并解压
# 将解压后的文件夹放入nltk路径，要查看自己的nltk路径可以输入nltk.data.path
# 注意，查到路径后，是要放入路径的Tokenizer文件夹
# 例如你查到的路径是'/home/xxx/anaconda3/envs/llm/nltk_data',那么放入路径应该是/home/xxx/anaconda3/envs/llm/nltk_data/tokenizers
#nltk.download('punkt')



# # 检查punkt是否加载成功
# try:
#     nltk.data.find('tokenizers/punkt/english.pickle')
#     print("punkt 数据加载成功")
# except LookupError:
#     print("punkt 数据未找到")

# 定义一个函数来读取文件的前N行
def read_first_n_lines(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [next(file).strip() for _ in range(n)]
    return lines

def BuildDataset(num:int):
    # 使用这个函数来读取英文和中文文件的前num行;
    english_sentences = read_first_n_lines('./dataset/english.txt', num)
    chinese_sentences = read_first_n_lines('./dataset/chinese.txt', num)
    # english_sentences = read_first_n_lines('./dataset/english_train.txt', num)
    # chinese_sentences = read_first_n_lines('./dataset/chinese_train.txt', num)
    print('--build Dataset--')
    return english_sentences,chinese_sentences

# 分别对英文和中文进行分词：
def BuildTokens(english_sentences,chinese_sentences):
    eng_tokens = [word_tokenize(sentence.lower()) for sentence in english_sentences]
    chi_tokens = [list(jieba.cut(sentence)) for sentence in chinese_sentences]
    print('--build Tokens--')
    return eng_tokens,chi_tokens

def build_vocab(tokenized_sentences):
    # Counter可以用于统计各个分词出现的频率，但在这里我们只统计词的索引，为每个分词对应一个唯一数字；
    vocab = Counter() 
    for sentence in tokenized_sentences:
        vocab.update(sentence) # 统计分词；
    vocab = {word: (idx + 3) for idx, word in enumerate(vocab)}
    vocab['<pad>'] = 0 # 填充字符；
    vocab['<sos>'] = 1 # 起始标识符；
    vocab['<eos>'] = 2 # 最终标识符；
    print('--build vocab--')
    return vocab # 词表

# 构建词表；
def BuildVocab(eng_tokens,chi_tokens):
    english_vocab = build_vocab(eng_tokens)
    chinese_vocab = build_vocab(chi_tokens)
    return english_vocab,chinese_vocab

# 把分词后的token句子转换为数值化的句子；
# 注意把sos，eos加入
def numericalize(tokens, vocab):
    result = []
    for sentence in tokens:
        tem = [vocab['<sos>']] # 添加起始字符；
        tem_1 = [vocab[word] for word in sentence if word in vocab]
        tem = tem + tem_1
        tem.append(vocab['<eos>']) # 添加终止字符；
        result.append(tem)
    # 把所有句子填充到相同长度，此处没有截断，即把所有句子填充到最长的句子长度；
    result = pad_sequence([torch.LongTensor(np.array(sentence)) for sentence in result],
                        batch_first=True, 
                        padding_value=vocab['<pad>'])
    return result

def BuildNumerical(eng_tokens, english_vocab,chi_tokens, chinese_vocab):
    eng_numerical = numericalize(eng_tokens, english_vocab)
    chi_numerical = numericalize(chi_tokens, chinese_vocab)
    # print(eng_numerical)
    print('--build numerical--')
    return eng_numerical,chi_numerical

# 汇总前面操作的函数，返回两个数值化后的数据集和两张词表；
def BuildData(num:int):
    english_sentences,chinese_sentences = BuildDataset(num)
    eng_tokens,chi_tokens = BuildTokens(english_sentences,chinese_sentences)
    english_vocab,chinese_vocab = BuildVocab(eng_tokens,chi_tokens)
    eng_numerical,chi_numerical = BuildNumerical(eng_tokens, english_vocab,chi_tokens, chinese_vocab)
    return eng_numerical,chi_numerical,english_vocab,chinese_vocab

class TranslationDataset(Dataset):
    def __init__(self, src_numerical, tgt, tgt_numerical, src_vocab, tgt_vocab,pad=0):
        # src_numerical, # 源语言序列的数值化表示；
        # tgt, # 目标语言序列的数值化表示；
        # tgt_numerical, # 同上
        # src_vocab, # 源语言词表-英文；
        # tgt_vocab # 目标语言词表-中文；
        super(TranslationDataset,self).__init__()
        self.src = src_numerical # 获取源文本的数值化表示；
        # 生成源语言序列的掩码,表示每个位置是否不是填充值 <pad>
        # .unsqueeze(-2)：在第二个维度（倒数第二个维度）添加一个维度
        # 使得掩码的形状变为 (num_sentences, 1, max_src_length)。
        self.src_mask = (src_numerical != pad).unsqueeze(-2) 
        print("Shape of Mask in dataset:",np.shape(self.src_mask))
        self.groundTruth = tgt_numerical # 获取目标文本的预测真值；
        self.tgt_mask = None
        if tgt is not None: 
            # 下面的做法的目的是为了使用（src + tgt) 来预测(tgt_y)；
            self.tgt = tgt[:, :-1] # 目标序列的输入，去掉最后一个词
            self.tgt_y = tgt[:, 1:] # 目标序列的输出，去掉第一个词
            # 创造掩码，用来隐藏padding和未来的单词，以防止模型在训练时看到未来的信息。
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # self.ntokens用来计数，来记录self.tgt_y中非填充元素的总数（只有填充元素可以计算Loss）.
            self.ntokens = (self.tgt_y != pad).sum()
        
        self.src_vocab = src_vocab # 原词表；
        self.tgt_vocab = tgt_vocab # 目标词表；

    # 创造掩码的函数，由于不需要调用类的实例属性和方法，可以看到没有self参数；
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 生成掩码矩阵，防止信息泄露；
        final_mask = ED.subsequent_mask(tgt.size(-1))
        # 将 final_mask 的数据类型转换为与 tgt_mask 相同，确保它们可以进行逻辑与操作。
        final_mask = final_mask.type_as(tgt_mask.data) # 转换类型；
        # 填充掩码（padding）和后续掩码（下三角）合并；
        final_mask = tgt_mask & Variable(final_mask) # 使得mask不受梯度影响；
        # final_mask = tgt_mask & final_mask
        # 用上述方法返回的掩码为每个句子的Attention Mask（前面描述的那种）。
        return final_mask

    def __len__(self): # 获取数据长度；
        return len(self.src)

    def __getitem__(self, idx): # 获取源文本和目标文本；
        src_sentence = self.src[idx]
        tgt_sentence = self.groundTruth[idx]
        return { # 返回：原序列；原序列掩码；目标序列；目标序列掩码；预测目标序列；真值；
            'src': src_sentence.type(torch.long),
            # 'src_mask':Variable(self.src_mask[idx]),#掩码的形状变为 (num_sentences, 1, max_src_length)
            'src_mask':self.src_mask[idx],#掩码的形状变为 (num_sentences, 1, max_src_length)
            'tgt': self.tgt[idx].type(torch.long),# 目标语言的输入序列，即目标句子的前 n-1 个词
            'tgt_mask': self.tgt_mask[idx], #掩码的形状变为 (num_sentences, 1, max_src_length)
            'tgt_y': self.tgt_y[idx].type(torch.long),# 目标语言的输出序列，即目标句子的后 n-1 个词
            'ntokens':self.ntokens, # 非填充值的标记数量，用于损失计算的归一化。
            'gt': tgt_sentence.type(torch.long)
        }

def BuildDataLoader(num:int):    
    eng_numerical,chi_numerical,english_vocab,chinese_vocab = BuildData(num)
    dataset = TranslationDataset(eng_numerical, # src序列；
                                 chi_numerical, # tgt序列；
                                 chi_numerical, # 目标文本；
                                 english_vocab, # 英语词表；
                                 chinese_vocab) # 中文词表；
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, english_vocab, chinese_vocab

# for batch in dataloader:
#     src = batch['src']
#     tgt = batch['tgt']

# a,b,c,d = BuildData()
# print(c.data == 0)
