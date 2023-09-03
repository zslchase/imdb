from torch import nn
import torch
import numpy as np
import math
import re
from random import *
import numpy as np
import torch.optim as optim
#--------------------------BiRNN--------------------------------------------------
class BiRNN(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers):
        super(BiRNN,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.encoder=nn.LSTM(embed_size,num_hiddens,num_layers,bidirectional=True)
        self.decoder=nn.Linear(4*num_hiddens,2)
    def forward(self,inputs):
        embeddings=self.embedding(inputs.permute(1,0))
        outputs,_=self.encoder(embeddings)
        encoding=torch.cat((outputs[0],outputs[-1]),-1)
        out=self.decoder(encoding)
        return out

#--------------------------Rnn--------------------------------------------------
#初始化模型参数
def get_params(embedding_size,hidden_size,output_size,device):
    def _one(shape):
        ts=torch.tensor(np.random.normal(0,0.01,size=shape),device=device,dtype=torch.float32)
        return nn.Parameter(ts,requires_grad=True)
    #隐藏层参数
    W_xh=_one((embedding_size,hidden_size))
    W_hh=_one((hidden_size,hidden_size))
    b_h=nn.Parameter(torch.zeros(hidden_size,device=device),requires_grad=True)
    #输出层参数
    W_hq=_one((hidden_size,output_size))
    b_q=nn.Parameter(torch.zeros(output_size,device=device),requires_grad=True)
    return nn.ParameterList([W_xh,W_hh,b_h,W_hq,b_q])
#初始化隐藏层状态
def init_rnn_state(batch_size,hidden_size,device):
    return (torch.zeros((batch_size,hidden_size),device=device),)
#模型定义
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.matmul(X,W_xh)+torch.matmul(H,W_hh)+b_h)
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
        return torch.cat(outputs,dim=0),(H,)
class MyBiRNN(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers):
        super(MyBiRNN,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.encoder=nn.LSTM(embed_size,num_hiddens,num_layers,bidirectional=True)
        self.decoder=nn.Linear(4*num_hiddens,2)
    def forward(self,inputs):
        embeddings=self.embedding(inputs.permute(1,0))
        outputs,_=self.encoder(embeddings)
        encoding=torch.cat((outputs[0],outputs[-1]),-1)
        out=self.decoder(encoding)
        return out
    
#--------------------------TextCNN--------------------------------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
    
#--------------------------bert--------------------------------------------------


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)  

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model) 
        self.pos_embed = nn.Embedding(max_len+2, d_model)  

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos.to('cuda:0'))
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : 
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        self.ll=nn.LayerNorm(d_model)
        self.d_k=d_k
        self.d_v=d_v
        self.n_heads=n_heads
        self.d_model=d_model
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask,self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)
        return self.ll(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,n_layers,d_k,d_v,n_heads,d_ff):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size,d_model,max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_k,d_v,n_heads,d_ff) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids,masked_pos):
        #inputs :[batch_size,seq_len]
        #masked_pos:[batch_size,max_pred]
        output = self.embedding(input_ids) #output: [batch_size, len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        #enc_self_attn_mask:[batch_size,len_k,len_q]
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf

#--------------------------lstm--------------------------------------------------
class lstm(nn.Module):
    def __init__(self,input_size,num_hiddens,batch_size,device):
        super(lstm,self).__init__()
        def Par():
            return (nn.Linear(input_size,num_hiddens),
                   nn.Linear(num_hiddens,num_hiddens))
        torch.Tensor
        self.W_xi,self.W_hi=Par() #输入门参数
        self.W_xf,self.W_hf=Par() #遗忘门参数
        self.W_xo,self.W_ho=Par() #输出门参数
        self.W_xc,self.W_hc=Par() #候选记忆元参数

        # #附加梯度
        # self.parameters=self.W_xi,self.W_hi,self.b_i,self.W_xf,self.W_hf,self.b_f,self.W_xo,self.W_ho,self.b_o,self.W_xc,self.W_hc,self.b_c
        self.batch_size=batch_size
        self.num_hiddens=num_hiddens
        self.device=device

    def init_lstm_state(self):
        return (torch.zeros((self.batch_size,self.num_hiddens),device=self.device), torch.zeros((self.batch_size,self.num_hiddens),device=self.device))
    
    def forward(self,inputs,state=None):
        if state==None:
            H,C=self.init_lstm_state()
        else:
            H,C=state
        outputs=[]
        for X in inputs:
            I = torch.sigmoid(self.W_xi(X) + self.W_hi(H))
            F = torch.sigmoid(self.W_xf(X) + self.W_hf(H))
            O = torch.sigmoid(self.W_xo(X) + self.W_ho(H))
            C_tilda = torch.tanh(self.W_xc(X) + self.W_hc(H))
            C=F*C+I*C_tilda
            outputs.append(H)
        return outputs,(H,)
    
class mylstm(nn.Module):
    def __init__(self,vocab_size,num_hiddens,embed_size,batch_size,device):
       super(mylstm,self).__init__()
       self.embedding=nn.Embedding(vocab_size,embed_size)
       self.lstm=lstm(embed_size,num_hiddens,batch_size,device)
       self.decoder=nn.Linear(num_hiddens,2)

    def forward(self,X):
        X_embed=self.embedding(X.permute(1,0))#X_embed：[seq_len x batch_size x emded_size]
        outputs,(H,)=self.lstm(X_embed)#outputs：[seq_len x batch_size x num_hiddens]
        encodings=outputs[-1]
        outputs=self.decoder(encodings)
        return outputs