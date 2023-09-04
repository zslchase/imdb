import os
import pickle
from tqdm import tqdm
import re
from collections import Counter,OrderedDict
from tkinter import _flatten
from torchtext.vocab import vocab
from random import *
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn,optim
from model import BERT
from preprocess import get_data_iter
cache_dir='./output/cache'
data_path='./data'
def test(net,data_iter,device):
    net.eval()
    acc=0
    num_iter=len(data_iter)
    with torch.no_grad():
        for input_ids,labels,masked_pos,masked_tokens in data_iter:
            input_ids,labels,masked_pos,masked_tokens=input_ids.to(device),labels.to(device),masked_pos.to(device),masked_tokens.to(device)
            logits_lm, logits_clsf = model(input_ids,  masked_pos)
            y_hat=logits_clsf.max(1)[1].data
            acc+=(y_hat==labels).float().mean().item()
    acc=acc/num_iter
    print("test_acc:{}".format(acc))
    return acc
train_iter,test_iter,Vocab=get_data_iter()
vocab_size,d_model,max_len,n_layers,d_k,d_v,n_heads,d_ff=len(Vocab),128,500,4,64,64,4,256
num_epochs=5
model = BERT(vocab_size,d_model,max_len,n_layers,d_k,d_v,n_heads,d_ff)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device='cuda:0'
print("Begin training...")
model=model.to(device)
for epoch in range(num_epochs):
    l_loss_sum,num_iter=0,len(train_iter)
    for input_ids,labels,masked_pos,masked_tokens in tqdm(train_iter,total=len(train_iter)):
        input_ids,labels,masked_pos,masked_tokens=input_ids.to(device),labels.to(device),masked_pos.to(device),masked_tokens.to(device)
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids,  masked_pos)
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, labels) # for sentence classification
        loss = loss_lm + loss_clsf

        loss.backward()
        optimizer.step()
        l_loss_sum+=loss.item()

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(l_loss_sum/num_iter))
    test(model,test_iter,'cuda:0')
