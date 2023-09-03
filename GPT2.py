from transformers import GPT2Tokenizer,GPT2Config,GPT2ForSequenceClassification
from preprocess import read_imdb,text_sub
import torch 
from torch import nn
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,Dataset
import json
import os
from tqdm import tqdm
import pickle
import re
#读取数据
cache_dir='./output/cache'
data_path='./data'
train_data=read_imdb('train')
test_data=read_imdb('test')
#分词
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side='left'
tokenizer.pad_token=tokenizer.eos_token
def preprocess(data,max_len):
    input,labels=[],[]
    for review,label in tqdm(data):
        text=tokenizer(review,return_tensors='pt',padding='max_length',max_length=max_len,truncation=True)
        input.append(text.get('input_ids').tolist())
        labels.append(label)
    return input,labels

if os.path.exists(os.path.join(cache_dir, 'train_processed_gpt2.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'train_processed_gpt2.pkl'), 'rb') as f:
        input_ids, labels = pickle.load(f)
else:
    input_ids,  labels = preprocess(train_data,max_len=400)
    input_ids,labels = torch.tensor(input_ids),torch.tensor(labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'train_processed_gpt2.pkl'), 'wb') as f:
        pickle.dump((input_ids, labels), f, protocol=1)
if os.path.exists(os.path.join(cache_dir, 'test_processed_gpt2.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'test_processed_gpt2.pkl'), 'rb') as f:
        test_input_ids, test_labels = pickle.load(f)
else:
    test_input_ids, test_labels = preprocess(test_data,max_len=400)
    test_input_ids, test_labels = torch.tensor(test_input_ids),torch.tensor(test_labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'test_processed_gpt2.pkl'), 'wb') as f:
        pickle.dump((test_input_ids, test_labels), f, protocol=1)

train_dataloader = DataLoader(list(zip(input_ids.unbind(0), labels.unbind(0))), batch_size=8, shuffle=True)
test_dataloader = DataLoader(list(zip(test_input_ids.unbind(0), test_labels.unbind(0))), batch_size=8, shuffle=True)
#模型实例化
config=GPT2Config.from_pretrained('gpt2',num_labels=2)
model=GPT2ForSequenceClassification.from_pretrained('./gpt',config=config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
optimizer = optim.AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)
loss=nn.CrossEntropyLoss()
def train(model,data_iter, optimizer, loss,device):
    model=model.to(device)
    model.train()
    l_sum,num_iter=0,len(data_iter)
    for (seq,label) in tqdm(data_iter, total=len(data_iter)):
        seq,label=seq.to(device),label.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        seq=seq.squeeze(1)
        logits=model(seq).logits
        l=loss(logits,label)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # y_hat=torch.argmax(logits,dim=-1).item()
        l_sum+=l.item()
    train_loss=l_sum/num_iter
    print("train_loss:{}".format(train_loss))
def test(model,data_iter, loss,device):
    model.eval()
    acc,l_sum,num_iter=0,0,len(data_iter)
    with torch.no_grad():
        for (seq,label) in tqdm(data_iter, total=len(data_iter)):
            seq,label=seq.to(device),label.to(device)
            seq=seq.squeeze(1)
            logits=model(seq).logits
            l=loss(logits,label)
            y_hat=torch.argmax(logits,dim=-1)
            l_sum+=l.item()
            acc+=(y_hat==label).float().mean().item()
    train_loss=l_sum/num_iter
    acc=acc/num_iter
    print("test_loss:{},test_acc:{}".format(train_loss,acc))        
        
num_epochs=4
device='cuda:0'
for epoch in range(num_epochs):
    print("epoch:{}".format(epoch+1))
    train(model,train_dataloader,optimizer,loss,device)
    test(model,test_dataloader,loss,device)