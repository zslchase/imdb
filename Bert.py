import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import pickle
from transformers import BertTokenizer,BertModel,AdamW
from preprocess import read_imdb,text_sub
import re
from model import BertClassification
cache_dir='./output/cache'
data_path='./data'

train_data=read_imdb('train')
test_data=read_imdb('test')
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
def preprocess(data):
    input,attention_mask,labels=[],[],[]
    for review,label in tqdm(data):
        encoded_sent=tokenizer.encode_plus(
            text=text_sub(review),
            add_special_tokens=True,
            max_length=500,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input.append(encoded_sent.get('input_ids'))
        attention_mask.append(encoded_sent.get('attention_mask'))
        labels.append(label)
    return input,attention_mask,labels
##分词
if os.path.exists(os.path.join(cache_dir, 'train_processed_bert.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'train_processed_bert.pkl'), 'rb') as f:
        input_ids, attention_masks, labels = pickle.load(f)
else:
    input_ids, attention_masks, labels = preprocess(train_data)
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'train_processed_bert.pkl'), 'wb') as f:
        pickle.dump((input_ids, attention_masks, labels), f, protocol=1)
if os.path.exists(os.path.join(cache_dir, 'test_processed_bert.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'test_processed_bert.pkl'), 'rb') as f:
        test_input_ids, test_attention_masks, test_labels = pickle.load(f)
else:
    test_input_ids, test_attention_masks, test_labels = preprocess(test_data)
    test_input_ids, test_attention_masks, test_labels = torch.tensor(test_input_ids), torch.tensor(test_attention_masks), torch.tensor(test_labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'test_processed_bert.pkl'), 'wb') as f:
        pickle.dump((test_input_ids, test_attention_masks, test_labels), f, protocol=1)
        
train_dataloader = DataLoader(list(zip(input_ids.unbind(0), attention_masks.unbind(0), labels.unbind(0))), batch_size=8, shuffle=True)
test_dataloader = DataLoader(list(zip(test_input_ids.unbind(0), test_attention_masks.unbind(0), test_labels.unbind(0))), batch_size=8, shuffle=True)
                                  
def train(net,optimizer,data_iter,loss,device):
    net.train()
    net=net.to(device)
    l_loss_sum=0
    num_iter=len(data_iter)
    for (seq,mask,label) in tqdm(data_iter,total=len(data_iter)):
        seq,mask,label=seq.to(device),mask.to(device),label.to(device).float()
        y_hat=net(seq,mask)
        l=loss(y_hat,label)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_loss_sum+=l.item()
    train_loss=l_loss_sum/num_iter
    print("train_loss:{}".format(train_loss))

def test(net,data_iter,device):
    net.eval()
    acc=0
    num_iter=len(data_iter)
    with torch.no_grad():
        for (seq,mask,label) in tqdm(data_iter,total=len(data_iter)):
            seq,mask,label=seq.to(device),mask.to(device),label.to(device)
            y_hat=net(seq,mask)
            acc+=(torch.round(y_hat)==label).float().mean().item()
    acc=acc/num_iter
    print("test_acc:{}".format(acc))
    return acc

net=BertClassification(freeze_bert=False)
optimizer=AdamW(net.parameters(),lr=2e-5)
bce_loss_fn = nn.BCELoss()
num_epochs = 2
device='cuda:0'
for i in range(num_epochs):
    print(f'Epoch: {i+1}')
    train(net,optimizer,train_dataloader,bce_loss_fn,device)
    test(net,test_dataloader,device)