import os
import re
from tqdm import tqdm
from collections import Counter
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import torch
from torch import nn
import numpy as np
import pickle
from model import BiRNN,TextCNN
from preprocess import get_data_iter,glove,load_pretrained_embedding
def evaluate_accuracy(data_iter,net):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y,_a,_b in data_iter:
            if isinstance(net,nn.Module):
                net.eval()
                acc_sum+=(torch.argmax(net(X.to(device)),dim=1)==y.to(device)).float().sum().cpu().item()
            n+=y.shape[0]
        net.train()
        return acc_sum/n
def train(net,train_iter,test_iter,optimizer,num_epochs,device):
    net=net.to(device)
    print("training on",device)
    loss=nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,idx=0.0,0.0,0,0
        for X,y,_a,_b in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.cpu().item()
            train_acc_sum+=(torch.argmax(y_hat,dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            idx+=1
        test_acc=evaluate_accuracy(test_iter,net)
        train_acc=train_acc_sum/n
        train_l=train_l_sum/idx
        print("epoch:{},train loss:{},train acc:{},test acc:{}".format(epoch+1,train_l,train_acc,test_acc))
train_iter,test_iter,vocab=get_data_iter()
##网络实例化
embsize,num_hiddens,num_layers=100,100,2
net=BiRNN(len(vocab),embsize,num_hiddens,num_layers)
##读取预训练权重
embeddings_dict=glove()
embed=load_pretrained_embedding(vocab.get_itos(),embeddings_dict)
net.embedding.weight.data.copy_(embed)
net.embedding.weight.requires_grad=False

lr,num_epochs=0.01,10
optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,net.parameters()),lr=lr)
train(net,train_iter,test_iter,optimizer,num_epochs,'cuda:0')