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
import numpy as np
cache_dir='./output/cache'
data_path='./data'
def read_imdb(type_):
    if os.path.exists(os.path.join(cache_dir,f'{type_}.pkl')):
        print('Cache found')
        with open(os.path.join(cache_dir,f'{type_}.pkl'),'rb') as f:
            data=pickle.load(f)
        return data
    data=[]
    for label in ['pos','neg']:
        folder_name=os.path.join(data_path,type_,label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name,file),'r',encoding='utf-8') as f:
                review=f.read().replace('\n',' ').lower()
                review=text_sub(review)
                data.append((review,1 if label =='pos' else 0))
    with open(os.path.join(cache_dir,f'{type_}.pkl'),'wb') as f:
        pickle.dump(data,f)
def text_sub(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokens(data):
    texts,labels=[],[]
    input,attention_mask=[],[]
    for review,label in tqdm(data):
        text=text_sub(review).split(' ')
        texts.append(text)
        labels.append(label)
    return texts,labels

def build_vocab(tokens):
    fre_dic=OrderedDict(sorted(Counter(_flatten(tokens)).items(),key=lambda x:x[1],reverse=True))
    word_vocab=vocab(fre_dic,min_freq=2,specials=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'])
    word_vocab.set_default_index(word_vocab['[UNK]'])
    return word_vocab
class Mydataset(Dataset):
    def __init__(self,comments,label,masked_pos,masked_tokens):
        self.comments=torch.tensor(comments,dtype=torch.int32)
        self.labels=torch.tensor(label,dtype=torch.long)
        self.masked_pos=torch.tensor(masked_pos,dtype=torch.int64)
        self.masked_tokens=torch.tensor(masked_tokens,dtype=torch.long)
    def __getitem__(self, index):
        return self.comments[index],self.labels[index],self.masked_pos[index],self.masked_tokens[index]
    def __len__(self):
        return len(self.comments)
def mask_data(texts,Vocab,max_len,max_pred,labels,train=True):
    max_len+=2
    input,m_pos,m_tokens=[],[],[]
    for text in texts:
        input_ids=[Vocab['[CLS]']]+text+[Vocab['[SEP]']]
        if len(input_ids) > max_len:
            input_ids=input_ids[:max_len]
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
        masked_idx=[i for i ,token in enumerate(input_ids) if token !=Vocab['[CLS]'] and token !=Vocab['SEP']]
        shuffle(masked_idx)
        masked_tokens,masked_pos=[],[]
        for pos in masked_idx[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = Vocab['[MASK]'] # make mask
            elif random() < 0.1:  # 10%
                index = randint(0, len(Vocab) - 1) # random index in vocabulary
                input_ids[pos] = index # replace
        # Zero Paddings
        if len(input_ids) < max_len:
            n_pad = max_len - len(input_ids)
            input_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        input.append(input_ids)
        m_pos.append(masked_pos)
        m_tokens.append(masked_tokens)
    datas=Mydataset(input,labels,m_pos,m_tokens)
    data_iter=DataLoader(datas,batch_size=64,shuffle=train)
    return data_iter


def get_data_iter():
    train_data=read_imdb('train')
    test_data=read_imdb('test')
    train_text,train_label=tokens(train_data)
    test_text,test_label=tokens(test_data)
    texts=train_text+test_text
    if os.path.exists(os.path.join(cache_dir,f'vocab.pkl')):
        print('Cache found')
        with open(os.path.join(cache_dir,f'vocab.pkl'),'rb') as f:
            Vocab=pickle.load(f)
    else:
        Vocab=build_vocab(texts)
        with open(os.path.join(cache_dir,f'vocab.pkl'),'wb') as f:
            pickle.dump(Vocab,f)
    train_text_num=[Vocab(i) for i in train_text]
    test_text_num=[Vocab(i) for i in test_text]

    train_iter=mask_data(train_text_num,Vocab,500,max_pred=10,labels=train_label,train=True)
    test_iter=mask_data(test_text_num,Vocab,500,10,test_label,train=False)
    return train_iter,test_iter,Vocab

def glove():
    embeddings_dict={}
    with open("./data/glove.6B.100d.txt","r",encoding="utf-8") as f:
        c=1
        for line in f:
            seq=line.split()
            word=seq[0]
            vector=np.asarray(seq[1:],"float32")
            vector=torch.tensor(vector,dtype=torch.float32)
            embeddings_dict[word]=vector
    return embeddings_dict

def load_pretrained_embedding(words,embeding_dict):
    embed=torch.zeros(len(words),100)
    oov_count=0
    for i,word in enumerate(words):
        try:
            embed[i,:]=embeding_dict[words[i]]
        except KeyError:
            oov_count+1
    if oov_count>0:
        print('Oov words:{}'.format(oov_count))
    return embed