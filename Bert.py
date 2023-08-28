import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import pickle
from transformers import BertTokenizer,BertModel,AdamW
import re
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
                data.append((review,1 if label =='pos' else 0))
    with open(os.path.join(cache_dir,f'{type_}.pkl'),'wb') as f:
        pickle.dump(data,f)
train_data=read_imdb('train')
test_data=read_imdb('test')

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

def text_sub(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
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

if os.path.exists(os.path.join(cache_dir, 'train_processed.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'train_processed.pkl'), 'rb') as f:
        input_ids, attention_masks, labels = pickle.load(f)
else:
    input_ids, attention_masks, labels = preprocess(train_data)
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'train_processed.pkl'), 'wb') as f:
        pickle.dump((input_ids, attention_masks, labels), f, protocol=1)
if os.path.exists(os.path.join(cache_dir, 'test_processed.pkl')):
    print('Cache found.')
    with open(os.path.join(cache_dir, 'test_processed.pkl'), 'rb') as f:
        test_input_ids, test_attention_masks, test_labels = pickle.load(f)
else:
    test_input_ids, test_attention_masks, test_labels = preprocess(test_data)
    test_input_ids, test_attention_masks, test_labels = torch.tensor(test_input_ids), torch.tensor(test_attention_masks), torch.tensor(test_labels,dtype=torch.long)
    with open(os.path.join(cache_dir, 'test_processed.pkl'), 'wb') as f:
        pickle.dump((test_input_ids, test_attention_masks, test_labels), f, protocol=1)
class myBert(nn.Module):
    def __init__(self,freeze_bert=False):
        super(myBert,self).__init__()
        num_hiddens,h1,out=768,64,1
        self.bert=BertModel.from_pretrained('./pre_bert')
        # self.lstm=nn.LSTM(768,256,2,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(num_hiddens,h1),
            nn.ReLU(),
            nn.Linear(h1,out)
        )
        self.freeze_bert = freeze_bert
    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        output = torch.sigmoid(self.classifier(last_hidden_state_cls)).squeeze()

        return output


train_dataloader = DataLoader(list(zip(input_ids.unbind(0), attention_masks.unbind(0), labels.unbind(0))), batch_size=32, shuffle=True)

test_dataloader = DataLoader(list(zip(test_input_ids.unbind(0), test_attention_masks.unbind(0), test_labels.unbind(0))), batch_size=32, shuffle=True)
                                  
def train(net,optimizer,data_iter,loss,device):
    net.train()
    net=net.to(device)
    l_loss_sum=0
    num_iter=len(data_iter)
    for batch,(seq,mask,label) in enumerate(data_iter):
        print(batch)
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
        for batch,(seq,mask,label) in enumerate(data_iter):
            seq,mask,label=seq.to(device),mask.to(device),label.to(device)
            y_hat=net(seq,mask)
            acc+=(torch.round(y_hat)==label).float().mean().item()
    acc=acc/num_iter
    print("test_acc:{}".format(acc))
    return acc


net=myBert(freeze_bert=False)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
optimizer=AdamW(net.parameters(),lr=2e-5)
bce_loss_fn = nn.BCELoss()
num_epochs = 2
device='cuda:0'
for i in range(num_epochs):
    print(f'Epoch: {i+1}')

    train(net,optimizer,train_dataloader,bce_loss_fn,device)
    test(net,test_dataloader,device)