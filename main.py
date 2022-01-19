# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:14:01 2021

@author: zzh
"""
import csv
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm, trange
import pandas as pd
csvFile  = open('lab2_edge.csv')
node_num=16863
def get_metrics(prediction, label):
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()
class node_embedding_net(nn.Module):
    def __init__(self,node_num,embedding_num):
        super(node_embedding_net, self).__init__()
        self.em = nn.Embedding(node_num,embedding_num)
    def forward(self,targ):
        emb = self.em(targ)
        return emb
class Custom_dataset(Dataset):
    def __init__(self, train_data_list):
        self.data_list = train_data_list
        self.counter=0
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]
    def __iter__(self):
        return iter([self.data_list[0][0], self.data_list[0][1]])
    def __next__(self):
        self.counter=self.counter+1
        return [self.data_list[self.counter][0], self.data_list[self.counter][1]]
def one_randwalk(edge,posi_len,start_point,negative_num):
    traj =[start_point]
    node = start_point
    for i in range(posi_len):
        node = random.choice(edge[node])
        traj.append(node)
    counter = 0
    while(1):
        nega_number = random.randint(0,node_num-1)
        if nega_number not in edge[start_point]:
            counter+=1
            traj.append(nega_number)
        if counter == negative_num:
            break
    return traj
def index2onehot(index):
    array = torch.zeros(node_num,dtype=torch.float64)
    array[index] = 1
    return array
def onehot2index(array):
    index = torch.argmax(array)
    return index    
reader = csv.reader(csvFile)
N_nodes = 20000
nodes = []
max_len = 20
edge = []
trajecrories = []
window_size = 3
EPOCH = 5
learning_rate = 0.01
embedding_num = 4096
max_node = 0
negative_num =15
one_hot_dic ={}
temp = 0
vali_list = []
for i in range(N_nodes):
    edge.append([])
for item in reader:
    if item[0]=='source':
        continue
    temp+=1
    if temp < 46000:
        source = int(item[0])
        if source>max_node:
            max_node = source
        if source not in nodes:
            nodes.append(source)
        target = int(item[1])
        if target>max_node:
            max_node = source
        if target not in nodes:
            nodes.append(target)
        edge[source].append(target)
        edge[target].append(source)
    else:
        vali_list.append((int(item[0]),int(item[1]),1))
for node in nodes:
    one_hot_dic[node] = index2onehot(node)
print('finishing buiding graph')
vali_list_0 = []
counter = 0
while(1):
    if counter == len(vali_list):
        break
    node1 = random.choice(range(node_num))
    node2 = random.choice(range(node_num))
    if node1 in edge[node2]:
        continue
    else:
        vali_list_0.append((node1,node2,0))
        counter+=1
for node in nodes:
    for i in range(5):
        one_walk = one_randwalk(edge,max_len,node,negative_num)
        trajecrories.append(one_walk)
train_data = []
for traj in trajecrories:
    train_data.append((traj[0],traj[1:]))
data_set = Custom_dataset(train_data)
train_loader = DataLoader(data_set,batch_size=1,shuffle=True)
net = node_embedding_net(node_num,embedding_num).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
simlar = nn.CosineSimilarity().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
logsigmoid = nn.LogSigmoid().cuda()
sigmoid = nn.Sigmoid().cuda()
ex_scedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)
'''
W1 = torch.randn((node_num, embedding_num),requires_grad=True,dtype=float).cuda()
W2 =  torch.randn((embedding_num, node_num),requires_grad=True,dtype=float).cuda()
'''
'''
W1 = np.random.uniform(-1, 1, (node_num, embedding_num))  # V x N
W1 = torch.tensor(W1,requires_grad=True,dtype=float).cuda()
W2 = np.random.uniform(-1, 1, (embedding_num, node_num))  # N x V
W2 = torch.tensor(W2,requires_grad=True,dtype=float).cuda()
'''

for i in range(EPOCH):
    with tqdm(total=len(train_loader)) as pbar:
        pbar.set_description('Processing:')
        for i,data in enumerate(train_loader):
            start,context = data
            loss = 0
            context = torch.tensor(context).cuda()
            start = torch.tensor(start).cuda()
            embe_start = net(start)
            walk = net(context[0:max_len])
            embe_negative = net(context[max_len+1:])
            loss -=torch.sum(logsigmoid(simlar(embe_start,walk)))
            loss -=torch.sum(logsigmoid(simlar(-embe_start,embe_negative)))
            if i% 10000 == 1:
                print(loss.item())
            loss.backward()
            optimizer.step()
            pbar.update(1)
    ex_scedule.step()
list_label = []
list_predict = []
'''
for node1 in nodes[0:500]:
    for node2 in nodes:
        if node1 == node2:
            continue
        if node2 in edge[node1]:
            list_label.append(1)
        else:
            list_label.append(0)
        node1 = torch.tensor(node1).cuda()
        node2 = torch.tensor(node2).cuda()
        embedding_node1 = net(node1)
        embedding_node2 = net(node2)
        embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
        embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
        prob = sigmoid(simlar(embedding_node1,embedding_node2))
        prob = prob.cpu()
        list_predict.append(prob.detach().numpy())
'''
for i,j,label in vali_list:
    list_label.append(label)
    node1 = torch.tensor(i).cuda()
    node2 = torch.tensor(j).cuda()
    embedding_node1 = net(node1)
    embedding_node2 = net(node2)
    embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
    embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
    prob = sigmoid(simlar(embedding_node1,embedding_node2))
    prob = prob.cpu()
    list_predict.append(float(prob.detach()))
for i,j,label in vali_list_0:
    list_label.append(label)
    node1 = torch.tensor(i).cuda()
    node2 = torch.tensor(j).cuda()
    embedding_node1 = net(node1)
    embedding_node2 = net(node2)
    embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
    embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
    prob = sigmoid(simlar(embedding_node1,embedding_node2))
    prob = prob.cpu()
    list_predict.append(float(prob.detach()))
        
print('start to compute auc')
auc = get_metrics(list_predict,list_label)
print(auc)
csvFile  = open('lab2_test.csv')            
reader = csv.reader(csvFile) 
list_id = []
prob_list = []
for item in reader:
    if item[0] == 'id':
        continue
    ids = int(item[0])
    source = int(item[1])
    target = int(item[2])
    node1 = torch.tensor(source).cuda()
    node2 = torch.tensor(target).cuda()
    embedding_node1 = net(node1)
    embedding_node2 = net(node2)
    embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
    embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))  
    prob = sigmoid(simlar(embedding_node1,embedding_node2))
    prob = prob.cpu()
    prob_list.append(float(prob.detach()))
    list_id.append(ids)
frame = pd.DataFrame({'id':list_id,'prob':prob_list})
frame.to_csv('Prediction.csv',index=False)
      
'''
        hid = torch.mm(target,W1)
        print(hid.size())
        W1.retain_grad()
        pre = torch.mm(hid,W2)
        W2.retain_grad()
        pre = softmax(pre)
        for contex_id in context:
            target = torch.tensor([contex_id]).cuda()
            W2.retain_grad()
            loss+= loss_fn(pre,target)
        loss.backward()
        print(loss.item())
        with torch.no_grad():
            W1-=learning_rate*W1.grad
            W2-=learning_rate*W2.grad
            W1.grad = None
            W2.grad = None
        '''
        
'''
        for c_emb in context:
            c_emb = c_emb.long().cuda()
            loss +=loss_fn(pred,c_emb)
        optimizer.zero_grad()
        print(loss.item())
        loss.backward()
        optimizer.step()
        '''
        
    

