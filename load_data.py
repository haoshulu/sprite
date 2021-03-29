 #!/usr/bin/env python3

from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
import torch.nn as nn

# %matplotlib inline
# import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data.sampler import SubsetRandomSampler
from topology import Autopo
import numpy as np


dataset = Autopo('./tmp')

#loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch_size = 64
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 1-train_ratio-val_ratio

shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))

if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

n_train=int(dataset_size*train_ratio)
n_val=int(dataset_size*val_ratio)

train_indices, val_indices, test_indices = indices[:n_train], indices[n_train+1:n_train+n_val], indices[n_train+n_val+1:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

edge_input_dim = 1
node_input_dim = 4
edge_hidden_dim = 4




class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_input_dim * node_input_dim)
        )
        self.conv1 = NNConv(dataset.num_node_features,16,edge_network)
        self.conv2 = NNConv(16,32,edge_network)
        self.conv3 = NNConv(32,256,edge_network)
        self.conv4 = NNConv(256,32,edge_network)
        #self.lin1 = torch.nn.Linear(32, )
        self.lin2 = torch.nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        output_ind=self.output_indices(batch)

        x = self.conv1(x, edge_index,edge_attr)
        #-------------------------------
        #print('x after conv1:',x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #-------------------------------
        #print('x after dropout:',x.shape)
        x = self.conv2(x, edge_index,edge_attr)
        #-------------------------------
        #print('x after conv2:',x.shape)
        x = F.relu(x) 
        x = self.conv3(x, edge_index,edge_attr)
        #-------------------------------
        #print('x after conv3:',x.shape)
        x = F.relu(x)  
        x = self.conv4(x, edge_index,edge_attr)
        #print('x after conv4:',x.shape,)
        x = F.relu(x)  
        #x = self.lin1(x)
        #-------------------------------
        #print('x after lin1:',x.shape,)
        x = self.lin2(x)
        #-------------------------------
        #print('x after lin2:',x.shape,'\n------------------\n*******************************\n',torch.sigmoid(x[output_ind]))

        return torch.sigmoid(x[output_ind])

    def output_indices(self, batch):
        num_element=len(batch)
        output_ind=[]
        count=0
        previous_num=torch.tensor(0,dtype=int).to(device)
        current_num=torch.tensor(-1,dtype=int).to(device)
 
        for id,item in enumerate(batch):
            if not torch.equal(item,current_num):
                count=0
            current_num=item
            count=count+1
            if torch.equal(current_num,previous_num) and count==2:
               output_ind.append(id)
               previous_num=previous_num+1
                
            #-------------------------------
            #print('id:',id,'item:',item,'output_ind:',output_ind,'\n------------------\n')

        return output_ind

def rse(y,yt):

    assert(y.shape==yt.shape)

    var=0
    m_yt=yt.mean()
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2 

    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2

    rse=mse/var

    return rse



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# #---------------------
# print('data',data)
# #---------------------


criterion = MSELoss(reduction='mean').to(device)

train_perform=[]
val_perform=[]

loss=0
for epoch in range(50):

    n_batch_val=0
    val_loss=0
 
    if epoch % 1 == 0:

         model.eval()
         for data in val_loader:
              n_batch_val=n_batch_val+1
              data.to(device)
              out=model(data)
              out = out.reshape(data.y.shape)
              assert(out.shape == data.y.shape)
              loss = F.mse_loss(out, data.y.float())
              val_loss += out.shape[0] * loss.item()

         val_perform.append(val_loss/n_batch_val/batch_size)
         print("val loss: ",val_loss/n_batch_val )
         n_batch_val=0
 
    
    train_loss=0
    n_batch_train=0

    model.train()

    for i,data in enumerate(train_loader):
         n_batch_train=n_batch_train+1
         data.to(device)
         optimizer.zero_grad()
         out=model(data)
         out=out.reshape(data.y.shape)
         assert(out.shape == data.y.shape)
         loss=F.mse_loss(out, data.y.float())
         loss.backward()
         optimizer.step()

         train_loss += out.shape[0] * loss.item()
   
    if epoch % 1 == 0:
         print('%d epoch training loss: %.3f' %
                  (epoch, train_loss/n_batch_train))

         train_perform.append(train_loss/n_batch_train/batch_size)

 


model.eval()

accuracy=0
n_batch_test=0
gold_list=[]
out_list=[]
for data in test_loader:
              n_batch_test+=1
              data.to(device)
              out=model(data).cpu().detach().numpy().reshape(-1)
              gold=data.y.cpu().numpy().reshape(out.shape)
              L=len(gold)
              rse_result=rse(out,gold)
              np.set_printoptions(precision=2,suppress=True)
              # correct = float(out.eq(gold).sum().item())
              # accuracy = correct / data.y.sum().item()
              print("RSE: ",rse_result)
              print("Truth:   ",gold.reshape([L]))
              print("Predict: ",out.reshape([L]))
              
              gold_list.extend(gold.reshape([L])[:])
              out_list.extend(out.reshape([L])[:])


#rse_total=rse(gold_list,out_list)
#print("RSE(total): ",rse_total)

print('gold_list: ',(np.reshape(gold_list,-1)))
print('out_list: ',(np.reshape(out_list,-1)))
np.set_printoptions(precision=2,suppress=True) 
print("train_history: ",train_perform)
print("val_history: ",val_perform)




