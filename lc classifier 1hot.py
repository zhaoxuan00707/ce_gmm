from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import os
import torch.utils.data as data_utils
import math

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
import torch.optim as optim
import random
random.seed(0)


import numpy as np
np.random.seed(0)

torch.manual_seed(123)

num_epochs = 1000
batch_size = 80
learning_rate = 0.0001

df_scan = pd.read_csv('C:/Users/Zhx/Desktop/Loan_status_2007-2020Q3.gzip',nrows=30000, low_memory=False)
pd.options.mode.chained_assignment = None
print(df_scan.shape)
#df_scan.head()
#for col in df_scan.columns:
    #print(col)
#print(df_scan['fico_range_high'])
#print(df_scan.head())
cat_names=[ 'term']
cont_names=['dti', 'loan_amnt','int_rate','annual_inc','fico_range_low']
lc=df_scan[cont_names+cat_names+['loan_status']]
lc = lc.dropna()

new=[]
for i in lc["int_rate"].values:
    i=float(i.replace("%", ""))/100
    new.append(i)
#print(new)
lc["int_rate"]=new

term=OneHotEncoder().fit_transform(lc['term'].values.reshape(-1,1)).toarray()
term_class=LabelEncoder().fit_transform((lc['term']))
lc['loan_status']=LabelEncoder().fit_transform(lc['loan_status'])


b=0
for i in range(30000):
    if lc['loan_status'][i]==0:
        b=b+1
#print(b)

p_low=b/30000

p=[p_low,1-p_low]

print(p)

x_cont_array = lc[cont_names].values
scaler = MinMaxScaler()
max = x_cont_array.max(axis=0)
min = x_cont_array.min(axis=0)
x_cont_encoded = scaler.fit_transform(x_cont_array)

#print(x_cont_encoded.shape,edu.shape,edu_class.shape)
x_new = np.concatenate((x_cont_encoded, term,term_class.reshape(30000,1)), axis=1)

        # x_dataframe=pd.DataFrame(x_new,columns=['age', 'final-weight', 'education-num','capital-gain','capital-loss','hours-per-week','workclass', 'education', 'marital-status', 'occupation', 'relationship'])
        # encoder = ce.TargetEncoder(cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship'])
        #x_dataframe = pd.DataFrame(x_new,
                                   #columns=['age', 'education-num', 'capital-gain', 'capital-loss',
                                            #'hours-per-week', 'education'])
        #encoder = ce.TargetEncoder(cols=['education'])
        #x_encoded = encoder.fit_transform(x_dataframe, y)  # dataframe
        # print(x_encoded.values)
        # x_cat_encoded = x_cat_encoded.to_numpy()

        # print(x_cat_encoded.shape)

        # print(x_cont_encoded)
        # x_encoded_array = np.concatenate((x_cont_encoded, x_cat_encoded), axis=1)
x_encoded = torch.from_numpy(x_new.astype(np.float32))

X=x_encoded
#print(adult[cont_names+cat_names])
Y=torch.tensor(lc['loan_status'].values)

dataset=data_utils.TensorDataset(X,Y)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
                                  nn.Linear(7, 4),

                                  nn.ReLU(),

                                  nn.Linear(4, 2),
                                  nn.ReLU(),
                                  nn.Linear(2, 1)
                                  )

    def forward(self, x):
        x1 = self.main(x)
        x2 = torch.sigmoid(x1)
        return x1, x2


classifier = Net()

optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

for i in range(1000):
    loss_c = 0
    correct = 0
    up=0
    for data in dataloader:
        x, y = data
        # print(x,y)


        # second_term=torch.pow(stdevs[y[i].numpy()],12)second_term=0.5*torch.log(torch.pow((stdevs[y[i].numpy()],12).view(1))
        # print(second_term)

        # x_encoded = x_encoded.view(x_encoded.size(0), -1)

        x = Variable(x[:,0:7])


        optimizer.zero_grad()
        output = classifier.forward(x)[1].reshape(80)
        weight = torch.FloatTensor([1 - p_low, p_low])
        #print(y)

        weight_ = weight[y.data.view(-1).long()].view_as(y)
        #print(weight_)
        criterion = nn.BCELoss(reduction='none')
        loss = criterion(output.float(), y.float())
        loss_class_weight = loss * weight_
        loss_class_weighted = loss_class_weight.mean()

        loss_class_weighted.backward()
        optimizer.step()
        loss_c += loss_class_weighted.item()
        output = (output > 0.5).float()
        # print(output)
        correct += (output == y).float().sum()
        up+=(output==1).float().sum()
        # print(correct)
    print(i,correct, loss_c, correct / 30000,up/30000)
    torch.save(
          classifier.state_dict()



            , 'C:/Users/Zhx/Desktop/cl/cl_classifier_{}.pth'.format(i))
