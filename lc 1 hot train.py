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
np.random.seed(123)

torch.manual_seed(0)



num_epochs = 1000
batch_size = 80
learning_rate = 0.00001

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
                                  nn.Linear(2, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, x):
        x = self.main(x)

        return x


classifier = Net()



dict_classifier=torch.load('C:/Users/Zhx/Desktop/cl/cl_classifier_100.pth')

classifier.load_state_dict(dict_classifier)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(7, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 24),
            nn.LeakyReLU(0.1),
            nn.Linear(24,12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 6),
            nn.LeakyReLU(0.1),
            nn.Linear(6, 3)
            )

    def forward(self, x):
        x = self.main(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3, 6),
            nn.LeakyReLU(0.1),
            nn.Linear(6, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 24),




        )
        self.decoder_cont=nn.Sequential(
            nn.Linear(24,5),
            nn.Tanh()
            )

        self.decoder_cats=nn.Sequential(
            nn.Linear(24,2),
            #nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.main(x)
        decoded_conts=self.decoder_cont(x)
        decoded_cats=self.decoder_cats(x)

        return decoded_conts, decoded_cats

model_encoder = Encoder()
model_decoder = Decoder()

b=0
for i in range(10000):
    if lc['loan_status'][i]==0:
        b=b+1
#print(b)

#p_low=b/32560
p_low=0.36
p=[p_low,1-p_low]

print(p)

def lkd(x, y):
    lkd_ = 0
    for i in range(batch_size):
        #first_term = 1 / 2 * (x[i] - means[y[i].numpy()]).view(4, 1).t() @ (x[i] - means[y[i].numpy()]).view(4, 1)
        # print(first_term)

        k=y[i].numpy()
        first_term=d_k_i_(x,i,k)*math.log(p[k])
        second_term=0.5 * torch.log(torch.pow(torch.square(stdevs)[k],3))

        # second_term=0.5*torch.log(torch.pow((stdevs[y[i].numpy()],12).view(1))
        # print(second_term)
        #second_term = 0.5 * torch.log(torch.pow(stdevs[y[i].numpy()], 12))
        lkd_ = (lkd_ + first_term+second_term)/batch_size
        #print('lkd',lkd_)
        return lkd_


# print(lkd(output,y))


# (torch.mm(torch.transpose((output[i]-means[y[i].numpy()]).view(3,1),0,1),(output[i]-means[y[i].numpy()]).view(3,1)))


# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
# model.parameters(), lr=learning_rate, weight_decay=1e-5)

# create parameters of means and stdevs
means = torch.tensor(np.random.randn(2, 3),requires_grad=True)
print('orgin',means)
# print(means)
#stdevs = torch.tensor(np.random.randn(2, 1),requires_grad=True)
stdevs= torch.tensor([[0.50],[0.50]],requires_grad=True)
print('origin',stdevs**2)


#covarance=torch.square(stdevs)




# parameters1 = [means, stdevs]

'''def d_i(i):
    sted_inv=1/stdevs[y[i].numpy()]
    d_i=(0.5*(output[i]-means[y[i].numpy()]).t()).view(1,3)@torch.diag(sted_inv.expand(1,3)[0])@(output[2]-means[y[i].numpy()])'''

e=0.0000000001
# distance of a sample to corresponding center on latent space
def d_k_i_(x, i, k):
    sted_inv = 1 / (torch.square(stdevs)[k])
    '''print('st',torch.square(stdevs))
    print('sted',1 / torch.square(stdevs))
    print(sted_inv)
    print(torch.diag(sted_inv.expand(1, 12).view(12)))
    #print('t',(x[i] - means[k]).t())'''
    #print((0.5 * (x[i] - means[k]).t()).view(1, 12).shape,torch.diag(sted_inv.expand(1, 12).view(12)).shape,(x[i] - means[k]).shape)
    d_k_i = (0.5 * (x[i] - means[k]).t()).view(1, 3) @ torch.diag(sted_inv.double().expand(1, 3).view(3)) @ (x[i] - means[k]).view(3, 1)
    #print('d',d_k_i)
    return d_k_i
#p=[p_low,1-p_low]
#print(p[1])

# share of p
def share_cls(x, i, k, a):
    sted_pow = torch.pow(torch.square(stdevs)[k], -0.5 * 3)
    #print(sted_pow)
    # c=torch.diag(b.expand(1,3)[0])
    share = sted_pow.double() @ torch.exp(-1 * d_k_i_(x, i, k) * (1 + a))*p[k]
    '''if share >= 0:
        print('true')
    else:
        print('false')
    print(share)'''
    '''if share.data >=0:
        print('true',share)
    else:
        print('flase',share)'''
    #print(share)
    return share


# sum of p
def sum_(x, i, a):
    sum = 0
    for k in range(2):
        sum = sum + share_cls(x, i, k, a)

    return sum


# p of a certain class
def cls_i(x, i, a,y):
    deri = share_cls(x, i, y[i].numpy(), a)


    deno = sum_(x, i, a)

    '''if deri.data<=deno.data:
        print('true',deri,deno)
    else:
           print('false',deri,deno)'''


    cls = torch.log(deri / deno+e)
    #print('deri',deri)
    #print('deno',deno)
    #print(cls)


    return cls


# classification loss of a batch
def cls(x, y):
    cls = 0
    for i in range(batch_size):

        #if y.size != [80, 1]:
         cls = (cls -cls_i(x, i, a,y))/batch_size


        #print('cls',cls)

    return cls


'''def cls(x,y):
          cls=0
          for i in range(127):
            deriv=0
            for in in range(9):
             deriv=deriv+share(x[i],y[i])
            share_i=share_cls(x[i],y[i])
            cls=cls+share_i/deriv
          return cls'''

# instance encoder
model_encoder = Encoder()
model_decoder = Decoder()




def train_encoder(x, y, lamda1):
    global a
    a = 0.01
    optimizer1 = optim.Adam([{'params': model_encoder.parameters()},
                            {'params': means},
                            {'params': stdevs}], lr=learning_rate)

    output = model_encoder(x.float())
    optimizer1.zero_grad()

    loss1 = (lamda1*lkd(output, y) + cls(output, y)).view(1)
    #params=list(model_encoder.parameters())+list(parameters1)


    loss1.backward(retain_graph=True)
    optimizer1.step()


    # print(cls(output,y))

    return(loss1)






def train_autoencoder(x,y):
    optimizer2 = optim.Adam([{'params': model_encoder.parameters()},
                             {'params': model_decoder.parameters()}
                             ], lr=learning_rate)

    output_ = model_encoder(x[:, 0:7].float())
    output_conts,output_cats = model_decoder(output_)
    #print(output_cats.shape)
    #print(output_cats.shape)

    weight = torch.FloatTensor([1 - p_low, p_low])
    # print(y)

    weight_ = weight[y.data.view(-1).long()].view_as(y)
    # print(weight_)
    criterion1 = nn.MSELoss(reduction='none')
    # print(output.float().shape)
    loss = criterion1(output_conts, x[:, 0:5])
    # print(loss.shape)
    # print(weight_.shape)
    loss_1 = torch.matmul(weight_.reshape(1, 80), loss)
    # print(loss_1.shape)
    loss_1_weighted = loss_1.mean()
    #index=0
    #criterion2_stack=0
    #k=-1
    #for i in cat_names:
        #k=k+1
    #num_categorical=16
    out_put=output_cats[:,0:2]
    t=0.5
    out_put_cat_aneal=out_put/t
    #index=num_categorical+index
        #print(x_original)
        #print(out_put)
    input_class=x[:,7]
    #print('input',input_class.shape)
        #print(input_original.shape)
        #print(out_put.shape)
        #print(input_original.shape)
        #print(criterion2(input_original,out_put))
    criterion2 = nn.CrossEntropyLoss(reduction='none')
    loss = criterion2(out_put_cat_aneal, input_class.long())
    # print(loss.shape)
    # print(weight_.shape)
    loss_2 = loss * weight_
    loss_2_weighted = loss_2.mean()
    criterion2_stack = loss_2_weighted

    optimizer2.zero_grad()

    loss2 = (loss_1_weighted+1/2*criterion2_stack+0.1*lkd(output_,y)).view(1)
    loss3 = (loss_1_weighted + 1/2*criterion2_stack).view(1)
    #print(criterion1(output_conts,x[:,0:6]))
    #print('stack',0.05*criterion2_stack)
    #print(0.1*lkd(output_,y))


    loss2.backward(retain_graph=True)
    optimizer2.step()



    return (loss3,loss2)


k=0
for epoch in range(num_epochs):
    k=k+1

    loss_encoder=0
    loss_autoencoder=0
    loss_real=0
    i=0

    for data in dataloader:
        i+=1
        x,y= data
        #print('x',x.shape)
        #print(x,y)
        '''x_cont = x[:, 0:5]
        x_cat = x[:, 5:21]
        x_cat_cls=x[:,21]
        #print(x_cat)

        #transform continuous features using min-max transform
        x_cont_array = x_cont.numpy()
        scaler = MinMaxScaler()
        max = x_cont_array.max(axis=0)
        min = x_cont_array.min(axis=0)

        x_cont_encoded = scaler.fit_transform(x_cont_array)


        x_cat_array = x_cat.numpy().astype(str)
        x_new=np.concatenate((x_cont_encoded,x_cat_array),axis=1)
        #x_dataframe=pd.DataFrame(x_new,columns=['age', 'final-weight', 'education-num','capital-gain','capital-loss','hours-per-week','workclass', 'education', 'marital-status', 'occupation', 'relationship'])
        #encoder = ce.TargetEncoder(cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship'])
        x_dataframe=pd.DataFrame(x_new,columns=cont_names+cat_names)
        #print(x_dataframe)
        encoder = ce.TargetEncoder(cols=[ 'education'])
        x_encoded = encoder.fit_transform(x_dataframe, y) #dataframe
        #print(x_encoded)
        #x_cat_encoded = x_cat_encoded.to_numpy()

        # print(x_cat_encoded.shape)

        # print(x_cont_encoded)
        #x_encoded_array = np.concatenate((x_cont_encoded, x_cat_encoded), axis=1)
        x_encoded = torch.from_numpy(x_encoded.values.astype(np.float32))
        #print(x_encoded)


        # second_term=torch.pow(stdevs[y[i].numpy()],12)second_term=0.5*torch.log(torch.pow((stdevs[y[i].numpy()],12).view(1))
        # print(second_term)

        #x_encoded = x_encoded.view(x_encoded.size(0), -1)

        #x_encoded = Variable(x_encoded)
        #print(img.shape)

        #print(x_encoded.shape)'''
        output = classifier.forward(x[:, 0:7])
        #print('out',output.shape)
        y_ = (output > 0.5).int().reshape(80)

        #x_dataframe_1 = pd.DataFrame(x_encoded.numpy(), columns=cont_names + cat_names)
        # print(x_dataframe)
        #encoder = ce.TargetEncoder(cols=['education'])
        #x_encoded_1 = encoder.fit_transform(x_dataframe_1, y)
        #x_encoded_2 = Variable(torch.from_numpy(x_encoded_1.values.astype(np.float32)))
        #print(y.shape)

        loss_encoder+=train_encoder(x[:, 0:7], y_, 0.1)
        # print(train_autoencoder(x_cat, x_encoded.float(), y))
        loss_autoencoder += train_autoencoder(x.float(), y_)[1]
        loss_real+=train_autoencoder(x.float(), y_)[0]

        #if i%3==0:
        #rint(x_cat.float())
           #
    if k%2==0:
      print(k)
      print(loss_encoder)

      print(loss_autoencoder)
      print(loss_real)

      print(means)
      print(torch.square(stdevs))
      torch.save({
            'encoder_state_dict':model_encoder.state_dict(),
            'decoder_state_dict': model_decoder.state_dict(),
            'means':means,
            'covariance':stdevs

            }, 'C:/Users/Zhx/Desktop/decoder/decoder_lending_{}.pth'.format(k))
