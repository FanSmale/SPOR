# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:05:24 2021

@author: 11325
"""
import torch
import torch.nn.functional as F  


class Net(torch.nn.Module):  


    def __init__(self, n_feature,n_hidden,n_hidden1,n_hidden2,n_output):  
        # Define the information of the layer

        super(Net, self).__init__()

        # Define the form of each layer uses 

        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)  
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  

        self.predict = torch.nn.Linear(n_hidden2, n_output) 

 

    def forward(self, x):  
        # The forward pass process of the neural network 

        x=self.hidden(x)
        x=self.hidden1(x)

        x = F.relu(self.hidden2(x)) 

        x = self.predict(x)

        return x

 

 


def train_iteration(model,data,label,num_epoch,learning_rate):
    '''
    The training process of given neural network regressor.

    Parameters
    ----------
    model :         The given neural network regressor.
    data :          The training set.
    label :         The label of data.
    num_epoch :     The number of training rounds.
    learning_rate : The learning rate of given neural network regressor.

    Returns
    -------
    model : The updated regressor.

    '''
    #Tranform data to tensor.
    data=torch.from_numpy(data).float()
    label=torch.from_numpy(label).float()

    #Define the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()  
    
    #Train the regressor
    for e in range(num_epoch):
        label_pre=model(data)
        loss=loss_func(label_pre,label)
        if (e%100==0):
            print(f'Epoch:{e},loss:{loss.data.numpy()}')
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
    return model