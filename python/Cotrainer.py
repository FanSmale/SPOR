# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:59:01 2020

@author: Li Yu
"""

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import os
import copy
import kNNRegressor
import random

import random as rd
rd.seed(1234)



def net_selectCriticalInstances(regressor,kValue,train_data,train_label,unlabel_data,poolsize):
    '''
    Find the most confident unlabeled instances in unlabel_data by the given regressor.  
    Parameters
    ----------
    regressor :     The learner.
    kValue :        The number of nearest neighbor.
    train_data :    Training data for regressor.
    train_label :   The label of training data.
    unlabel_data :  Unlabeled data.
    poolsize :      The number of sampled unlabeled instances from unlabeled data.
    
    Returns
    -------
    resultData:     The most confident unlabeled instances.
    resultClassValue:The pseudo label of resultData.
    resultIndex:    The index of resultData in unlabel_data.

    '''
    
    
    #Step 1. Prepare the temp parameters.
    if unlabel_data.shape[0]<poolsize:     
        tempPool=np.arange(unlabel_data.shape[0])
    elif unlabel_data.shape[0]>=poolsize:
        tempPool=random.sample(range(0,unlabel_data.shape[0]),poolsize)
    tempInstance=[]
    maxDelta=-1
    resultClassValue=[]
    delta=[]
    tempNeighbor=[]
    tempValue=[]
    temp_train_data=torch.from_numpy(train_data)
    label_pre=regressor(temp_train_data)
    label_pre=label_pre.data.numpy()
    difference=train_label-label_pre   
    
    #Step 2. Enumerating the sampled unlabeled instances and computes the confidence value.
    for i,paraUnlabel in enumerate(tempPool):
        tempInstance=copy.deepcopy(unlabel_data[paraUnlabel])
        tempInstance=tempInstance.reshape(1,-1)
        tempnetInstance=torch.from_numpy(tempInstance)
        tempnetLabel=regressor(tempnetInstance)
        tempLabel=tempnetLabel.data.numpy()
        
        tempTrain_data=copy.deepcopy(train_data)
        tempTrain_label=copy.deepcopy(train_label)
        tempNeighbor=kNNRegressor.findneighbor(kValue,train_data,train_label,tempInstance)
        insertInstance=np.array(tempInstance)
        tempTrain_data=np.concatenate((tempTrain_data, insertInstance),axis=0)
        tempTrain_label=np.concatenate((tempTrain_label, tempLabel),axis=0)
        tempOldError=0
        tempNewError=0
        tempNewValue = 0
        tempOldValue=0
        

       #Step 2.1 Enumerating the k nearest neighbor and calculates the improvement of self-predictions.
        for j,paraNeighbor in enumerate(tempNeighbor):
            tempNeighborInstance=train_data[paraNeighbor]
            tempNeighborInstance=tempNeighborInstance.reshape(1,-1)
            tempOldError=difference[paraNeighbor]
            tempOldValue+=tempOldError*tempOldError
            tempNewError=(kNNRegressor.selfpredict(kValue,tempTrain_data,tempTrain_label,tempNeighborInstance)-tempTrain_label[paraNeighbor])
            tempNewValue+=tempNewError*tempNewError
        delta.append(tempOldValue/tempNeighbor.shape[0] - tempNewValue/tempNeighbor.shape[0])
        tempValue.append(tempLabel) 
        
    #Step 3 Find the instances with maximum confidence value and return it.
    maxDelta=np.array(max(delta))
    resultData=[]
    resultIndex=[]
    for i,paraUnlabel in enumerate(tempPool):
        if delta[i]>=maxDelta and delta[i]>0:
            tempInstance=copy.deepcopy(unlabel_data[paraUnlabel])
            tempInstance=tempInstance.reshape(1,-1)
            resultIndex.append(paraUnlabel)
            resultData.append(tempInstance)
            resultClassValue.append(tempValue[i])
    resultIndex=sorted(resultIndex,reverse=True)
    resultData=np.array(resultData).reshape(-1,train_data.shape[1])
    resultClassValue=np.array(resultClassValue).reshape(-1,1)
    return resultData,resultClassValue,resultIndex





def treat_selectCriticalInstances(regressor,kValue,train_data,train_label,unlabel_data,cotrainer_instances,cotrainer_labels,poolsize,numIteration,regressormode):
    
    '''
    Find the most confident unlabeled instances in unlabel_data by the given regressor.  
    Parameters
    ----------
    regressor :         The learner.
    kValue :            The number of nearest neighbor.
    train_data :        Training data for regressor.
    train_label :       The label of training data.
    unlabel_data :      Unlabeled data.
    cotrainer_instances:Selected instances remain in the training set.
    cotrainer_labels:   The label of cotrainer_instances.
    poolsize :          The number of sampled unlabeled instances from unlabeled data.
    regressormode:      The mode of computing confidence value.
    
    Returns
    -------
    resultData:         The most confident unlabeled instances.
    resultClassValue:   The pseudo label of resultData.
    resultIndex:        The index of resultData in unlabel_data.
    cotrainer_instances:The remianing unlabeled instances in training set.
    '''
    
    #Step 1. The self-pacced paradigm for controling learning space of selected instances. 
    if numIteration<40:
        numchoose=1+cotrainer_instances.shape[0]
    elif numIteration>=40 and numIteration<60:
        numchoose=1+cotrainer_instances.shape[0]
    elif numIteration>=60 and numIteration<80:
        numchoose=1+cotrainer_instances.shape[0]
    else:
        numchoose=2+cotrainer_instances.shape[0]
    print(numchoose)
    
    #Step 1.1 Prepare the temp parameters.
    if unlabel_data.shape[0]<poolsize:     
        tempPool=np.arange(unlabel_data.shape[0])
    elif unlabel_data.shape[0]>=poolsize:
        tempPool=random.sample(range(0,unlabel_data.shape[0]),poolsize)
        
    tempInstances=np.zeros((poolsize,unlabel_data.shape[1]))
    for i,paraInstance in enumerate(tempPool):
        tempInstances[i]=unlabel_data[paraInstance]
        
    #Step 2. Enumerating the sampled unlabeled instances and computes the confidence value.
    delta1,label1=confidence(regressor,kValue,train_data,train_label,tempInstances[:,0:tempInstances.shape[1]],0,regressormode)
    delta2,label2=confidence(regressor,kValue,train_data,train_label,cotrainer_instances[:,0:cotrainer_instances.shape[1]],0,regressormode)
    tempDelta=np.concatenate((delta1,delta2),axis=1)
    tempLabel=np.concatenate((label1,label2),axis=1)
    tempIndex=np.argsort(tempDelta)
    tempIndex=tempIndex[:,::-1]
    
    resultData=np.zeros((1,unlabel_data.shape[1]))
    resultlabel=np.zeros((1,1))
    resultindex=np.zeros((1,1))
    deleteIndex=np.zeros((1,1))
    
    #Step 3. Find the suitable instances and removing unsuitable instances in cotrainer_instances.
    for i in range(numchoose):
        if tempIndex[0][i]<poolsize:
            unlabel_index=tempPool[tempIndex[0][i]]         
            resultData=np.concatenate((resultData,unlabel_data[unlabel_index,:].reshape(1,-1)),axis=0)
            resultlabel=np.concatenate((resultlabel,tempLabel[:,tempIndex[0][i]].reshape(1,-1)),axis=0)
            resultindex=np.concatenate((resultindex,np.array(unlabel_index).reshape(1,1)),axis=0)
        else:
            resultData=np.concatenate((resultData,cotrainer_instances[tempIndex[0][i]-100,:].reshape(1,-1)),axis=0)
            resultlabel=np.concatenate((resultlabel,tempLabel[:,tempIndex[0][i]].reshape(1,-1)),axis=0)
            deleteIndex=np.concatenate((deleteIndex,np.array(tempIndex[0][i]).reshape(1,1)),axis=0)
    if deleteIndex.shape[0]>1:
        deleteIndex=deleteIndex[1:deleteIndex.shape[0],:]
        deleteIndex=np.sort(deleteIndex,axis=0)
        deleteIndex=deleteIndex[::-1]
        for i in range(deleteIndex.shape[0]):
            cotrainer_instances=np.delete(cotrainer_instances,np.int(deleteIndex[i]-100),axis=0)
    
    #Step 4. Return selected instances. 
    resultindex=resultindex[1:resultindex.shape[0],:]
    resultindex=np.sort(resultindex,axis=0)
    returnIndex=np.zeros((resultindex.shape[0],1))
    for i in range(resultindex.shape[0]):
        returnIndex[i][0]=resultindex[resultindex.shape[0]-i-1][0]
        
    resultlabel=resultlabel[1:resultlabel.shape[0],:]
    
    resultData=resultData[1:resultData.shape[0],:]        
    return resultData,resultlabel,returnIndex.astype(np.int),cotrainer_instances.reshape(-1,unlabel_data.shape[1])



def confidence(regressor,kValue,train_data,train_label,confidence_data,confidencemode,regressormode):
    '''
    Compute the confidence value and pseudo labels of sampled unlabeled instancesby the given regressor.  
    Parameters
    ----------
    regressor :         The learner.
    kValue :            The number of nearest neighbor.
    train_data :        Training data for regressor.
    train_label :       The label of training data.
    confidence_data:    The sampled unlabeled instances.
    regressormode:      The mode of computing confidence value.
    
    Returns
    -------
    delata:             The confidentce value of confidencemode.
    label:              The pseudo labels of confidencemode.
    '''
    #Step 1. Prepare the tool parameters.
    if regressormode==0:
        temp_train_data=torch.from_numpy(train_data).float()
        label_pre=regressor(temp_train_data)
        label_pre=label_pre.data.numpy()
    else:
        label_pre=regressor.predict(train_data).reshape(-1,1)
    
    difference=train_label-label_pre
    delta=np.zeros((1,confidence_data.shape[0]))
    label=np.zeros((1,confidence_data.shape[0]))
    
    #Step 2. Enumerate confidence_data and compute the confidence value of each instance.
    for i,tempInstance in enumerate(confidence_data):
        
        #Step 2.1 Predict and find the neighbor of current instance.
        tempInstance=tempInstance.reshape(1,-1) 
        if regressormode==0:
            tempnetInstance=torch.from_numpy(tempInstance).float()
            tempnetLabel=regressor(tempnetInstance)
            tempLabel=tempnetLabel.data.numpy()
        else:
            tempLabel=regressor.predict(tempInstance).reshape(-1,1)
        
        tempTrain_data=copy.deepcopy(train_data)
        tempTrain_label=copy.deepcopy(train_label)
        if confidencemode==0:
            tempNeighbor=kNNRegressor.findneighbor(kValue,train_data,train_label,tempInstance)
        else:
            tempNeighbor=kNNRegressor.findSelfneighbor(kValue,train_data,train_label,tempInstance)
        
        insertInstance=np.array(tempInstance)
        tempTrain_data=np.concatenate((tempTrain_data, insertInstance),axis=0)
        tempTrain_label=np.concatenate((tempTrain_label, tempLabel),axis=0)
        tempOldError=0
        tempNewError=0
        tempNewValue=0
        tempOldValue=0
        
        #Step 2.2 Compute the improvement of self-prediction.
        for j,paraNeighbor in enumerate(tempNeighbor):
            tempNeighborInstance=train_data[paraNeighbor]
            tempNeighborInstance=tempNeighborInstance.reshape(1,-1)
            #tempOldError=kNNRegressor.selfpredict(kValue,train_data,train_label,tempNeighborInstance)-train_label[paraNeighbor]
            tempOldError=difference[paraNeighbor]
            tempOldValue+=tempOldError*tempOldError
            tempNewError=(kNNRegressor.selfpredict(kValue,tempTrain_data,tempTrain_label,tempNeighborInstance)-tempTrain_label[paraNeighbor])
            tempNewValue+=tempNewError*tempNewError
            
        #Step 3. Return the confidence value and pseudolabels.
        delta[0][i]=tempOldValue/tempNeighbor.shape[0] - tempNewValue/tempNeighbor.shape[0]
        label[0][i]=tempLabel   
    return delta,label

            
def main():
    path=os.getcwd()+'\kin8nm.csv'
    f= open(path,encoding='utf-8')
    data=pd.read_csv(f)
    all_features=data.iloc[:,0:data.shape[1]-1]
    all_labels=data.iloc[:,data.shape[1]-1:data.shape[1]]
    all_features = all_features.apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    all_labels=all_labels.apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    num_index=all_features.shape[0]
    data=all_features[0:num_index].values.astype(np.float32)
    label=all_labels[0:num_index].values.astype(np.float32)
    train_data,unuse_data,train_label,unuse_label=train_test_split(data,label,test_size=0.99)
    unlabel_data,test_data,unlabel_label,test_label=train_test_split(unuse_data,unuse_label,test_size=(0.3)/(1-0.01))
    mse=0

    resultData,resultLabel,deleteIndex=net_selectCriticalInstances(1,3,train_data,train_label,unlabel_data,100,0.8)
       
    print(resultData,resultLabel,deleteIndex)   
    print(mse/unlabel_data.shape[0])
    
if __name__ == '__main__':
    main()            
            
            