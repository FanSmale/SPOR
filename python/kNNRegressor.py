# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:51:27 2020

@author: 11325
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def EuclideanDistance(x, y):
    '''
    Get the Euclidean Distance between to matrix (x-y)^2 = x^2 + y^2 - 2xy.

    Parameters
    ----------
    x : The feature of first instance.
    y : The feature of second instance.

    Returns
    -------
    dis : The Euclidean distance between x and y.

    '''

    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def predict(kValue,train_data,train_label,paraInstance):
    '''
    Predict the given instance.

    Parameters
    ----------
    kValue :        The number of nearest neighbor.
    train_data :    The training set.
    train_label :   The label of train_data.
    paraInstance :  The given instance.

    Returns
    -------
    Instance_predict : The prediction of paraInstance.

    '''
    tempNeighbor=findneighbor(kValue,train_data,train_label,paraInstance)
    
    tempPre=np.sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    Instance_predict=tempPre/kValue
    return Instance_predict

def selfpredict(kValue,train_data,train_label,paraInstance):
    '''
    Predict the given instance of training set.

    Parameters
    ----------
    kValue :        The number of nearest neighbor.
    train_data :    The training set.
    train_label :   The label of train_data.
    paraInstance :  The given instance.

    Returns
    -------
    Instance_predict : The prediction of paraInstance.

    '''
    tempNeighbor=findSelfneighbor(kValue,train_data,train_label,paraInstance)
    tempPre=np.sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    Instance_predict=tempPre/kValue
    return Instance_predict

def findneighbor(kValue,train_data,train_label,paraInstance):
    '''
    Find the nearest neighbor of given instances in training set.

    Parameters
    ----------
    kValue :        The number of nearest neighbor.
    train_data :    The training set.
    train_label :   The label of train_data.
    paraInstance :  The given instance.

    Returns
    -------
    returnIndex :   The index of nearest neighbor in training set.

    '''
    tempIndex=[]
    resultIndex=[]
    returnIndex=[]
    tempIndex=EuclideanDistance(train_data, paraInstance)
    resultIndex=np.argsort(tempIndex,0)
    returnIndex=resultIndex[0:kValue:1]  
    return returnIndex

def findSelfneighbor(kValue,train_data,train_label,paraInstance):
    '''
    Find the nearest neighbor of given instances in training set.

    Parameters
    ----------
    kValue :        The number of nearest neighbor.
    train_data :    The training set.
    train_label :   The label of train_data.
    paraInstance :  The given instance.

    Returns
    -------
    returnIndex :   The index of nearest neighbor in training set.

    ''' 
    tempIndex=[]
    resultIndex=[]
    returnIndex=[]
    tempIndex=EuclideanDistance(train_data, paraInstance)
    resultIndex=np.argsort(tempIndex,0)
    returnIndex=resultIndex[1:kValue+1:1]   
    return returnIndex
    
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
    for i,paraInstance in enumerate(train_data):    
        paraInstance=paraInstance.reshape(1,-1)
        paraLabel=predict(3,train_data,train_label,paraInstance)
        #print(train_data.shape[1])
        neighborLabel=selfpredict(3,train_data,train_label,paraInstance)
        #neighborLabel2=findSelfneighbor(3,train_data,train_label,paraInstance,i)
        print(neighborLabel)
        #print(neighborLabel2)
        mse+=(paraLabel-unlabel_label[i])*(paraLabel-unlabel_label[i])
        
        
    print(mse/unlabel_data.shape[0])
if __name__ == '__main__':
    main()
