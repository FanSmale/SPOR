# coding: utf-8 -*-
"""
Created on Thu Dec 17 20:45:30 2020

@author: 11325
"""

import torch
import os 
import random as rd 
import numpy as np
import cvxopt
import pandas as pd
import Cotrainer
import copy
import torch.nn.functional as F  # 激励函数都在这
from sklearn.metrics import mean_squared_error as mse
import warnings
import model



def setup_seed(seed):
    '''
    Set gloabal random seed

    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

warnings.filterwarnings("ignore")





def SPOR(regressor1,regressor2,base_regressor1,base_regressor2,kValue1,train_data1,train_label1,kValue2,train_data2,train_label2,unlabel_data,test_data,test_label,poolSize,train_iterations,learning_rate):
    
    '''
    The training function that two regressors explore the unalbeled instances with help of self-paced paradigm and safe scheme.  
    
    Parameters
    ----------
    regressor1:     The first learner.
    regressor2:     The second learner.
    base_regressor1:The first learner trained on labeled instances.
    base_regressor2:The second learner trained on labeled instances.
    kValue1 :       The number of maximum nearest neighbor for regressor1.
    train_data1:    Training data for regressor1.
    train_label1 :  The label of training_data1.
    kValue2 :       The number of maximum nearest neighbor for regressor2.
    train_data2:    Training data for regressor2.
    train_label2 :  The label of training_data2.
    unlabel_data :  Unlabeled data.
    test_data:      The test data.
    test_label:     The real label of test_data
    poolsize :      The number of sampled unlabeled instances from unlabeled data.
    train_iterations:The maximum of training iteration.
    learing_rate:   The learning rate for base regressors.
    
    Returns
    -------
    data1:          The expanded training set of regressor1.
    label1:         The label of data1.
    data2:          The expanded training set of regressor2.
    label2:         The label of data2.
    regressor1:     The updated regressor1.
    regressor1:     The updated regressor2.
    log_mse:        The mean squared error in the training process.

    '''
    #Step 1. Prepare the tool parameters.
    data1=copy.deepcopy(train_data1)
    label1=copy.deepcopy(train_label1)
    data2=copy.deepcopy(train_data2)
    label2=copy.deepcopy(train_label2)
    #tempFlag=np.zeros((unlabel_data.shape[0],1))
    #unlabel_data=np.concatenate((unlabel_data,tempFlag),axis=1)
    add_data1=np.empty(shape=[0,train_data1.shape[1]])
    add_data2=np.empty(shape=[0,train_data1.shape[1]])
    temp_data1=np.empty(shape=[0,train_data1.shape[1]])
    temp_data2=np.empty(shape=[0,train_data1.shape[1]])
    temp_label1=np.empty(shape=[0,train_label1.shape[1]])
    temp_label2=np.empty(shape=[0,train_label2.shape[1]])
    temp_train_data1=np.empty(shape=[0,train_data1.shape[1]])
    temp_train_data2=np.empty(shape=[0,train_data1.shape[1]])
    
    numaddinstance1=0
    numaddinstance2=0
    log_mse=np.zeros((1,int((train_iterations/10)+1)))
    
    #Step 2. Explore the unlabeled instances
    for i in range(train_iterations):
        
        #Step 2.1 Regressor1 selected confident instances for regressor2.
        temp_train_data1=np.concatenate((data1,add_data1),axis=0)
        temp_train_label1=np.concatenate((label1,temp_label1),axis=0)
        add_data2,add_label2,delete_index,add_unlabeldata=Cotrainer.treat_selectCriticalInstances(regressor1,kValue1,temp_train_data1,temp_train_label1,unlabel_data,temp_data2,temp_label2,poolSize,i,0)
        
        #Step 2.2 Learn the safe pseudo labels for selected unlabeled instances.
        if i!=0 and i<=100:
            first_add_data2=torch.from_numpy(add_data2).float()
            first_add_label2=regressor2(first_add_data2).data.numpy()
            second_add_label2=regressor1(first_add_data2).data.numpy()
            semi_label2=np.concatenate((add_label2,first_add_label2),axis=1)
            base_label2=base_regressor2(first_add_data2).data.numpy()
            base_label1=base_regressor1(first_add_data2).data.numpy()
            #temp_label2=safer(first_add_label2,second_add_label2)
            if i<30:
                temp_label2=safer(semi_label2,base_label2)
            else:
                temp_label2=add_label2
            
            add_label2=temp_label2
        else:        
            first_add_data2=torch.from_numpy(add_data2).float()
            first_add_label2=regressor2(first_add_data2).data.numpy()
            add_label2=(add_label2+first_add_label2)/2
        
        #Step 2.3 Update the regressor2.
        temp_train_data2=np.concatenate((data2,add_data2),axis=0)
        temp_train_label2=np.concatenate((label2,add_label2),axis=0)
        regressor2=model.train_iteration(regressor2,temp_train_data2,temp_train_label2,100,learning_rate) 
        #regressor2=regressor2.fit(temp_train_data2,temp_train_label2)
        temp_data2=copy.deepcopy(add_data2)
        temp_label2=copy.deepcopy(add_label2)
        
        #Step 2.4 Remove selected unlabeled instances in the unlabeled data set.
        for j,tempadd_index in enumerate(delete_index):
            #print(delete_index)
            unlabel_data=np.delete(unlabel_data,tempadd_index,axis=0)
        for j,tempadd_instance in enumerate(add_unlabeldata):
            #print(delete_index)
            tempadd_instance=tempadd_instance.reshape(1,-1)
            unlabel_data=np.concatenate((unlabel_data,tempadd_instance),axis=0)
        
         #Step 2.5 Regressor2 selected confident instances for regressor1.
        temp_train_data2=np.concatenate((data2,add_data2),axis=0)
        temp_train_label2=np.concatenate((label2,temp_label2),axis=0)
        add_data1,add_label1,delete_index,add_unlabeldata=Cotrainer.treat_selectCriticalInstances(regressor2,kValue2,temp_train_data2,temp_train_label2,unlabel_data,temp_data1,temp_label1,poolSize,i,0)
        
        #Step 2.6 Learn the safe pseudo labels for selected unlabeled instances.
        if i!=0 and i<100:
            second_add_data1=torch.from_numpy(add_data1).float()
            first_add_label1=regressor1(second_add_data1).data.numpy()
            second_add_label2=regressor2(second_add_data1).data.numpy()
            semi_label1=np.concatenate((add_label1,first_add_label1),axis=1)
            base_label1=base_regressor1(second_add_data1).data.numpy()
            base_label2=base_regressor2(second_add_data1).data.numpy()
            #temp_label1=safer(first_add_label1,second_add_label2)
            if i<30:
                temp_label1=safer(semi_label1,base_label1)
            else:
                temp_label1=add_label1
            
            add_label1=temp_label1
        else:
            
            second_add_data1=torch.from_numpy(add_data1).float()
            first_add_label1=regressor1(second_add_data1).data.numpy()
            add_label1=(add_label1+first_add_label1)/2
        
        #Step 2.7 Update the regressor1.
        temp_train_data1=np.concatenate((data1,add_data1),axis=0)
        temp_train_label1=np.concatenate((label1,add_label1),axis=0)
        regressor1=model.train_iteration(regressor1,temp_train_data1,temp_train_label1,100,learning_rate)
        temp_data1=copy.deepcopy(add_data1)
        temp_label1=copy.deepcopy(add_label1)   
        
        #Step 2.8 Remove selected unlabeled instances in the unlabeled data set.
        for j,tempadd_index in enumerate(delete_index):
            #print(delete_index)
            unlabel_data=np.delete(unlabel_data,tempadd_index,axis=0)
        for j,tempadd_instance in enumerate(add_unlabeldata):
            #print(delete_index)
            tempadd_instance=tempadd_instance.reshape(1,-1)
            unlabel_data=np.concatenate((unlabel_data,tempadd_instance),axis=0)
        #Step 2.9 Log the mean square error.
        if i%10==0:                
            log_mse[0][int(i/10)]=regressor_mse(regressor1,regressor2,test_data,test_label,0)
            
    #Step 3. Concatenate the labeled instances and selected unlabeled instances
    add_data2=add_data2[:,0:add_data2.shape[1]]
    add_data1=add_data1[:,0:add_data1.shape[1]]
    data2=np.concatenate((train_data2,add_data2),axis=0)
    label2=np.concatenate((train_label2,add_label2),axis=0)
    data1=np.concatenate((train_data1,add_data1),axis=0)
    label1=np.concatenate((train_label1,add_label1),axis=0)
    
    #Step 3.1 Update and return the regressors according to expanded training set.  
    print(numaddinstance1,numaddinstance2)    
    regressor1=model.train_iteration(regressor1,data1,label1,1000,learning_rate)
    regressor2=model.train_iteration(regressor2,data2,label2,1000,learning_rate)
    #regressor2=regressor2.fit(data2,label2)
    log_mse[0][log_mse.shape[1]-1]=regressor_mse(regressor1,regressor2,test_data,test_label,0)
    return data1,label1,data2,label2,regressor1,regressor2,log_mse 

 

def regressor_mse(regressor1,regressor2,test_data,test_label,mode):
    """
    Compute the average mean squared error of regressor1 and regressor2.
    
    Parameters
    ----------
    regressor1:   The first regressor1.
    regressor2:   The first regressor2.
    test_data:    The test data.
    test_label:   The label of test_data.
    mode:         The characteristic of regressor type, 0 is basic net regressor.
        
    Return
    ----------
    result_mse:   The average mean squared error.
    
    """
    temp_testdata=torch.from_numpy(test_data)
    temp_pre1=regressor1(temp_testdata)
    temp_pre1=temp_pre1.data.numpy()
    if mode==0:
        temp_pre2=regressor2(temp_testdata)
        temp_pre2=temp_pre2.data.numpy()        
    else:
        temp_pre2=regressor2.predict(test_data).reshape(-1,1)
        
    temp_pre=(temp_pre1+temp_pre2)/2
    result_mse=mse(temp_pre,test_label)
    
    return result_mse

def read_data(path,label_Index,unlabel_Index):\
    
    """
    Load and splite data from path.
    
    Parameters
    ----------
    path:         The path of data.
    label_Index:  The number of labeled instances.
    unlabel_Index:The number of unlabeled instances
        
    Return
    ----------
    train_data:   The training data for regressors.
    train_label:  The real label of train_data.
    unlabel_data: Unlabeld data that will be explored in the future training process.
    unlabel_label:The real label of unlabel_data.
    test_data:    Test data for regressors.
    test_label:   The label of test_data.
    num_features: The number of features.
    
    """
    #Step 1. Load the data from document path.
    data=pd.read_csv(path)
    all_features=data.iloc[:,0:data.shape[1]-1]
    all_labels=data.iloc[:,data.shape[1]-1:data.shape[1]]
    
    #Step 2. Normalize the data
    all_features = all_features.apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    all_labels=all_labels.apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    num_index=all_features.shape[0]
    num_features=all_features.shape[1]
    
    #Step 3. Splite data with a certain ratio.
    data=all_features[0:num_index].values.astype(np.float32)
    label=all_labels[0:num_index].values.astype(np.float32)
    train_data=data[0:label_Index,:]
    train_label=label[0:label_Index,:]
    unlabel_data=data[label_Index:unlabel_Index,:]
    unlabel_label=label[label_Index:unlabel_Index,:]
    test_data=data[unlabel_Index:data.shape[0],:]
    test_label=label[unlabel_Index:label.shape[0],:]
    return train_data,train_label,unlabel_data,unlabel_label,test_data,test_label,num_features

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')
    
    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')
    

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq,lb,ub)

    return np.array(sol['x'])  

  
def safer(candidate_prediction,baseline_prediction):
    '''
    Learn the safe prediction that not worse than baseline_prediction.

    Parameters
    ----------
    candidate_prediction : The semi-supervised predictions.
    baseline_prediction : The supervised predictions.

    Returns
    -------
    safer_prediction : The learned safe predictions.

    '''
    
    semi_pre=copy.deepcopy(candidate_prediction.astype(np.float64))
    supervised_pre=copy.deepcopy(baseline_prediction)
    prediction_num=candidate_prediction.shape[1]
    H=np.dot(semi_pre.T,semi_pre)*2
    f=-2*np.dot(semi_pre.T,supervised_pre)
    Aeq=np.ones((1,prediction_num))
    beq=1.0
    lb=np.zeros((prediction_num,1))
    ub=np.ones((prediction_num,1))
    sln=quadprog(H, f, None,None,Aeq, beq,)
    safer_prediction=np.zeros((semi_pre.shape[0],1))
    for i in range(safer_prediction.shape[0]):
        tempsafer=0
        for j in range(prediction_num):
            tempsafer=tempsafer+sln[j]*semi_pre[i,j]            
        safer_prediction[i][0]=tempsafer
    return safer_prediction

def main():
    '''
    The main function of SPOR

    Returns
    -------
    None.

    '''
    
    
    path1=os.getcwd()+'\data'# The data path
    data=['cpu_small','Folds5x2_pp','kin8nm','parkinsons','wind','wine_quality','space_ga','pollen','puma8NH','abalone']#The namelist of the test data sets. 

    for i in range(len(data)):
        dataname=data[i]
        
        #Prepare experimental settings.
        if i>1:
            lr1=0.01
            lr2=0.01
        else:
            lr1=0.1
            lr2=0.1
        #    
        for j in range(2):            
            log=np.zeros((1,11))
            result=0
            result1=0
            result2=0
            result3=0
            if j==0:
                label_index=50
            else:
                label_index=(j)*400
            
            for  k in range(20): 
                print(k)
                path=path1+'\\%s'%(dataname)+'.arff_{}.csv'.format(k)
                f= open(path,encoding='utf-8')
                
                #Splite the loaded data and train the base regressors.
                train_data,train_label,unlabel_data,unlabel_label,test_data,test_label,num_features=read_data(f,label_index,2000)
                net1 = model.Net(n_feature=num_features,n_hidden=32,n_hidden1=64,n_hidden2=32, n_output=1)
                net2 = model.Net(n_feature=num_features,n_hidden=32,n_hidden1=128,n_hidden2=32, n_output=1)
                
                net1=model.train_iteration(net1,train_data,train_label,3000,lr1)
                net2=model.train_iteration(net2,train_data,train_label,3000,lr1)
                net3=model.train_iteration(net1,train_data,train_label,3000,lr1)
                net4=model.train_iteration(net1,train_data,train_label,3000,lr1)
                if label_index>200 and i!=2:
                    lr2=0.1
                temp_test_data=torch.from_numpy(test_data)
                pre1=net1(temp_test_data).data.numpy()
                pre2=net2(temp_test_data).data.numpy()
                pre=(pre1+pre2)/2
            
                #Explore the unlabeled data and compute the mean squared error of regressors.
                train_data1,train_label1,train_data2,train_label2,net1,net2,log_mse=SPOR(net1,net2,net3,net4,3,train_data,train_label,3,train_data,train_label,unlabel_data,test_data,test_label,100,100,lr2)
                log=np.concatenate((log,log_mse),axis=0)
                test_pre1=net1(temp_test_data)
                test_pre2=net2(temp_test_data)
                test_pre1=test_pre1.data.numpy()
                test_pre2=test_pre2.data.numpy()
                temp_pre=(test_pre1+test_pre2)/2
                result+=mse(pre,test_label)
                result3+=mse(temp_pre,test_label)
                result1+=mse(test_pre1,test_label)
                result2+=mse(test_pre2,test_label)
            
            #Log the mean squared error.
            log=log[1:log.shape[0],:]
            log_dt=pd.DataFrame(log)
            logname=dataname+'_{}'.format(label_index)
            log_dt.to_csv(os.getcwd()+'\log'+'\\'+logname+'.csv',index=False)
            result=result/20
            result1=result1/20
            result2=result2/20
            result3=result3/20
    print(result,result1,result2,result3)
    
if __name__ == '__main__':
    main()
