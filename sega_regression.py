# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:28:10 2019
Tested on Tue Apr 25 21:28:10 2023

@author: Yiliao
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model,metrics,neighbors,kernel_ridge
from sklearn.base import clone
import math
import scipy.io
import matplotlib.pyplot as plt
import time

class IncLDD_R:
    """Incremental Local Drift Degree Data Stream Regression"""
    
    def __str__(self):
        return "Class: Inc_LDD_stream_regression"
    def __init__(self, dataset, base_learner, knowledge_base_size=1000, win_size=100,para_KNN = 100):        
        self.data = dataset
        self.learner = base_learner 
        self.win_size = win_size         
        self.kb_size = knowledge_base_size
        self.KNN_size = para_KNN
        self.predictions = []       
    
    def run_online(self):
        data = self.data
        n_size_data = self.data.shape[0]        
        knowledgebase = self.data[:self.kb_size, :] #training data
        num_kb_win = math.ceil(self.kb_size/self.win_size)
        error_debug = []
        
#        kb_data_dic={}
#        kb_ssd_dic={}              
        #print(kf)
        predictions = []
        buffer = np.ndarray(shape=(0,data.shape[1]), dtype=float) # define the size of the window                 
        
        #initialization    
        kb_data_list = np.split(knowledgebase,num_kb_win,axis=0)    
        knowledgebase_norm,min_max_data = scale_linear_bycolumn(knowledgebase)
        kb_ssd_array, dist_indx = compSSD(knowledgebase_norm,num_kb_win,self.KNN_size)
        kb_ssd_array_0 = kb_ssd_array
        
        learner_list = []
        for learn_data  in kb_data_list:
            _learner = self.learner
            learner_list.append(clone(_learner).fit(learn_data[:,:-1],learn_data[:,-1]))
                    
        K_uq = np.zeros(shape=(self.kb_size))
        K_vp = np.zeros(shape=(num_kb_win))
        Oq = np.ndarray(shape = 0)
        
        
        for i in range(self.kb_size, n_size_data):
            
            min_ssd_idx = np.argsort(kb_ssd_array)[0:2]
   
            learner_predictions = []                
            for _learner_indx in min_ssd_idx:
                learner_predictions.append(learner_list[_learner_indx].predict([data[i,:-1]])[0])
                #update learner------------------------------------------------
                learn_data = np.vstack([kb_data_list[_learner_indx], data[i,:]])
                learner_list[_learner_indx] = learner_list[_learner_indx].fit(learn_data[:,:-1],learn_data[:,-1])
                #-------------------------------------------------------------- 
            error_debug.append(learner_predictions)
            predictions.append(sum(learner_predictions)/len(learner_predictions))
    
            K_vp, K_uq, Oq = incrOnS(knowledgebase_norm, data[i,:], min_max_data, dist_indx, num_kb_win, K_uq, K_vp, Oq, self.KNN_size)         
            deltaSSD = -K_vp - (1+1/num_kb_win)*np.sum(np.split(K_uq,num_kb_win),axis=1)
            kb_ssd_array = deltaSSD #+ kb_ssd_array
            kb_ssd_array_0 = deltaSSD + kb_ssd_array_0

            
            buffer = np.vstack([buffer, data[i,:]])
            
            
            if buffer.shape[0] == self.win_size:
    
               # update knowledgebase
    
               knowledgebase =  np.concatenate((knowledgebase,buffer),axis = 0)[-self.kb_size:,:]
               
               learner_buffer = clone(self.learner).fit(buffer[:,:-1],buffer[:,-1])
               learner_list.append(learner_buffer)
               del learner_list[0]
               
               buffer = np.ndarray(shape=(0,data.shape[1]), dtype=float)
               
               kb_data_list = np.split(knowledgebase,num_kb_win,axis=0)    
               knowledgebase_norm,min_max_data = scale_linear_bycolumn(knowledgebase)
               kb_ssd_array, dist_indx = compSSD(knowledgebase_norm,num_kb_win,self.KNN_size)
               kb_ssd_array_0 = kb_ssd_array

               K_uq = np.zeros(shape=(self.kb_size))
               K_vp = np.zeros(shape=(num_kb_win))
               Oq = np.ndarray(shape = 0)
               
               
                    
        acc = metrics.mean_absolute_error(self.data[self.kb_size:n_size_data, -1], predictions)

        return acc,self.data[self.kb_size:n_size_data, -1:],np.asarray(predictions),np.asarray(error_debug)
       
def incrOnS(knowledgebase_norm,newinstance,min_max_data,dist_indx,num_kb_win,K_uq,K_vp,Oq,para_KNN=100):
    
    _instance = np.ndarray(shape=(1,len(newinstance))); _instance[0,:] = newinstance 
    # normalize
    _instance_norm, __ = scale_linear_bycolumn(_instance,min_max_data=min_max_data)
    
    Np = int(knowledgebase_norm.shape[0])
    Nq = int(knowledgebase_norm.shape[0]/num_kb_win)
    
    distances_list = dist_indx[0]
#    indices = dist_indx[1]
    indx_seg = dist_indx[2]
    nbrs = NearestNeighbors(n_neighbors=knowledgebase_norm.shape[0],algorithm = 'ball_tree').fit(knowledgebase_norm)
    _distances_0,_indices_0 = nbrs.kneighbors(_instance_norm)  
    _distances = np.zeros(shape=(1,_distances_0.shape[1]))
    _distances[0][_indices_0[0]] = _distances_0[0]
    _indices = _indices_0[0][:para_KNN]
    
    # incr K_vp
    delta_K_vp = np.ndarray(shape=(0))
    for _seg in indx_seg:
        _K_vp =len(np.intersect1d(_indices,_seg))
#        print(_K_vp)
        delta_K_vp = np.append(delta_K_vp,[_K_vp/Np/Nq],axis=0)
    K_vp = delta_K_vp #+ K_vp
    
    # incrK_uq
    
#    for ss in range(Np):
#        du = distances_list[ss]
#        d_vu = _distances[ss]
#        Kd = (du>d_vu).sum()
#        if Kd>1:
#            _Oq = len(du)+1-Kd
#            Oq = np.append(Oq,[_Oq],axis=0)
#            if len(du) <= max(Oq):
#               K_uq[ss] = K_uq[ss] + 1
#               del du[-1]
               
    return K_vp, K_uq, Oq    
        
def compSSD(knowledgebase_norm, num_kb_win, para_KNN = 100):    # the number of para_KNN determine how sensitive ssd to drift
    # the segments should be normalized before they input to comSSD  
    Np = int(knowledgebase_norm.shape[0] - knowledgebase_norm.shape[0]/num_kb_win)
    Nq = int(knowledgebase_norm.shape[0]/num_kb_win)
    # KNN of P
    fitbase = knowledgebase_norm
    nbrs = NearestNeighbors(n_neighbors=para_KNN,algorithm = 'ball_tree').fit(fitbase) #Ntrain-Nseg

    distances,indices = nbrs.kneighbors(fitbase)
    indicesP = indices[:Np,:]; indicesQ = indices[Np:,:];
    K_up = (indicesP<=Np).sum(1)  # |K_{u,P}(k)|
    K_uq = (indicesP>Np).sum(1)   # |K_{u,Q}(k)|
    # KNN of Q
    K_vq = (indicesQ>=Np).sum(1)
    K_vp = (indicesQ<Np).sum(1)
    # sd on every instance
    sd_P = (K_up/Np-K_uq/Nq)/Np
    sd_Q = (K_vq/Nq-K_vp/Np)/Nq
    sd = sd_P.sum()+sd_Q.sum()
    # compute ssd
    ssd_p = np.split(sd_P,num_kb_win-1,axis=0)
    ssd = np.zeros(shape = num_kb_win)
    ssd[0:-1] = np.sum(ssd_p,1)+sd_Q.sum()/(num_kb_win-1)
    
    indx_base = np.arange(0,knowledgebase_norm.shape[0],1)
    indx_seg = np.split(indx_base,num_kb_win,axis=0) 
    distances_list = np.ndarray.tolist(distances)
    dist_indx = [distances_list,indices,indx_seg]
    return ssd, dist_indx  #output ssd and the distance matrix
            
def scale_linear_bycolumn(data, high=1, low=0,  min_max_data=None):
    if min_max_data is None:
        mins_data = np.min(data, axis=0)
        maxs_data = np.max(data, axis=0)
        avg_data = np.mean(data, axis=0)
        std_data = np.std(data, axis = 0)
#        std_data[std_data == 0] = 1
    else:
        mins_data = min_max_data[0]
        maxs_data = min_max_data[1]
        avg_data = min_max_data[2]
        std_data = min_max_data[3]
    rng = maxs_data - mins_data
    rng[rng==0]=1
    #normalize 
    data_norm = high - (((high - low) * (maxs_data - data)) / rng)

    data_norm[:,-1] = data_norm[:,-1]*(data.shape[1]-1)

    return data_norm, [mins_data, maxs_data, avg_data, std_data]

        
  
    
if __name__=="__main__":
    experiment_type = ["regression_synthetic", "regression_real"]
    _experiment_type = experiment_type[1]
  
    path_regression = trainpath = "../data/regression/"
    if "real" in _experiment_type:
        #     real-world data
        datamat = scipy.io.loadmat(path_regression+"Datastream.mat")
        data_all = datamat["DataStreamP"].tolist()[0]
        dataset_list = ['house', 'CCPP', 'Sensor3', 'Sensor8', 'Sensor20', 'Sensor46', 'SMEAR', 'Solar']
        a            = [  1e-06,   1e-04  , 1e-04  ,  1e-03   ,  1e-04    ,  1e-04   ,  1e-02  , 1e-01 ]
        win_size_list = [  200,      200,    200,      200,       200,        200,       400,    200]
        hist_size_list = [  10,       10,     10,      10,         10,         10,       10,      10]
        hist_size_list = [int(c*1.5) for c in hist_size_list]
        run_list = [0,1,2,3,4,5,6,7] 
    else:
        # synthetic data
        datamat = scipy.io.loadmat(path_regression+"SynData.mat")
        data_all = datamat["DataStreamP"].tolist()[0]
        dataset_list = ['Non-Drift', 'Vir-Drift', 'Sudd-Drift', 'Incr-Drift', 'Rec-Drift-Grad', 'Rec-Drift-Mix']
        a       =     [1e-02, 1e-02 , 1e-02 ,  1e-02,  1e-02,  1e-02]
        win_size_list = [100, 100, 100, 100, 100, 100]
        hist_size_list = [10, 10, 10, 10, 10, 10]
        run_list = [0,1,2,3,4,5]

    mean_list = []
    std_list = []       
    acc_list_dict = {}
    run_time = []
    run_list = run_list
    output_list_dataset = []
    output_list_predvalue = []
    output_list_runtime = []
    output_list_accuracy = []
    debug_list = []       
    for i in run_list:          
        win_size = win_size_list[i]
        knowledge_base_size = hist_size_list[i]*win_size
        
        _data = data_all[i]
        _a = a[i];
        ridge = linear_model.Ridge(alpha=_a,fit_intercept=True,normalize=False)
        baysian = linear_model.BayesianRidge(tol=0.001, alpha_1=1e-06)    
        kernelridge = kernel_ridge.KernelRidge(alpha=_a)
        neigh = neighbors.KNeighborsRegressor(n_neighbors=5)
        
        _predictor = ridge
        time_start = time.time()    
        _incLDD_R = IncLDD_R(_data, _predictor, knowledge_base_size,win_size, para_KNN = 100)   
        
        acc_result,realvalue,predvalue , error_debug = _incLDD_R.run_online()
        time_end = time.time()
        print(dataset_list[i], time_end-time_start)
        print(acc_result)
        output_list_dataset.append(realvalue)
        output_list_accuracy.append(acc_result)
        output_list_predvalue.append(predvalue)
        debug_list.append(error_debug)

#plt.plot(output_list_dataset[0])    
#plt.plot(output_list_predvalue[0])
#plt.show()