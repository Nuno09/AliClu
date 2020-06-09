# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:36:13 2018

@author: kisha_000
"""

import numpy as np
import pandas as pd
from synthetic_data import compute_jump_matrix, ctmc_sequences

def generate_dataset(n_sequences,dataset):
    
    #initialize list that will contain the auxliary dataframes to be concataneted
    concat = [] 
    
    ###############################################################
    #CLUSTER 1 and 2 - A->B
    ##############################################################
    if(dataset==1):
        clusters = 1
    else:
        clusters = 2
    # rates of the clusters
    rates = [1000,0.1]
    
    #generate sequences
    for i in range(0,clusters):
        
        alfa = [1,0] #initial distribution for the states
        Q = np.zeros((2,2)) # Q-matrix
        rate = rates[i]     #rate of the transition
        Q[0][0:2] = [-rate,rate] 
        P = compute_jump_matrix(Q)     #jump matrix
        df_aux = ctmc_sequences(5,alfa,Q,P,n_sequences) #temporal sequences
        concat.append(df_aux)

    ################################################################
    #CLUSTER 3 e 4 - A->B->C->D   
    ################################################################
    #number of clusters
    clusters = 2
    # rates of the clusters
    rate1 = [1,1,1]
    rate2 = [1000,1000,1000]
    rates = [rate1,rate2]
        
    #generate sequences
    for i in range(0,clusters):
        
        rate = rates[i]     #rate of the transition
        n_events = len(rate) + 1  #number of events in the sequence
        #initial distribution for the states
        alfa = [0]*n_events
        alfa[0] = 1 
        #Q-matrix
        Q = np.zeros((n_events,n_events)) 
        for i in range(0,len(rate)):    
            Q[i][i:i+2] = [-rate[i],rate[i]]
        #jump matrix
        P = compute_jump_matrix(Q)     
        #temporal sequences
        df_aux = ctmc_sequences(5,alfa,Q,P,n_sequences) 
        concat.append(df_aux)

    ################################################################
    #CLUSTER 5 - A->E->B->C
    ################################################################
    rate = [1000,1000,1000] #rate of the clusters
    n_events = 5  #number of events in the sequence
    #initial distribution for the states
    alfa = [0]*n_events
    alfa[0] = 1 
    #Q-matrix
    Q = np.zeros((n_events,n_events)) 
    Q[0][0] = -rate[0]
    Q[0][4] = rate[0]
    Q[1][1] = -rate[2]
    Q[1][2] = rate[2]
    Q[4][1] = rate[1]
    Q[4][4] = -rate[1]
    #jump matrix
    P = compute_jump_matrix(Q)     
    #temporal sequences
    df_aux = ctmc_sequences(5,alfa,Q,P,n_sequences) 
    concat.append(df_aux)
    
    df_encoded = pd.concat(concat,ignore_index = True)
    #numerate patients from 0 to N-1, where N is the number patients
    df_encoded['id_patient'] = df_encoded.index.tolist()
    df_encoded.to_csv('patient_temporal_sequences_experiment3.csv')
    #print(df_encoded)
    return df_encoded