# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:24:58 2017

@author: kisha_000
"""

import pandas as pd
import numpy as np


def compute_jump_matrix(Q):

    size = len(Q)

    P = np.zeros((size,size))

    for i in range(0,size):
        q = -Q[i][i]
        P[i][:] = Q[i][:]
        if(q==0):
            P[i][i] = 1
        else:
            P[i][i] = 0
            P[i][:] = P[i][:]/sum(P[i][:])

    return P


def ctmc_sequences(n,pi,Q,P,n_sequences):
    #Obtain a sample path with n events for a continuous-time Markov chain with
    #initial distribution pi and generator matrix Q. The procedure is repetead n_sequences 
    #times to obtain sequences 
    
    #dictionary used to replace state by alphabetical order letters
    di = {0: "A", 1: "B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J"}
    
    #initialize the dataframe structure
    df = pd.DataFrame(index = range(0,n_sequences),columns = ['id_patient','aux_encode'])
    df = df.fillna(0)

    ns = len(Q) #number of states
    
    for j in range(0,n_sequences): 
        df.loc[j,'id_patient'] = j
        #initialize the process at t=0 with initial state y_0 drawn from pi
        t = 0
        i = int(np.random.choice(ns,1,p=pi)) #initial state
        y_0 = i
        df.loc[j,'aux_encode'] = '0.' + di[y_0]
        for k in range(0,n-1):
            q = -Q[i][i]
            if(q==0):
                break
            else:
                s = float(np.random.exponential(q,1)) #exponential holding time
                t = t + s
                i = int(np.random.choice(ns,1,p=P[i][:]))
                df.loc[j,'aux_encode'] = df.loc[j,'aux_encode'] + ',' + str(round(s,2)) + '.' + str(di[i])
                
    return df


#generator matrix
#Q = np.zeros((7,7))
#Q[0][0:2] = [-1, 1]
#Q[1][0:3] = [2,-3,1]
#Q[2][1:4] = [2,-3,1]
#Q[3][2:5] = [2,-3,1]
#Q[4][3:6] = [2,-3,1]
#Q[5][4:7] = [2,-3,1]
#pi = [1,0,0,0,0,0,0]

# 2 SEQUENCES A->B
#Q = np.zeros((2,2))
#rate = 0.03
#Q[0][0:2] = [-rate,rate]
#pi = [1,0]

# 3 SEQUENCES A->B->C
#Q = np.zeros((3,3))
#rate_1 = 0.01
#rate_2 = 0.1
#Q[0][0:2] = [-rate_1,rate_1]
#Q[1][1:3] = [-rate_2,rate_2]
#pi = [1,0,0]

# 3 SEQUENCES A->B->C with A->C also possible
#Q = np.zeros((3,3))
#rate_1 = 0.01
#rate_2 = 0.01
#rate_3 = 0.001
#Q[0][0:3] = [-(rate_1+rate_3),rate_1,rate_3]
#Q[1][1:3] = [-rate_2,rate_2]
#pi = [1,0,0]

#jump matrix
#P = compute_jump_matrix(Q)

#df2 = ctmc_sequences(5,pi,Q,P,5)
        