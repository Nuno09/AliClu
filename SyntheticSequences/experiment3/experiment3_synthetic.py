# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:44:56 2018

@author: kisha_000
"""

# Validation of Experiment with synthetic data


from sequence_alignment import main_algorithm
from clustering import convert_to_distance_matrix, hierarchical_clustering
from hierarchical_validation import validation, final_decision
from dataset import generate_dataset
import itertools
import numpy as np
import pandas as pd

#np.random.seed(123)
np.seterr(all='raise')


###############################################################################
#               PARAMETERS
###############################################################################
#pre-defined scoring system for TNW Algorithm
match=1.
mismatch=-1.1
#initialize pre-defined scoring dictionary
s = {'OO': match}
#get all combinations of letters
comb = list(itertools.product('ABCDEFGHIJZ',repeat = 2))
#construct the pre-defined scoring system
for pairs in comb:
    if(pairs[0]==pairs[1]):
        s[pairs[0]+pairs[1]] = match
    else:
        s[pairs[0]+pairs[1]] = mismatch

#gap penalty for TNW Algorithm
#gap=0
#gap_values = np.linspace(-1,1,21)
#gap_values = np.linspace(0,1,11)
gap_values = [0]

#Temporal penalty for temporal penalty function of TNW Algorithm
T = 2

#number of bootstrap samples M for validation step
M = 250

#number of maximum clusters to analyze on validation step
max_K = 8

#distance metric used in hierarchical clustering
method = 'ward'


###############################################################################
#           TEMPORAL SEQUENCE GENERATION          
###############################################################################
#df_encoded = generate_dataset(5,1)
df_encoded = generate_dataset(50,1)

################################################################################
##            SEQUENCE ALIGNMENT, HIERARCHICAL CLUSTERING & VALIDATION
################################################################################
concat_for_final_decision = []
for gap in gap_values:
    
    print('GAP PENALTY:', gap)
    
    #pairwise sequence alignment results
    results = main_algorithm(df_encoded,gap,T,s,0)
    
    #reset indexes
    df_encoded = df_encoded.reset_index()
    
    #convert similarity matrix into distance matrix
    results['score'] = convert_to_distance_matrix(results['score'])
    
    #exception when all the scores are the same, in this case we continue with the next value of gap
    if((results['score']== 0).all()):
        print('entrei')
        continue
    else:
        #hierarchical clustering
        Z = hierarchical_clustering(results['score'],method,gap)
        
        #validation
        chosen = validation(M,df_encoded,results,Z,method,max_K+1)
        chosen_k = chosen[2]
        df_avgs = chosen[0]
        df_stds = chosen[1]
        
        chosen_results = df_avgs.loc[chosen_k]
        chosen_results['gap'] = gap
        concat_for_final_decision.append(chosen_results)

df_final_decision = pd.concat(concat_for_final_decision,axis=1).T
final_k_results = final_decision(df_final_decision)
