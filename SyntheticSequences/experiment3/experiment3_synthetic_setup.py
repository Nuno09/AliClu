# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:44:56 2018

@author: kisha_000
"""

# Validation of Experiment with synthetic data

import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
from sequence_alignment import main_algorithm
from clustering import convert_to_distance_matrix, hierarchical_clustering
from hierarchical_validation import validation, final_decision
from clustering_scores import cluster_indices, cluster_external_index
from scipy.cluster.hierarchy import cut_tree
from dataset import generate_dataset
from statistics import mean, stdev, median
from tabulate import tabulate
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
gap_values = np.linspace(-1,1,21)
#gap_values = np.linspace(0,1,11)
#gap_values = [0.5]

#string for the filename
if(len(gap_values)==11):
    string_gap = '_positive_gap_'
else:
    string_gap = '_all_gap_'
print(string_gap)
    
#Temporal penalty for temporal penalty function of TNW Algorithm
T = 2
print('Tp:',T)

#number of bootstrap samples M for validation step
M = 250

#number of maximum clusters to analyze on validation step
max_K = 8

#distance metric used in hierarchical clustering
method = 'ward'
print(method)


###############################################################################
#           TEMPORAL SEQUENCE GENERATION - 2 SEQUENCES A->B->C->D
#                                                      A------->D
###############################################################################
dataset=2
print('dataset:',dataset)
#number of clusters
if(dataset==1):
    clusters = 4
else:
    clusters = 5
#n_sequences/cluster
n_sequences_list = [5,15,25,50,100]
#n_sequences_list = [5]

# list to store final averages,median and std of clustering indices for each test
# with n_sequences_list
final_avgs_statistics = {k:{} for k in n_sequences_list}
final_stds_statistics = {k:{} for k in n_sequences_list}
final_medians_statistics = {k:{} for k in n_sequences_list}

#list to store ratio of correct decision for each test with n_sequences_list 
ratio_list = []

for n_sequences in n_sequences_list:
        
    print(n_sequences)
    #generated clusters indexes
    partition_generated = []
    for i in range(0,clusters):
        if(i == 0):
            start = 0
            end = n_sequences
        else:
            start = i*n_sequences
            end = start + n_sequences
        cluster_aux = list(range(start,end))
        partition_generated.append(cluster_aux)
        
    # To compute statistics of clustering indices between partition generated and partition found
    final_statistics = {'Rand':[], 'Adjusted Rand':[], 'Fowlkes and Mallows':[],
                       'Jaccard':[], 'Adjusted Wallace':[]}
    
     
    #number of times to repeat experiment
    n_experiment = 25
    #count the number of times the procedure retrieves the correct number of clusters
    count_correct = 0
    
    #REPEAT SEQUENCE GENERATION n_experiment TIMES 
    for i in range(0,n_experiment):
        
        print('number of experiment:',i)
        #initialize list that will contain the auxliary dataframes to be concataneted
        concat = [] 
        
        #generate sequences
        df_encoded = generate_dataset(n_sequences,dataset)
         
        ###########################################################################
        ##            SEQUENCE ALIGNMENT, HIERARCHICAL CLUSTERING & VALIDATION
        ###########################################################################
        concat_for_final_decision = []
        for gap in gap_values:
            #print(gap)

            #pairwise sequence alignment results
            results = main_algorithm(df_encoded,gap,T,s,0)
            
            #reset indexes
            df_encoded = df_encoded.reset_index()
            
            #convert similarity matrix into distance matrix
            results['score'] = convert_to_distance_matrix(results['score'])
            
            #exception when all the scores are the same, in this case we continue with the next value of gap
            if((results['score']== 0).all()):
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
        
        #count the number of times the decision is correct
        if(final_k_results['k'] == clusters):
            count_correct = count_correct + 1
            
            #alignment with the chosen gap
            results = main_algorithm(df_encoded,final_k_results['gap'],T,s,0)
            
            #convert similarity matrix into distance matrix
            results['score'] = convert_to_distance_matrix(results['score'])
            
            #hierarchical clustering
            Z = hierarchical_clustering(results['score'],method,gap)
            
            #compute clustering indices between partition_generated and partition_found
            c_assignments_found = cut_tree(Z,final_k_results['k'])
            partition_found = cluster_indices(c_assignments_found,df_encoded.index.tolist())
            computed_indexes = cluster_external_index(partition_generated,partition_found)
            final_statistics['Rand'].append(computed_indexes[0])
            final_statistics['Adjusted Rand'].append(computed_indexes[1])
            final_statistics['Fowlkes and Mallows'].append(computed_indexes[2])
            final_statistics['Jaccard'].append(computed_indexes[3])
            final_statistics['Adjusted Wallace'].append(computed_indexes[4])
    
        
    if(count_correct > 1):
        
        final_avgs_statistics[n_sequences]['Rand'] = mean(final_statistics['Rand'])
        final_avgs_statistics[n_sequences]['Adjusted Rand'] = mean(final_statistics['Adjusted Rand'])
        final_avgs_statistics[n_sequences]['Fowlkes and Mallows'] = mean(final_statistics['Fowlkes and Mallows'])
        final_avgs_statistics[n_sequences]['Jaccard'] = mean(final_statistics['Jaccard'])
        final_avgs_statistics[n_sequences]['Adjusted Wallace'] = mean(final_statistics['Adjusted Wallace'])
    
        final_stds_statistics[n_sequences]['Rand'] = stdev(final_statistics['Rand'])
        final_stds_statistics[n_sequences]['Adjusted Rand'] = stdev(final_statistics['Adjusted Rand'])
        final_stds_statistics[n_sequences]['Fowlkes and Mallows'] = stdev(final_statistics['Fowlkes and Mallows'])
        final_stds_statistics[n_sequences]['Jaccard'] = stdev(final_statistics['Jaccard'])
        final_stds_statistics[n_sequences]['Adjusted Wallace'] = stdev(final_statistics['Adjusted Wallace'])
        
        final_medians_statistics[n_sequences]['Rand'] = median(final_statistics['Rand'])
        final_medians_statistics[n_sequences]['Adjusted Rand'] = median(final_statistics['Adjusted Rand'])
        final_medians_statistics[n_sequences]['Fowlkes and Mallows'] = median(final_statistics['Fowlkes and Mallows'])
        final_medians_statistics[n_sequences]['Jaccard'] = median(final_statistics['Jaccard'])
        final_medians_statistics[n_sequences]['Adjusted Wallace'] = median(final_statistics['Adjusted Wallace'])


    #percentage of correct decision
    ratio_correct = count_correct / n_experiment
    ratio_list.append(ratio_correct)
    
    

#BAR PLOT OF PERCENTAGE OF CORRECT DECISION
ind = range(0,len(n_sequences_list))
plt.bar(ind, ratio_list)
plt.xticks(ind, (n_sequences_list))
plt.ylabel('Percentage')
plt.xlabel('Number of sequences/cluster')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Percentage of correct number of clusters')
#plt.show()
title = 'experiment3_synthetic_' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_percentage.png'
directory = 'experiment3_synthetic_results/'
plt.savefig(directory+title, bbox_inches='tight')

#Table of final statistics (between original and found clusters)
table_avgs = []
table_stds = []
table_medians = []
for n_sequences in final_avgs_statistics.keys():
    if(count_correct>1):          
        table_avgs.append([n_sequences,final_avgs_statistics[n_sequences]['Rand'],
                            final_avgs_statistics[n_sequences]['Adjusted Rand'],
                            final_avgs_statistics[n_sequences]['Fowlkes and Mallows'],
                            final_avgs_statistics[n_sequences]['Jaccard'],
                            final_avgs_statistics[n_sequences]['Adjusted Wallace']])
        table_stds.append([n_sequences,final_stds_statistics[n_sequences]['Rand'],
                            final_stds_statistics[n_sequences]['Adjusted Rand'],
                            final_stds_statistics[n_sequences]['Fowlkes and Mallows'],
                            final_stds_statistics[n_sequences]['Jaccard'],
                            final_stds_statistics[n_sequences]['Adjusted Wallace']])
        table_medians.append([n_sequences,final_medians_statistics[n_sequences]['Rand'],
                            final_medians_statistics[n_sequences]['Adjusted Rand'],
                            final_medians_statistics[n_sequences]['Fowlkes and Mallows'],
                            final_medians_statistics[n_sequences]['Jaccard'],
                            final_medians_statistics[n_sequences]['Adjusted Wallace']])
    elif(count_correct == 1):
        table_avgs.append([n_sequences,final_statistics['Rand'],
                            final_statistics['Adjusted Rand'],
                            final_statistics['Fowlkes and Mallows'],
                            final_statistics['Jaccard'],
                            final_statistics['Adjusted Wallace']])
        table_stds.append([n_sequences,'-','-','-','-','-'])
        table_medians.append([n_sequences,'-','-','-','-','-'])

    else:
        table_avgs.append([n_sequences,'-','-','-','-','-'])
        table_stds.append([n_sequences,'-','-','-','-','-'])
        table_medians.append([n_sequences,'-','-','-','-','-'])


file_name = 'experiment3_synthetic_' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_statistics.txt'
f = open(directory+file_name,'w')
f.write('Averages \n')
f.write(tabulate(table_avgs, headers=['Number of sequences/cluster', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write(' \n Standard Deviations \n')
f.write(tabulate(table_stds, headers=['Number of sequences/cluster', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write(' \n Medians \n')
f.write(tabulate(table_medians, headers=['Number of sequences/cluster', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write('\n Ratio list: \n ')
f.write(str(ratio_list))
f.close()
