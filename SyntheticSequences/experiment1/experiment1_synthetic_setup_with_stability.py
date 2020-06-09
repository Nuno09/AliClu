# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:42:47 2018

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
from cluster_stability import cluster_validation
from scipy.cluster.hierarchy import cut_tree
from synthetic_data import compute_jump_matrix, ctmc_sequences
from statistics import mean, stdev, median
from tabulate import tabulate
import itertools
import numpy as np
import pandas as pd
import pickle

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
#gap_values = [-0.1, 0, 0.1]
#gap_values = [0.5]
if(len(gap_values) == 11):
    print('Only positive gap values')
else:
    print('Negative and positive gap values')

#Temporal penalty for temporal penalty function of TNW Algorithm
T = 0.25

#number of bootstrap samples M for validation step
M = 250

#number of maximum clusters to analyze on validation step
max_K = 8

#distance metric used in hierarchical clustering
method = 'ward'
print(method)


###############################################################################
#           TEMPORAL SEQUENCE GENERATION - 2 SEQUENCES A->B
###############################################################################
#number of clusters
clusters = 3
print('number of clusters:',clusters)
# rates of the clusters 
rates = [1000,10,0.1]
print('rates:',rates)
#n_sequences/cluster
n_sequences_list = [5,15,25,50,100]
#n_sequences_list = [5,15]

# list to store final averages,median and std of clustering indices for each test
# with n_sequences_list
final_avgs_statistics = {k:{} for k in n_sequences_list}
final_stds_statistics = {k:{} for k in n_sequences_list}
final_medians_statistics = {k:{} for k in n_sequences_list}

#To store all stability measures for each cluster
cluster_stability_statistics = {k:{} for k in n_sequences_list}
for sequences in n_sequences_list:
    cluster_stability_statistics[sequences] = {k:{} for k in range(1,clusters+1)}
    for k in range(1,clusters+1):
        cluster_stability_statistics[sequences][k]['J_median'] = []
        cluster_stability_statistics[sequences][k]['D_median'] = []
        cluster_stability_statistics[sequences][k]['A_median'] = []
        cluster_stability_statistics[sequences][k]['J_avg'] = []
        cluster_stability_statistics[sequences][k]['D_avg'] = []
        cluster_stability_statistics[sequences][k]['A_avg'] = []
        cluster_stability_statistics[sequences][k]['J_std'] = []
        cluster_stability_statistics[sequences][k]['D_std'] = []
        cluster_stability_statistics[sequences][k]['A_std'] = []
        cluster_stability_statistics[sequences][k]['lenght'] = []
      
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
        
    # To store the computed statistics of clustering indices between partition generated and partition found
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
        for i in range(0,clusters):
            
            alfa = [1,0] #initial distribution for the states
            Q = np.zeros((2,2)) # Q-matrix
            rate = rates[i]     #rate of the transition
            Q[0][0:2] = [-rate,rate] 
            P = compute_jump_matrix(Q)     #jump matrix
            df_aux = ctmc_sequences(5,alfa,Q,P,n_sequences) #temporal sequences
            concat.append(df_aux)
        
        df_encoded = pd.concat(concat,ignore_index = True)
        #numerate patients from 0 to N-1, where N is the number patients
        df_encoded['id_patient'] = df_encoded.index.tolist()
        
         
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
            
            #reset indexes
            df_encoded = df_encoded.reset_index()
            #cluster stability
            cluster_stability = cluster_validation(M,method,int(final_k_results['k']),partition_found,df_encoded,results)    
            for k in range(0,clusters):
                cluster_stability_statistics[n_sequences][k+1]['J_median'].append(cluster_stability[0][k])
                cluster_stability_statistics[n_sequences][k+1]['D_median'].append(cluster_stability[1][k])
                cluster_stability_statistics[n_sequences][k+1]['A_median'].append(cluster_stability[2][k])
                cluster_stability_statistics[n_sequences][k+1]['J_avg'].append(cluster_stability[3][k])
                cluster_stability_statistics[n_sequences][k+1]['D_avg'].append(cluster_stability[4][k])
                cluster_stability_statistics[n_sequences][k+1]['A_avg'].append(cluster_stability[5][k])
                cluster_stability_statistics[n_sequences][k+1]['J_std'].append(cluster_stability[6][k])
                cluster_stability_statistics[n_sequences][k+1]['D_std'].append(cluster_stability[7][k])
                cluster_stability_statistics[n_sequences][k+1]['A_std'].append(cluster_stability[8][k])
                cluster_stability_statistics[n_sequences][k+1]['lenght'].append(cluster_stability[9][k])
    
    if(count_correct > 1):
        #statistics of clustering indices between partition original and found
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

        #statistics of cluster stability
        for k in range(0,clusters):
            cluster_stability_statistics[n_sequences][k+1]['J_median'].append([mean(cluster_stability_statistics[n_sequences][k+1]['J_median']),stdev(cluster_stability_statistics[n_sequences][k+1]['J_median'])])
            cluster_stability_statistics[n_sequences][k+1]['D_median'].append([mean(cluster_stability_statistics[n_sequences][k+1]['D_median']),stdev(cluster_stability_statistics[n_sequences][k+1]['D_median'])])
            cluster_stability_statistics[n_sequences][k+1]['A_median'].append([mean(cluster_stability_statistics[n_sequences][k+1]['A_median']),stdev(cluster_stability_statistics[n_sequences][k+1]['A_median'])])
            cluster_stability_statistics[n_sequences][k+1]['J_avg'].append([mean(cluster_stability_statistics[n_sequences][k+1]['J_avg']),stdev(cluster_stability_statistics[n_sequences][k+1]['J_avg'])])
            cluster_stability_statistics[n_sequences][k+1]['D_avg'].append([mean(cluster_stability_statistics[n_sequences][k+1]['D_avg']),stdev(cluster_stability_statistics[n_sequences][k+1]['D_avg'])])
            cluster_stability_statistics[n_sequences][k+1]['A_avg'].append([mean(cluster_stability_statistics[n_sequences][k+1]['A_avg']),stdev(cluster_stability_statistics[n_sequences][k+1]['A_avg'])])
            cluster_stability_statistics[n_sequences][k+1]['J_std'].append([mean(cluster_stability_statistics[n_sequences][k+1]['J_std']),stdev(cluster_stability_statistics[n_sequences][k+1]['J_std'])])
            cluster_stability_statistics[n_sequences][k+1]['D_std'].append([mean(cluster_stability_statistics[n_sequences][k+1]['D_std']),stdev(cluster_stability_statistics[n_sequences][k+1]['D_std'])])
            cluster_stability_statistics[n_sequences][k+1]['A_std'].append([mean(cluster_stability_statistics[n_sequences][k+1]['A_std']),stdev(cluster_stability_statistics[n_sequences][k+1]['A_std'])])
            cluster_stability_statistics[n_sequences][k+1]['lenght'].append([mean(cluster_stability_statistics[n_sequences][k+1]['lenght']),stdev(cluster_stability_statistics[n_sequences][k+1]['lenght'])])

    #percentage of correct decision
    ratio_correct = count_correct / n_experiment
    ratio_list.append(ratio_correct)
    
###############################################################################
#BAR PLOT OF PERCENTAGE OF CORRECT DECISION
###############################################################################
ind = range(0,len(n_sequences_list))
plt.bar(ind, ratio_list)
plt.xticks(ind, (n_sequences_list))
plt.ylabel('Percentage')
plt.xlabel('Number of sequences/cluster')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Percentage of correct number of clusters')
#plt.show()
if(len(gap_values) == 11):
    title = 'experiment1_synthetic_nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' + '_percentage.png'
else:
    title = 'experiment1_synthetic_nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' + '_percentage.png'

directory = 'experiment1_synthetic_results_with_stability/'
plt.savefig(directory+title, bbox_inches='tight')

###############################################################################
#Table of final statistics (between original and found clusters)
###############################################################################
table_avgs = []
table_stds = []
table_medians = []
for n_sequences in final_avgs_statistics.keys():
    if(final_avgs_statistics[n_sequences]):          
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
    else:
        table_avgs.append([n_sequences,'-','-','-','-','-'])
        table_stds.append([n_sequences,'-','-','-','-','-'])
        table_medians.append([n_sequences,'-','-','-','-','-'])

if(len(gap_values)==11):
    file_name = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' +'_statistics' + '.txt'
else:
    file_name = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' +'statistics' + '.txt'

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

###############################################################################
#Table of final cluster stability statistics
###############################################################################

if(len(gap_values)==11):
    file_name_cluster_stability = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' +'_cluster_stability' + '.txt'
else:
    file_name_cluster_stability = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' +'cluster_stability' + '.txt'

f = open(directory+file_name_cluster_stability,'w')
for n_sequences in n_sequences_list:
    table_cluster_stability = []
    f.write('Number of sequences/cluster: ' + str(n_sequences))
    f.write('\n')
    if(final_avgs_statistics[n_sequences]):
        for k in range(1,clusters+1):
            table_cluster_stability.append([str(k),str(round(cluster_stability_statistics[n_sequences][k]['lenght'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['lenght'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['J_median'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['J_median'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['D_median'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['D_median'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['A_median'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['A_median'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['J_avg'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['J_avg'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['D_avg'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['D_avg'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['A_avg'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['A_avg'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['J_std'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['J_std'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['D_std'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['D_std'][-1][1],4)),
                                            str(round(cluster_stability_statistics[n_sequences][k]['A_std'][-1][0],4))+'/'+str(round(cluster_stability_statistics[n_sequences][k]['A_std'][-1][1],4))])
    else:
        for k in range(1,clusters+1):
            table_cluster_stability.append(['-','-','-','-','-','-','-','-','-','-','-'])
            
    f.write(tabulate(table_cluster_stability, headers = ['Cluster Number','Lenght', 'J_median','D_median','A_median','J_avg','D_avg','A_avg','J_std','D_std','A_std']))
    f.write('\n \n')
f.close()

#store dictionary
if(len(gap_values)==11):
    file_name_cluster_dict = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' +'_cluster_stability' + '.pickle'
else:
    file_name_cluster_dict = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' +'cluster_stability' + '.pickle'
pickle_out = open(directory + file_name_cluster_dict,"wb")
pickle.dump(cluster_stability_statistics, pickle_out)
pickle_out.close()