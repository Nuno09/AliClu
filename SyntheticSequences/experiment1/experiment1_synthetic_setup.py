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
from synthetic_data import compute_jump_matrix, ctmc_sequences
from statistics import mean, stdev, median
from tabulate import tabulate
import itertools
import numpy as np
import pandas as pd
import clustereval

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
min_K = 2

#distance metric used in hierarchical clustering
method = 'complete'
print(method)


###############################################################################
#           TEMPORAL SEQUENCE GENERATION - 2 SEQUENCES A->B
###############################################################################
#number of clusters
clusters = 2
print('number of clusters:',clusters)
# rates of the clusters 
rates = [10,1]
print('rates:',rates)
#n_sequences/cluster
n_sequences_list = [5,15,25,50,100]
# #n_sequences_list = [5]

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
    final_statistics_external = {'R':[], 'AR':[], 'FM':[], 'J':[], 'AW':[], 'VD':[], 'H':[],
           'H\'':[], 'F':[], 'VI':[], 'MS':[]}
    final_statistics_internal = {'CVNN':[], 'XB**':[], 'S_Dbw':[], 'DB*':[], 'S':[], 'SD':[]}
    
    ###############################################################################
    # write filename and start by writing the parameters on the .txt file
    ###############################################################################
    #file_name = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '.txt'
    ##write all important parameters on the first lines
    #f = open(file_name,'w')
    #f.write('PARAMETERS:\n')
    #text = ['Temporal Penalty Constant: ' + str(T), 'Number of Bootstrap Samples: ' + str(M),
    #            'Number of maximum clusters K to be analyzed: ' + str(max_K),
    #            'Distance metric used in hierarchical clustering: ' +str(method)]
    #for line in text:
    #    f.write('\n')
    #    f.write(line)
    #f.write('\n')
    ###############################################################################
    
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
        #write the generated sequences
        ###########################################################################
    #    f.write('GENERATED SEQUENCES:\n')
    #    f.write('Number of sequences in each cluster: ' + str(n_sequences) + '\n')
    #    f.write(tabulate(df_encoded, headers='keys', tablefmt='psql', showindex=False))
    #    f.write('\n \n')
        ###########################################################################
        
        ###########################################################################
        ##            SEQUENCE ALIGNMENT, HIERARCHICAL CLUSTERING & VALIDATION
        ###########################################################################
        concat_for_final_decision = []
        for gap in gap_values:
            
            #######################################################################
            #write gap penalty on .txt file
            #######################################################################
    #        f.write('Gap Penalty: ' + str(gap))
    #        f.write('\n')
            #######################################################################
            
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
                chosen = validation(M,df_encoded,results,Z,method,min_K, max_K+1)
                chosen_k = chosen[2]
                df_avgs = chosen[0]
                df_stds = chosen[1]
                
                ###################################################################
                #write information on .txt file about averages and standard deviations
                ###################################################################
    #            f.write('Clustering indices averages \n')
    #            f.write(tabulate(df_avgs, headers='keys', tablefmt='psql', showindex=False))
    #            f.write('\n')
    #            f.write('Clustering indices standard deviations \n')
    #            f.write(tabulate(df_stds, headers='keys', tablefmt='psql', showindex=False))
    #            f.write('\n')
    #            f.write('Chosen k: ' + str(chosen_k))
    #            f.write('\n')
                ###################################################################
                
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

            #for SD index
            c_assignments_found_max = cut_tree(Z, max_K)
            max_partition_found = cluster_indices(c_assignments_found_max, df_encoded.index.tolist())

            computed_indexes_external = clustereval.calculate_external(partition_generated,partition_found)
            computed_indexes_internal = clustereval.calculate_internal(results[['patient1', 'patient2', 'score']],
                                                                       partition_found, int(final_k_results['k']),
                                                                       max_partition_found)
            for key in final_statistics_external.keys():
                final_statistics_external[key].append(computed_indexes_external[list(final_statistics_external.keys()).index(key)])

            for key in final_statistics_internal.keys():
                final_statistics_internal[key].append(computed_indexes_internal[key])

    
        
        
        
        ###########################################################################
        #write the final decision table and final number of clusters
        ###########################################################################
    #    f.write('Final Decision: \n')
    #    f.write(tabulate(df_final_decision, headers='keys', tablefmt='psql', showindex=False))
    #    f.write('\n')
    #    f.write('Final Result: \n')
    #    f.write(tabulate(final_k_results.to_frame().T, headers='keys', tablefmt='psql', showindex=False))
    #    f.write('\n')
    #    f.write('Final : ' +str(final_k_results['k']))
    #    f.write('\n')
    #    f.write('---------------------------------------------------- \n')
        ###########################################################################
    
    if(count_correct > 1):
        for key in final_statistics_external.keys():
            final_avgs_statistics[n_sequences][key] = mean(final_statistics_external[key])
            final_stds_statistics[n_sequences][key] = stdev(final_statistics_external[key])
            final_medians_statistics[n_sequences][key] = median(final_statistics_external[key])
        for key in final_statistics_internal.keys():
            final_avgs_statistics[n_sequences][key] = mean(final_statistics_internal[key])
            final_stds_statistics[n_sequences][key] = stdev(final_statistics_internal[key])
            final_medians_statistics[n_sequences][key] = median(final_statistics_internal[key])


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
if(len(gap_values) == 11):
    title = 'experiment1_synthetic_nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' + '_percentage.png'
else:
    title = 'experiment1_synthetic_nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' + '_percentage.png'

directory = 'experiment1_synthetic_results/'
plt.savefig(directory+title, bbox_inches='tight')

external_indexes = ['R', 'AR', 'FM', 'J', 'AW', 'VD', 'H', 'H\'', 'F', 'VI', 'MS']

internal_indexes = ['CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']

#Table of final statistics (between original and found clusters) for external indixes
table_avgs_external = []
table_stds_external = []
table_medians_external = []
for n_sequences in final_avgs_statistics.keys():
    table_avgs_external.append([n_sequences])
    table_stds_external.append([n_sequences])
    table_medians_external.append([n_sequences])
    for index in external_indexes:
        table_avgs_external[-1].append(final_avgs_statistics[n_sequences][index])
        table_stds_external[-1].append(final_stds_statistics[n_sequences][index])
        table_medians_external[-1].append(final_medians_statistics[n_sequences][index])

# Table of final statistics for internal indixes
table_avgs_internal = []
table_stds_internal = []
table_medians_internal = []
for n_sequences in final_avgs_statistics.keys():
    table_avgs_internal.append([n_sequences])
    table_stds_internal.append([n_sequences])
    table_medians_internal.append([n_sequences])
    for index in internal_indexes:
        table_avgs_internal[-1].append(final_avgs_statistics[n_sequences][index])
        table_stds_internal[-1].append(final_stds_statistics[n_sequences][index])
        table_medians_internal[-1].append(final_medians_statistics[n_sequences][index])

if(len(gap_values)==11):
    file_name_external = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' +'_statistics_external' + '.txt'
else:
    file_name_external = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' +'statistics_external' + '.txt'

if(len(gap_values)==11):
    file_name_internal = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_positivegap' +'_statistics_internal' + '.txt'
else:
    file_name_internal = 'experiment1_synthetic_' + 'nclusters_' + str(clusters) + '_rates_' + str(rates) + '_' + str(method) +  '_' +'statistics_internal' + '.txt'

f = open(directory+file_name_external,'w+')
f.write('Averages \n')
f.write(tabulate(table_avgs_external, headers=['Number of sequences/cluster', 'R','AR','FM','J','AW','VD', 'H', 'H\'', 'F', 'VI', 'MS']))
f.write(' \n Standard Deviations \n')
f.write(tabulate(table_stds_external, headers=['Number of sequences/cluster', 'R','AR','FM','J','AW','VD', 'H', 'H\'', 'F', 'VI', 'MS']))
f.write(' \n Medians \n')
f.write(tabulate(table_medians_external, headers=['Number of sequences/cluster', 'R','AR','FM','J','AW','VD', 'H', 'H\'', 'F', 'VI', 'MS']))
f.write('\n Ratio list: \n ')
f.write(str(ratio_list))
f.close()

f = open(directory+file_name_internal,'w+')
f.write('Averages \n')
f.write(tabulate(table_avgs_internal, headers=['Number of sequences/cluster', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']))
f.write(' \n Standard Deviations \n')
f.write(tabulate(table_stds_internal, headers=['Number of sequences/cluster', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']))
f.write(' \n Medians \n')
f.write(tabulate(table_medians_internal, headers=['Number of sequences/cluster', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']))
f.write('\n Ratio list: \n ')
f.write(str(ratio_list))
f.close()
