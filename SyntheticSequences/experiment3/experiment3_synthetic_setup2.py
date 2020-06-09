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
#gap_values = np.linspace(0,1,11,dtype='float')
#gap_values = [0.5]

#string for the filename
if(len(gap_values)==11):
    string_gap = '_positive_gap_'
else:
    string_gap = '_all_gap_'
    
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
dataset=1
print('dataset:',dataset)
#number of clusters
if(dataset==1):
    clusters = 4
else:
    clusters = 5
#n_sequences/cluster
#n_sequences_list = [5,15,25,50,100]
n_sequences_list = [5,15,25,50]
#n_sequences_list = [5]

# this dictionary will help us to store the results of the final number
# of clusters for each value of gap
cluster_vs_gap_statistics = final_medians_statistics = {k:{} for k in n_sequences_list}
for key in cluster_vs_gap_statistics.keys():
    for gap in gap_values:
        cluster_vs_gap_statistics[key][gap] = {}

#dataframe
df_all_percentages = pd.DataFrame(0,index=gap_values,columns = n_sequences_list,dtype='float')

#list to store ratio of correct decision for each test with n_sequences_list 
ratio_list = []

for n_sequences in n_sequences_list:
        
    
    df_percentages = pd.DataFrame(0,index = gap_values,columns = range(2,max_K+1), dtype='float')
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
        
    #number of times to repeat experiment
    n_experiment = 25

    #REPEAT SEQUENCE GENERATION n_experiment TIMES 
    for i in range(0,n_experiment):
        
        print('number of experiment:',i)

        #generate sequences
        df_encoded = generate_dataset(n_sequences,dataset)
         
        ###########################################################################
        ##            SEQUENCE ALIGNMENT, HIERARCHICAL CLUSTERING & VALIDATION
        ###########################################################################
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
                
                #statistics
                df_percentages.loc[gap][chosen_k] +=1
                if(chosen_k == clusters):
                    if(chosen_k in cluster_vs_gap_statistics[n_sequences][gap].keys()):
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][0] += 1
                    else:
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k] = [1]
                
                    #compute clustering indices between partition_generated and partition_found
                    c_assignments_found = cut_tree(Z,chosen_k)
                    partition_found = cluster_indices(c_assignments_found,df_encoded.index.tolist())
                    computed_indexes = cluster_external_index(partition_generated,partition_found)
                    #print(computed_indexes)

                    if(len(cluster_vs_gap_statistics[n_sequences][gap][chosen_k])==1):
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k].append({'Rand':[computed_indexes[0]], 'Adjusted Rand':[computed_indexes[1]], 'Fowlkes and Mallows':[computed_indexes[2]],
                       'Jaccard':[computed_indexes[3]], 'Adjusted Wallace':[computed_indexes[4]]})
                    else:
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][1]['Rand'].append(computed_indexes[0])
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][1]['Adjusted Rand'].append(computed_indexes[1])
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][1]['Fowlkes and Mallows'].append(computed_indexes[2])
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][1]['Jaccard'].append(computed_indexes[3])
                        cluster_vs_gap_statistics[n_sequences][gap][chosen_k][1]['Adjusted Wallace'].append(computed_indexes[4])

        df_all_percentages.loc[:,n_sequences] = df_percentages.loc[:,clusters]
        
df_all_percentages = df_all_percentages/n_experiment
df_all_percentages.index = np.around(df_all_percentages.index,2)
df_all_percentages.plot.bar(figsize=(15,8)).legend(bbox_to_anchor=(1, 1),fontsize=15,title='Number of sequences per cluster')
plt.gca().yaxis.grid(True)
plt.ylim([0,1.1])
plt.title('Percentage of correct decisions on the number of clusters ',fontsize=25)
plt.xlabel('Gap penalty values',labelpad=20,fontsize=20)    
plt.ylabel('Percentage',labelpad=10,fontsize=20)    
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.show()
title = 'experiment3_synthetic_setup2_' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_percentage.png'
directory = 'experiment3_synthetic_results/'
plt.savefig(directory+title, bbox_inches='tight')

title_csv = 'experiment3_synthetic_setup2' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_dataframe.csv'
df_all_percentages.to_csv(directory+title_csv)

#table of final statistics (between original and found clusters)
table_avgs = []
table_stds = []
table_medians = []
for n_sequences in cluster_vs_gap_statistics.keys():
    for gap in cluster_vs_gap_statistics[n_sequences]:
        if(cluster_vs_gap_statistics[n_sequences][gap]):
            table_avgs.append([n_sequences,gap,mean(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Rand']),
                                mean(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Rand']),
                                mean(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Fowlkes and Mallows']),
                                mean(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Jaccard']),
                                mean(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Wallace'])])
            table_medians.append([n_sequences,gap,median(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Rand']),
                                median(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Rand']),
                                median(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Fowlkes and Mallows']),
                                median(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Jaccard']),
                                median(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Wallace'])])
            if(len(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Rand'])>1):
                table_stds.append([n_sequences,gap,stdev(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Rand']),
                                stdev(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Rand']),
                                stdev(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Fowlkes and Mallows']),
                                stdev(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Jaccard']),
                                stdev(cluster_vs_gap_statistics[n_sequences][gap][clusters][1]['Adjusted Wallace'])])
            else:
                table_stds.append([n_sequences,gap,'-','-','-','-','-'])
                
file_name = 'experiment3_synthetic_setup2' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_statistics.txt'
f = open(directory+file_name,'w')
f.write('Averages \n')
f.write(tabulate(table_avgs, headers=['Number of sequences/cluster','Gap', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write(' \n Standard Deviations \n')
f.write(tabulate(table_stds, headers=['Number of sequences/cluster', 'Gap', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write(' \n Medians \n')
f.write(tabulate(table_medians, headers=['Number of sequences/cluster','Gap', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace']))
f.write('\n \n \n Dataframe: \n  ')
f.write(str(df_all_percentages))
f.close()

#store dictionary
file_name_dict = 'experiment3_synthetic_setup2' + 'dataset' + str(dataset) + string_gap + 'Tp_' + str(T) + '_' + str(method) +  '_' + '_statistics.pickle'
pickle_out = open(directory + file_name_dict,"wb")
pickle.dump(cluster_vs_gap_statistics, pickle_out)
pickle_out.close()

#read dictionary
#pickle_in = open(file_name_dict,"rb")
#cluster_vs_gap_statistics = pickle.load(pickle_in)