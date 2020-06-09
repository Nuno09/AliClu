# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:56:45 2018

@author: kisha_000
"""

# Validation of Hierarchical Clustering

#from IPython.core.display import display, HTML
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree
from fastcluster import linkage
from clustering_scores import cluster_indices, cluster_external_index
from statistics import mean, stdev
from tabulate import tabulate
import pandas as pd
import numpy as np
import itertools

np.seterr(all='raise')

#Function that performs validation of hierarchical clustering as on the paper
#'On validation of Hierarchical Clustering'
#INPUT:
#       M: number of bootstrap samples
#       df_encoded: dataframe containing the temporal sequences
#       results: dataframe with all pairwise alignments
#       Z: result of hierarchical clustering on results
#       method: distance metric used on hierarchical clustering of results, this will be used
#               again for hierarchical clustering of the bootstrap samples
#       max_K: maximum number of clusters that we want to analyze
def validation(M,df_encoded,results,Z,method,max_K):    
    ##############################################################################
    # HOW MANY CLUSTERS?
    ###############################################################################
    # bootstrap method - sampling without replacement
    
    #dictionary to store all computed indexes for each number of clusters K=2,...max_K
    dicio_statistics = {k:{} for k in range(2,max_K)}
    for k in range(2,max_K):
        dicio_statistics[k]['rand'] = []
        dicio_statistics[k]['adjusted'] = []
        dicio_statistics[k]['FM'] = []
        dicio_statistics[k]['jaccard'] = []
        dicio_statistics[k]['adjusted_wallace'] = []
        
        
    #for each bootstrap sample
    for i in range(M):
        # sampling rows of the original data
        idx = np.random.choice(len(df_encoded), int((3/4)*len(df_encoded)), replace = False)
        idx = np.sort(idx)
        #get all the possible combinations between the sampled patients
        patient_comb_bootstrap = list(itertools.combinations(df_encoded.loc[idx,'id_patient'],2))
        patient_comb_bootstrap = pd.DataFrame(patient_comb_bootstrap,columns = ['patient1','patient2'])
        #extract the scores regarding the previous sampled combinations to be used in hierarchical clustering
        results_bootstrap = pd.merge(results, patient_comb_bootstrap, how='inner', on=['patient1','patient2'])
        # Hierarchical Clustering of the bootstrap sample
        Z_bootstrap = linkage(results_bootstrap['score'],method)
        
        #for each number of clusters k=2,...,max_K
        for k in range(2,max_K):
            c_assignments_original = cut_tree(Z,k)
            c_assignments_bootstrap = cut_tree(Z_bootstrap,k)
            #list of clusters for the clustering result with the original data
            partition_original = cluster_indices(c_assignments_original,df_encoded.index.tolist())
            #list of clusters for the clustering result with the bootstrap sample
            partition_bootstrap = cluster_indices(c_assignments_bootstrap,idx)
                       
            #compute 4 different cluster external indexes between the partitions
            computed_indexes = cluster_external_index(partition_original,partition_bootstrap)
            #print(computed_indexes)
            dicio_statistics[k]['rand'].append(computed_indexes[0])
            dicio_statistics[k]['adjusted'].append(computed_indexes[1])
            dicio_statistics[k]['FM'].append(computed_indexes[2])
            dicio_statistics[k]['jaccard'].append(computed_indexes[3])
            dicio_statistics[k]['adjusted_wallace'].append(computed_indexes[4])
    
    ###########################################################################
    #  DECISION ON THE NUMBER OF CLUSTERS
    # The correct number of clusters is the k that yield most maximum average values of
    # clustering indices.
    # Also the k found before needs to have a low value of standard deviation - it has to 
    # be the minimum between all k's or a value that is somehow still low compared to others
    ###########################################################################
    
    #dataframe that stores the clustering indices averages for each k
    df_avgs = pd.DataFrame(index = range(2,max_K),columns = ['k','Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace','k_score_avg'], dtype='float')
    #dataframe that stores the AR and AW indices standard deviations for each k
    df_stds = pd.DataFrame(index = range(2,max_K),columns = ['k','Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard','Adjusted Wallace'],dtype = 'float')
    
    #computing the means and standard deviations
    for k in range(2,max_K):
        df_avgs.loc[k]['k'] = k
        df_avgs.loc[k]['Rand'] = mean(dicio_statistics[k]['rand'])
        df_avgs.loc[k]['Adjusted Rand'] = mean(dicio_statistics[k]['adjusted'])
        df_avgs.loc[k]['Fowlkes and Mallows']= mean(dicio_statistics[k]['FM'])
        df_avgs.loc[k]['Jaccard']= mean(dicio_statistics[k]['jaccard'])
        df_avgs.loc[k]['Adjusted Wallace'] = mean(dicio_statistics[k]['adjusted_wallace'])
        df_avgs.loc[k]['k_score_avg'] = 0
        
        df_stds.loc[k]['k'] = k
        df_stds.loc[k]['Rand'] = stdev(dicio_statistics[k]['rand'])
        df_stds.loc[k]['Adjusted Rand'] = stdev(dicio_statistics[k]['adjusted'])
        df_stds.loc[k]['Fowlkes and Mallows']  =stdev(dicio_statistics[k]['FM'])
        df_stds.loc[k]['Jaccard'] = stdev(dicio_statistics[k]['jaccard'])
        df_stds.loc[k]['Adjusted Wallace'] = stdev(dicio_statistics[k]['adjusted_wallace'])
        #df_stds.loc[k]['k_score_std'] = 0
        #df_stds.loc[k]['k_score_std_2'] = 0
        
    #weights given to each clustering indice, Rand Index does not value as much as the other indices
    weights = {'Adjusted Rand': 1/4, 'Fowlkes and Mallows': 1/4,
                   'Jaccard':1/4, 'Adjusted Wallace':1/4}
    #found the maximum value for each clustering index and locate in which k it happens
    # compute the scores for each k as being the sum of weights whenever that k has maximums of clustering indices
    for column in df_avgs.drop(columns = ['k','Rand','k_score_avg']).columns:
        idx_max = df_avgs[column].idxmax()
        df_avgs.loc[idx_max]['k_score_avg'] = df_avgs.loc[idx_max]['k_score_avg'] + weights[column]
    
    #final number of clusters chosen by analysing df_avgs
    final_k = df_avgs['k_score_avg'].idxmax()
    
    #same approach followed as for df_avgs
#    for column in df_stds.drop(columns = ['k','k_score_std','k_score_std_2']).columns:
#        idx_min = df_stds[column].idxmin()
#        idx_min_2 = df_stds[column].nsmallest(2).idxmax()
#        df_stds.loc[idx_min]['k_score_std'] = df_stds.loc[idx_min]['k_score_std'] + weights[column]
#        df_stds.loc[idx_min_2]['k_score_std_2'] = df_stds.loc[idx_min_2]['k_score_std_2'] + weights[column]
#        
    #At least 3 clustering indices (except Rand) have to agree on same minimum
    # std for the chosen k above
#    if(df_stds.loc[final_k_avg]['k_score_std']>=3*(2/9)):
#        final_k= final_k_avg
#    elif(df_stds.loc[final_k_avg]['k_score_std_2']>=3*(2/9)):
#        final_k = final_k_avg
#    else:
#        #the final k changes to the second best score with the avgs
#        final_k_avg_2 = df_avgs['k_score_avg'].nlargest(2).idxmin()
#        if(df_stds.loc[final_k_avg_2]['k_score_std']>=3*(2/9)):
#            final_k = final_k_avg_2
#        else:
#            final_k = final_k_avg
#    
    #table_avgs= tabulate(df_avgs, headers='keys', tablefmt='psql', showindex=False)
    #print(table_avgs)
    #display(HTML(table_avgs))
    #table_stds= tabulate(df_stds, headers='keys', tablefmt='psql', showindex=False)
    #print(table_stds)
    
    #bar chart of standard deviation
    # Create a figure instance
    #plt.figure(2)
    #df_stds.drop(columns = 'k').plot.bar()
    #plt.show()
    
    #print('NUMBER OF CLUSTERS:',final_k)
    return [df_avgs,df_stds,final_k]

#Function to retrieve the final number of cluster
#This function is used after having retrieved all the information when the method was used
#for different variations of the gap penalty
def final_decision(df_final_decision):
    
    final_k = df_final_decision['k'].value_counts().idxmax()
    df_final_decision = df_final_decision[df_final_decision['k']==final_k]
    
    df_aux = pd.DataFrame(0,index = range(0,len(df_final_decision)),columns = ['k_score'],dtype = 'float')
    df_final_decision = df_final_decision.reset_index(drop = 'True')
    #weights given to each clustering indice, Rand Index does not value as much as the other indices
    weights = {'Adjusted Rand': 1/4, 'Fowlkes and Mallows': 1/4,
                   'Jaccard':1/4, 'Adjusted Wallace':1/4}
    #found the maximum value for each clustering index and locate in which k it happens
    # compute the scores for each k as being the sum of weights whenever that k has maximums of clustering indices
    for column in df_final_decision.drop(columns = ['k','Rand','k_score_avg','gap']).columns:
        idx_max = df_final_decision[column].idxmax()
        df_aux.loc[idx_max]['k_score'] = df_aux.loc[idx_max]['k_score'] + weights[column]
    
    #final number of clusters and best results
    final_k_results = df_final_decision.loc[df_aux['k_score'].idxmax()]
    
    return final_k_results
   