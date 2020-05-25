# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:56:45 2018

@author: kisha_000
"""

# Validation of Hierarchical Clustering

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree
from fastcluster import linkage
from clustering_scores import cluster_indices, cluster_external_index, CVNN,S_Dbw
from statistics import mean, stdev
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict

import sys
sys.path.insert(0, '/home/nuno/Documentos/IST/Tese/Clustereval')
import clustereval
indexes = ['Rand', 'Adjusted Rand', 'Fowlkes and Mallows', 'Jaccard', 'Adjusted Wallace', 'Van Dongen', 'Huberts',
           'Huberts Normalized', 'F-Measure', 'VI', 'Minkowski', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']

external_indexes = ['Rand', 'Adjusted Rand', 'Fowlkes and Mallows', 'Jaccard', 'Adjusted Wallace', 'Van Dongen', 'Huberts',
                   'Huberts Normalized', 'F-Measure', 'VI', 'Minkowski']

internal_indexes = ['CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']

min_indexes = ['Van Dongen', 'VI', 'Minkowski', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'SD']




#Function that performs validation of hierarchical clustering as on the paper
#'On validation of Hierarchical Clustering'
#INPUT:
#       M: number of bootstrap samples
#       df_encoded: dataframe containing the temporal sequences
#       results: dataframe with all pairwise alignments
#       Z: result of hierarchical clustering on results
#       method: distance metric used on hierarchical clustering of results, this will be used
#               again for hierarchical clustering of the bootstrap samples
#       min_K: minimum number of clusters that we want to analyze
#       max_K: maximum number of clusters that we want to analyze
def validation(M,df_encoded,results,Z,method,min_K,max_K,automatic,pp,gap,Tp):
    ##############################################################################
    # HOW MANY CLUSTERS?
    ###############################################################################
    # bootstrap method - sampling without replacement

    #dictionary to store all computed indexes for each number of clusters K=min_K,...max_K
    nn_history = defaultdict(dict)
    trees = defaultdict(dict)
    dicio_statistics = {k:{} for k in range(min_K,max_K)}

    for k in range(min_K,max_K):
        for index in indexes:
            dicio_statistics[k][index] = []

        c_assignments_original = cut_tree(Z, k)

        # list of clusters for the clustering result with the original data
        partition_original = cluster_indices(c_assignments_original, df_encoded.index.tolist())

        trees[k] = partition_original


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

        #for each number of clusters k=min_K,...,max_K
        for k, partition in trees.items():

            c_assignments_bootstrap = cut_tree(Z_bootstrap,k)
            #list of clusters for the clustering result with the bootstrap sample
            partition_bootstrap = cluster_indices(c_assignments_bootstrap,idx)
            #compute 4 different cluster external indexes between the partitions
            #computed_indexes = cluster_external_index(partition,partition_bootstrap)
            computed_indexes = clustereval.calculate_external(partition, partition_bootstrap)



            #print(computed_indexes)
            for pos, index in enumerate(external_indexes):
                dicio_statistics[k][index].append(computed_indexes[pos])

    for k, partition in trees.items():
        calc_idx = clustereval.calculate_internal(results[['patient1', 'patient2', 'score']], partition, k, trees[max_K - 1])
        for index in internal_indexes:
            dicio_statistics[k][index].append(calc_idx[index])
    ###########################################################################
    #  DECISION ON THE NUMBER OF CLUSTERS
    # The correct number of clusters is the k that yield most maximum average values of
    # clustering indices.
    # Also the k found before needs to have a low value of standard deviation - it has to
    # be the minimum between all k's or a value that is somehow still low compared to others
    ###########################################################################

    #dataframe that stores the clustering indices averages for each k
    col = indexes.copy()
    col.extend(['k', 'k_score_avg'])
    df_avgs = pd.DataFrame(index = range(min_K,max_K),columns = col, dtype='float')
    #dataframe that stores the AR and AW indices standard deviations for each k
    df_stds = pd.DataFrame(index = range(min_K,max_K),columns = col, dtype = 'float')

    #computing the means and standard deviations
    for k in range(min_K,max_K):
        df_avgs.loc[k]['k'] = k
        df_stds.loc[k]['k'] = k
        for index in indexes:
            if index not in internal_indexes:
                df_avgs.loc[k][index] = mean(dicio_statistics[k][index])
                df_stds.loc[k][index] = stdev(dicio_statistics[k][index])
            else:
                df_avgs.loc[k][index] = dicio_statistics[k][index][0]
                df_stds.loc[k][index] = dicio_statistics[k][index][0]

        df_avgs.loc[k]['k_score_avg'] = 0
        df_stds.loc[k]['k_score_std'] = 0

        #df_stds.loc[k]['k_score_std_2'] = 0

    #weights given to each clustering indice, Rand Index does not value as much as the other indices
    weights = {index: 1/len(indexes) for index in indexes}
    #found the maximum value for each clustering index and locate in which k it happens
    # compute the scores for each k as being the sum of weights whenever that k has maximums of clustering indices
    columns = df_avgs.columns
    analyzed_columns = columns[2:-3]
    for column in analyzed_columns:

        if column in min_indexes:
            idx_min = df_avgs[column].idxmin()
            df_avgs.loc[idx_min]['k_score_avg'] = df_avgs.loc[idx_min]['k_score_avg'] + weights[column]
            continue


        idx_max = df_avgs[column].idxmax()
        df_avgs.loc[idx_max]['k_score_avg'] = df_avgs.loc[idx_max]['k_score_avg'] + weights[column]

    #idx_min_s_dbw = df_avgs['s_dbw'].idxmin()
    #idx_min_cvnn = df_avgs['cvnn'].idxmin()
    #df_avgs.loc[idx_min_s_dbw]['k_score_avg'] = df_avgs.loc[idx_min_s_dbw]['k_score_avg'] + weights['s_dbw']
    #df_avgs.loc[idx_min_cvnn]['k_score_avg'] = df_avgs.loc[idx_min_cvnn]['k_score_avg'] + weights['cvnn']

    #final number of clusters chosen by analysing df_avgs
    final_k = df_avgs['k_score_avg'].idxmax()


    if(automatic==0 or automatic==1):

        fig1 = plt.figure(figsize=(10,5))
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('tight')
        ax.axis('off')
        #colLabels=df_avgs.loc[:, df_avgs.columns != 'k_score_avg'].columns
        colLabels1 = external_indexes.copy()
        colLabels1.append('k')
        cell_text1 = []
        for row in range(len(df_avgs)):
            cell_text1.append(df_avgs.iloc[row,list(range(len(external_indexes))) + [-2]].round(decimals=3))
        plt.title('Average values of eleven external indices \n gap: %.2f, Tp: %.2f, %s link' %(gap,Tp,method))
        plt.table(cellText=cell_text1, colLabels=colLabels1, loc='center',cellLoc='center',fontsize=20)
        pp.savefig(fig1)



        fig2 = plt.figure(3, figsize=(12, 7))
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('tight')
        ax.axis('off')
        # colLabels=df_avgs.loc[:, df_avgs.columns != 'k_score_avg'].columns
        colLabels2 = internal_indexes.copy()
        colLabels2.append('k')
        cell_text2 = []
        for row in range(len(df_avgs)):
            cell_text2.append(df_avgs.iloc[row, list(range(len(external_indexes), len(indexes))) + [-2]].round(decimals=3))
        plt.title('Average values of six internal indices \n gap: %.2f, Tp: %.2f, %s link' % (gap, Tp, method))
        plt.table(cellText=cell_text2, colLabels=colLabels2, loc='center', cellLoc='center', fontsize=20)
        pp.savefig(fig2)


        #bar chart of standard deviation - standard deviation of all measures
        # Create a figure instance
    #    plt.figure(2)
    #    df_stds.loc[:,df_stds.columns != 'k'].plot.bar(figsize=(15,8))
    #    plt.title('Standard deviation of five measures versus number of clusters',fontsize=25)
    #    plt.xlabel('Number of clusters',labelpad=20,fontsize=20)
    #    plt.ylabel('Standard deviation',labelpad=10,fontsize=20)
    #    plt.xticks(size = 20)
    #    plt.yticks(size = 20)
    #    plt.show()


        fig3 = plt.figure(4)
        df_stds.loc[:,'Adjusted Rand'].plot.bar(figsize=(15,8),color='forestgreen')
        plt.title('Standard deviation of Adjusted Rand versus number of clusters \n gap: %.2f, Tp: %.2f, %s link' %(gap,Tp,method),fontsize=25)
        plt.xlabel('Number of clusters',labelpad=20,fontsize=15)
        plt.ylabel('Standard deviation',labelpad=10,fontsize=15)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        #plt.show()

        pp.savefig(fig3)


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
    weights = {index: 1/len(indexes) for index in indexes}
    #found the maximum value for each clustering index and locate in which k it happens
    # compute the scores for each k as being the sum of weights whenever that k has maximums of clustering indices
    for column in df_final_decision.drop(columns = ['k','Rand','k_score_avg','gap']).columns:
        if column in min_indexes:
            idx_min = df_final_decision[column].idxmin()
            df_aux.loc[idx_min]['k_score'] = df_aux.loc[idx_min]['k_score'] + weights[column]
            continue
        idx_max = df_final_decision[column].idxmax()
        df_aux.loc[idx_max]['k_score'] = df_aux.loc[idx_max]['k_score'] + weights[column]

    #idx_min_s_dbw = df_final_decision['s_dbw'].idxmin()
    #df_aux.loc[idx_min_s_dbw]['k_score'] = df_aux.loc[idx_min_s_dbw]['k_score'] + weights['s_dbw']
    #idx_min_cvnn = df_final_decision['cvnn'].idxmax()
    #df_aux.loc[idx_min_cvnn]['k_score'] = df_aux.loc[idx_min_cvnn]['k_score'] + weights['cvnn']


    #final number of clusters and best results
    final_k_results = df_final_decision.loc[df_aux['k_score'].idxmax()]

    return final_k_results
