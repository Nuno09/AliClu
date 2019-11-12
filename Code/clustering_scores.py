# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:26:07 2018

@author: kisha_000
"""


import numpy as np
import math
import time


def cluster_indices(cluster_assignments,idx):
    n = cluster_assignments.max()
    clusters = []
    for cluster_number in range(0, n + 1):
        aux = np.where(cluster_assignments == cluster_number)[0].tolist()
        cluster = list(idx[i] for i in aux )
        clusters.append(cluster)
    return clusters

def cluster_external_index(partition_a, partition_b):
    #size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    #contigency table
    ct = np.zeros((R+1,C+1))
    #fill the contigency table
    for i in range(0,R+1):
        for j in range(0,C):
            if(i in range(0,R)):  
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                ct[i][j] = n_common_elements
            else:
                ct[i][j] = ct[:,j].sum()
                      
        ct[i][j+1] = ct[i].sum()
    N = ct[R][C]
    #condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(ct[0:R][:,range(0,C)]**2)   
    sum_R_squared = np.sum(ct[0:R,C]**2)
    sum_R = np.sum(ct[0:R,C])
    sum_C_squared = np.sum(ct[R,0:C]**2)
    sum_C = np.sum(ct[R,0:C])
    #computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    for i in range(0,R):
        for j in range(0,C):
            a = a + ct[i][j]*(ct[i][j]-1)
    a = a/2
    #computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared- sum_all_squared)/2
    #computing the number of pair in different cluster in partition A but in the same cluster in partition B
    c = (sum_C_squared - sum_all_squared)/2
    #computing the number of pairs in different cluster both in partition A and partition B
    d = (N**2 + sum_all_squared - (sum_R_squared + sum_C_squared))/2
    
    #Rand Index
    rand_index = (a+d)/(a+b+c+d)

    #Adjusted Rand Index
    nc = ((sum_R_squared - sum_R)*(sum_C_squared -sum_C))/(2*N*(N-1))
    nd = (sum_R_squared - sum_R + sum_C_squared - sum_C)/4
    if(nd==nc):
        adjusted_rand_index = 0
    else:      
        adjusted_rand_index = (a-nc)/(nd - nc)
   
    #Fowlks and Mallows
    if((a+b)==0 or (a+c)==0):
        FM = 0
    else:     
        FM = a/math.sqrt((a+b)*(a+c))
    
    #Jaccard
    if(a+b+c == 0):
        jaccard = 1
    else:
        jaccard = a/(a+b+c)
        
    #Adjusted Wallace
    if((a+b)==0):
        wallace = 0
    else:
        wallace = a/(a+b)
    SID_B = 1-((sum_C_squared-sum_C)/(N*(N-1)))
    if((SID_B)==0):
        adjusted_wallace = 0
    else:
        adjusted_wallace = (wallace-(1-SID_B))/(1-(1-SID_B))

    return [rand_index, adjusted_rand_index, FM, jaccard, adjusted_wallace]


def adjusted_wallace(partition_a,partition_b):
        #size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    #contigency table
    ct = np.zeros((R+1,C+1))
    #fill the contigency table
    for i in range(0,R+1):
        for j in range(0,C):
            if(i in range(0,R)):  
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                ct[i][j] = n_common_elements
            else:
                ct[i][j] = ct[:,j].sum()
                      
        ct[i][j+1] = ct[i].sum()  
    
    N = ct[R][C]
    #condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(ct[0:R][:,range(0,C)]**2)   
    sum_R_squared = np.sum(ct[0:R,C]**2)
    sum_C_squared = np.sum(ct[R,0:C]**2)
    sum_C = np.sum(ct[R,0:C])
    #computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    for i in range(0,R):
        for j in range(0,C):
            a = a + ct[i][j]*(ct[i][j]-1)
    a = a/2
    #computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared- sum_all_squared)/2    
    #Adjusted Wallace
    SID_B = 1-((sum_C_squared-sum_C)/(N*(N-1)))
    wallace = a/(a+b)
    adjusted_wallace = (wallace-(1-SID_B))/(1-(1-SID_B))
    
    return adjusted_wallace

def CVNN(partition_a, results, k, nn_history):
    dict_partition = {i: partition_a[i] for i in range(len(partition_a))}
    separation_arr = []
    separation_sum = 0
    distance_sum = 0
    compactness_sum = 0
    for key, cluster in dict_partition.items():
        for obj in cluster:
            if obj in nn_history:
                if k in nn_history[obj]:
                    neighbours = nn_history[obj][k]
                    separation_sum += (len(neighbours) / k)
            else:
                #separation calculations
                query = 'patient1 == ' + str(obj) + ' or patient2 == ' + str(obj)
                filtered_res = results.query(query)
                outside_arr = cluster[:cluster.index(obj)] + cluster[cluster.index(obj)+1:]
                outside_query = 'patient1 != ' + str(outside_arr) + ' and patient2 != ' + str(outside_arr)
                outside_filtered_res = filtered_res.query(outside_query)
                #print("obj = " + str(obj) + "cluster = " + str(outside_arr))
                #print(outside_filtered_res)
                #tree = spatial.KDTree(outside_filtered_res[['patient1', 'patient2', 'score']].to_numpy())
                neighbours = getNN(obj, outside_filtered_res, k)
                nn_history[obj][k] = neighbours
                separation_sum += (len(neighbours) / k)

        #distance calculations(compactness)
        df_distance = results.query('patient1 == ' + str(cluster) + ' and patient2 == ' + str(cluster))
        distance_sum+=df_distance['score'].sum()

        separation_arr.append(separation_sum / len(cluster))
        compactness_sum += (2*distance_sum + 1) / (len(cluster)*(len(cluster) - 1) + 1)
        separation_sum = 0
        distance_sum = 0


    separation = max(separation_arr)
    compactness = compactness_sum

    metric = (separation / len(partition_a)) + (compactness / len(partition_a))
    return metric

def getNN(obj, results, k):
    neighbours = []
    # find max score(nearest neighbour)
    nn = results.nlargest(k, ['score'])
    for index, row in nn.iterrows():
        if row['patient1'] == obj:
            neighbours.append(row['patient2'])
        else:
            neighbours.append(row['patient1'])
    return neighbours

def S_Dbw(partition_a, results):
    partition_dict = {i: partition_a[i] for i in range(len(partition_a))}
    return Scat(partition_dict, results) + Dens_bw(partition_dict, results)

def Scat(d, results):
    sum_stds = 0
    for k, v in d.items():
        if len(v) > 2:
            scores_df = getCombination_Df(v, results)
            std = scores_df[['score']].std().values[0]
        else:
            std = 0
        sum_stds += std / results[['score']].std().values[0]
    return sum_stds / len(d)

def Dens_bw(d, results):
    sum_density_combinations = 0
    for k,v in d.items():
        sum_density = 0
        for k2, v2 in d.items():
            if k2 != k:
                union_of_clusters = v + v2
                union_of_clusters_df = getCombination_Df(sorted(union_of_clusters),results)

                if len(v) > 1:
                    cluster_i = getCombination_Df(v,results)
                    center_i = cluster_i.iloc[[(cluster_i.shape[0]) - 1 / 2]].score.values[0]
                else:
                    center_i = 0

                if len(v2) > 1:
                    cluster_j = getCombination_Df(v2, results)
                    center_j = cluster_j.iloc[[(cluster_j.shape[0]) - 1 / 2]].score.values[0]
                else:
                    center_j = 0


                center_u = abs(center_i - center_j)/2
                std_clusters = union_of_clusters_df['score'].std()
                for index, row in union_of_clusters_df.iterrows():
                    density_centers = max(calculate_density(row['score'], center_i, std_clusters), calculate_density(row['score'], center_j, std_clusters))
                    if density_centers != 0:
                        sum_density += calculate_density(row['score'],center_u,std_clusters) / density_centers
        sum_density_combinations += sum_density

    return sum_density_combinations

def getCombination_Df(A,results):
    df_p1 = results[results['patient1'].isin(A)]
    df = df_p1[df_p1['patient2'].isin(A)]
    return df

def calculate_density(x, u, std):
    distance = abs(u - x)
    if std - distance < 0:
        return 0
    else:
        return 1

def cluster_validation_indexes(cluster_a,cluster_b):
    #jaccard index
    num_jaccard = len(set(cluster_a).intersection(cluster_b))
    den_jaccard = len(set(cluster_a).union(cluster_b))
    jaccard = num_jaccard/den_jaccard
    
    #the asymmetric measure gama - rate of recovery
    gama = num_jaccard/len(cluster_a)
    
    #the symmetric Dice coefficient
    dice = num_jaccard/(len(cluster_a)+len(cluster_b))
    
    return [jaccard, gama, dice]


