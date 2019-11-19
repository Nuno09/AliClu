# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:26:07 2018

@author: kisha_000
"""


import numpy as np
import math
import pandas as pd


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
    sum_variance_vectors = 0
    variance_vector_dataset = results[['score']].var(ddof=0)
    variance_vector_dataset = variance_vector_dataset.to_numpy()

    for k, v in d.items():
        scores_df = getCombination_Df(v, results)
        variance_vector = scores_df[['score']].var(ddof=0)
        variance_vector = variance_vector.to_numpy()
        sum_variance_vectors += math.sqrt(np.dot(np.transpose(variance_vector),variance_vector)) / \
                                math.sqrt(np.dot(np.transpose(variance_vector_dataset),variance_vector_dataset))
    return sum_variance_vectors / len(d)

def Dens_bw(d, results):
    sum_density_combinations = 0
    for k,v in d.items():
        sum_density = 0
        v_dict = {i: v[i] for i in range(len(v))}
        for key, point in v_dict.items():
            mean_points = getCombination_Df([point], results)
            mean_points_dict = {point: mean_points['score'].mean()}
        centeri = getCenter(v_dict,results)
        for k2, v2 in d.items():
            if k2 != k:
                v2_dict = {j: v2[j] for j in range(len(v2))}
                centerj = getCenter(v2_dict,results)

                centeru = getPair(results, centeri, centerj)
                final_u = getNearest(centeru['score'].values[0] / 2, mean_points_dict)

                union_of_clusters = v + v2
                union_of_clusters_df = getCombination_Df(sorted(union_of_clusters),results)

                std_clusters = union_of_clusters_df['score'].std()
                density_centers = max(calculate_density(centeri, union_of_clusters_df, std_clusters),
                                      calculate_density(centerj, union_of_clusters_df, std_clusters))
                if density_centers != 0:
                    sum_density += calculate_density(final_u, union_of_clusters_df, std_clusters) / density_centers
        sum_density_combinations += sum_density

    return sum_density_combinations

def getCombination_Df(A,results):
    df_p1 = results[results['patient1'].isin(A)]
    df = df_p1[df_p1['patient2'].isin(A)]

    return df

def calculate_density(center, union, std):
    density = 0
    distance_df = union.query('patient1 == ' + str(center) + ' and patient2 == ' + str(center))
    for key, row in distance_df.iterrows():
        density += function_f(row['score'], std)

    return density

def function_f(distance, std):
    if std - distance < 0:
        return 0
    else:
        return 1

def getCenter(v, results):
    min_distance = math.inf
    for k, el1 in v.items():
        dist_sum = 0
        for k2, el2 in v.items():
            if el2 > el1:
                row = getPair(results, el1, el2)
                dist_sum += row['score'].values[0]
        if dist_sum < min_distance:
            min_distance = dist_sum
            center = el1
    return center

def getPair(results, el1, el2):
    row = results.query('patient1 == ' + str(el1) + ' and patient2 == ' + str(el2))
    if row.empty:
        row = results.query('patient1 == ' + str(el2) + ' and patient2 == ' + str(el1))
        if row.empty:
            row = pd.DataFrame(np.array([math.inf]), columns=['score'])

    return row

def getNearest(score, mean_points):
    closest = math.inf
    final_u = 0
    for k, v in mean_points.items():
        distance = abs(score - v)
        if distance < closest:
            final_u = k

    return final_u

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


