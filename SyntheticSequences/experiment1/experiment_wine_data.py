# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:08:04 2018

@author: kisha_000
"""

# Validation of Experiment with wine data

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, cut_tree
from fastcluster import linkage
from clustering_scores import cluster_indices,cluster_external_index
from statistics import mean, stdev
from tabulate import tabulate
import pandas as pd
import numpy as np

np.seterr(all='raise')

max_K = 9
M = 250
###############################################################################
#       HIERARCHICAL CLUSTERING
###############################################################################
# load data
df = pd.read_table('wine.txt',sep = ',', header = None)

# Preprocessing step - transformation into ranks
df_ranks = df.loc[:,1:].rank(axis = 0,method = 'first')

# generate the linkage matrix
Z = linkage(df_ranks, 'ward')

# dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
        Z,
        #truncate_mode = 'lastp',
        #p=6,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        )
plt.show()


###############################################################################
#       HOW MANY CLUSTERS?
###############################################################################
# bootstrap method - sampling without replacement
np.random.seed(123)

#dictionary to store all computed indexes for each number of clusters K=2,...9
dicio_statistics = {k:{} for k in range(2,10)}
for k in range(2,10):
    dicio_statistics[k]['rand'] = []
    dicio_statistics[k]['adjusted'] = []
    dicio_statistics[k]['FM'] = []
    dicio_statistics[k]['jaccard'] = []

# number of bootstrap samples
M = 250

#for each bootstrap sample
for i in range(M):
    # sampling rows of the original data
    idx = np.random.choice(len(df_ranks), int((3/4)*len(df_ranks)), replace = False)
    # Hierarchical Clustering of the bootstrap sample
    Z_bootstrap = linkage(df_ranks.loc[idx,:],'ward')

    #for each number of clusters k=2,...,9
    for k in range(2,10):
        c_assignments_original = cut_tree(Z,k)
        c_assignments_bootstrap = cut_tree(Z_bootstrap,k)
        #list of clusters for the clustering result with the original data
        partition_original = cluster_indices(c_assignments_original,df.index.tolist())
        #list of clusters for the clustering result with the bootstrap sample
        partition_bootstrap = cluster_indices(c_assignments_bootstrap,idx)

        #compute 4 different cluster external indexes between the partitions
        computed_indexes = cluster_external_index(partition_original,partition_bootstrap)
        dicio_statistics[k]['rand'].append(computed_indexes[0])
        dicio_statistics[k]['adjusted'].append(computed_indexes[1])
        dicio_statistics[k]['FM'].append(computed_indexes[2])
        dicio_statistics[k]['jaccard'].append(computed_indexes[3])

#obtain the average cluster external indexes for each number of clusters and show the results in a table
rand_avg = []
adjusted_avg = []
FM_avg = []
jaccard_avg = []
table = []
#obtain the standard deviation of adjusted rand index for each number of clusters
adjusted_std = []

for k in range(2,10):
    rand_avg.append(mean(dicio_statistics[k]['rand']))
    adjusted_avg.append(mean(dicio_statistics[k]['adjusted']))
    FM_avg.append(mean(dicio_statistics[k]['FM']))
    jaccard_avg.append(mean(dicio_statistics[k]['jaccard']))
    table.append([k,mean(dicio_statistics[k]['rand']),mean(dicio_statistics[k]['adjusted']),mean(dicio_statistics[k]['FM']),mean(dicio_statistics[k]['jaccard'])])
    adjusted_std.append(stdev(dicio_statistics[k]['adjusted']))
    
print(tabulate(table, headers=['Number of clusters', 'Rand','Adjusted Rand','Fowlkes and Mallows','Jaccard']))

#bar chart
width = 1/1.5
plt.figure(1)
plt.title('Standard Deviation of Adjusted Rand Index versus number of clusters')
plt.bar(range(2,10), adjusted_std, width, color="blue")
plt.show()
