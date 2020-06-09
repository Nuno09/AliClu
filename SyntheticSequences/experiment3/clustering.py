# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:35:56 2018

@author: kisha_000
"""

from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram,cophenet
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt


#Function that receives a similarity matrix in a list form and converts it to a 
#matrix. Negate all entries and add an offset to make all values positive.
def convert_to_distance_matrix(similarity_matrix):
    
    distance_matrix = -similarity_matrix
    distance_matrix = distance_matrix + abs(distance_matrix[distance_matrix.idxmin()])
    
    return distance_matrix

#Function that performs agglomerative clustering. It receives as input a distance matrix
#and a distance metric used to measure distance between clusters.
# It outputs a dendrogram and the cophenetic correlation coefficient.
def hierarchical_clustering(distance_matrix,method,gap):

    #agglomerative clustering
    Z = linkage(distance_matrix, method)
    
    #dendrogram plot
#    plt.figure(figsize=(25, 10))
#    plt.title('Hierarchical Clustering Dendrogram - gap: %.2f ' %gap,fontsize=45)
#    plt.xlabel('object index',labelpad=20,fontsize=40)    
#    plt.ylabel('distance',labelpad=10,fontsize=40)    
#    plt.xticks(size = 40)
#    plt.yticks(size = 40)
#    dendrogram(
#            
#            Z,
#            #truncate_mode = 'lastp',
#            #p=6,
#            leaf_rotation=90.,  # rotates the x axis labels
#            leaf_font_size=15.,  # font size for the x axis labels
#            )
#    plt.show()
#    
#    # Cophenetic Correlation Coefficient
#    c, coph_dists = cophenet(Z, distance_matrix)
#    print('Cophenetic Correlation Coefficient:', c)
    
    return Z