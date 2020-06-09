# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:26:25 2018

@author: kisha_000
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle

directory = 'experiment3_synthetic_results/'
file_name_dict = 'experiment3_synthetic_setup2dataset1_all_gap_Tp_2_ward__statistics.pickle'
pickle_in = open(directory+file_name_dict,"rb")
cluster_vs_gap_statistics = pickle.load(pickle_in)

n_experiment=25
gap_values = np.linspace(-1,1,21)
n_sequences_list = [5,15,25,50]
df_all_percentages = pd.DataFrame(0,index=gap_values,columns = n_sequences_list,dtype='float')

for n_sequences in cluster_vs_gap_statistics:
    for gap in cluster_vs_gap_statistics[n_sequences]:
        if(cluster_vs_gap_statistics[n_sequences][gap]):
            df_all_percentages.loc[gap,n_sequences] = cluster_vs_gap_statistics[n_sequences][gap][4][0]
            
df_all_percentages  = df_all_percentages/n_experiment
df_all_percentages = df_all_percentages.replace(0,-0.01)


df_all_percentages.index = np.around(df_all_percentages.index,2)
df_all_percentages.plot.bar(figsize=(15,8),width=0.6).legend(bbox_to_anchor=(0.25, 1),fontsize=20,title='Number of sequences per cluster')
plt.gca().yaxis.grid(True)
plt.ylim([-0.05,1.1])
plt.title('Percentage of correct decisions on the number of clusters ',fontsize=25)
plt.xlabel('Gap penalty values',labelpad=20,fontsize=20)    
plt.ylabel('Percentage',labelpad=10,fontsize=20)    
plt.xticks(size = 20)
plt.yticks(size = 20)