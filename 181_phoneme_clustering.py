import pandas as pd
import numpy as np
datasets_add = './Datasets/'

phones_code_dic_df = pd.read_csv(datasets_add+'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')

clustering_1 = [[14, 0], [10, 19], [3, 21], [2, 32], [30, 26], [25, 12], [7, 23], [17, 38], [24, 15], [6, 37], [4, 11]]
clustering_2 = [[10, 19, 14, 0, 3, 21], [16, 5, 13], [30, 26, 25, 12, 7 ,23, 17, 28, 35], [20, 36, 27] ]  ### rows of table and keep vowels seperated
clustering_3 = [ [0, 14, 13], [30, 26], [25,12], [19,10, 5, 7, 23, 20, 8], [2, 32, 38, 17], [3, 21, 16 ,27]] ### columns of table and keep vowels seperated

phones_code_dic_df['clustering_1'] = phones_code_dic_df['ids']
for ii in range(len(clustering_1)):
    for item_cluster in clustering_1[ii]:
        phones_code_dic_df['clustering_1'][phones_code_dic_df['ids'] == item_cluster]= np.min(clustering_1[ii])

uniques_cluster = phones_code_dic_df['clustering_1'].unique()

# for ii_reindx in range(len(uniques_cluster)):
#     phones_code_dic_df['clustering_1'][phones_code_dic_df['clustering_1'] ==uniques_cluster[ii_reindx] ] = ii_reindx
phones_code_dic_df.to_csv(datasets_add+'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
