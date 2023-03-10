import pandas as pd
import numpy as np
datasets_add = './Datasets/'

phones_code_dic_df = pd.read_csv(datasets_add+'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')

clustering_1 = [[14, 0], [10, 19], [3, 21], [2, 32], [30, 26], [25, 12], [7, 23], [17, 38], [24, 15], [6, 37], [4, 11]]
clustering_2 = [[10, 19, 14, 0, 3, 21], [16, 5, 13], [30, 26, 25, 12, 7 ,23, 17, 28, 35], [20, 36, 27] ]  ### rows of table and keep vowels seperated
clustering_3 = [ [0, 14, 13], [30, 26], [25,12], [19,10, 5, 7, 23, 20, 8], [2, 32, 38, 17], [3, 21, 16 ,27]] ### columns of table and keep vowels seperated
clustering_4 = [[11,15,4,22,29,33,24,1,28,9,18,31,35,37,6],[0,2,3,5,7,8,10,12,13,14,16,17,19,20,21,23,25,26,27,30,32,34,36,38]]# vowls and cons
phones_code_dic_df['clustering_4'] = phones_code_dic_df['ids']
for ii in range(len(clustering_4)):
    for item_cluster in clustering_4[ii]:
        phones_code_dic_df['clustering_4'][phones_code_dic_df['ids'] == item_cluster]= np.min(clustering_4[ii])

uniques_cluster = phones_code_dic_df['clustering_4'].unique()

# for ii_reindx in range(len(uniques_cluster)):
#     phones_code_dic_df['clustering_4'][phones_code_dic_df['clustering_4'] ==uniques_cluster[ii_reindx] ] = ii_reindx
phones_code_dic_df.to_csv(datasets_add+'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
