from data_utilities import *

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import matplotlib.pyplot as plt

patients_list = ['DM1005', 'DM1012', 'DM1008', 'DM1013','DM1007']
raw_denoised = 'raw'
datasets_add = './Datasets/'
save_result_path = datasets_add  + '/Results_com/'
phonemes = np.array(['UW', 'AH', 'IH', 'IY', 'AE', 'R', 'L', 'K', 'S', 'N', 'D', 'V',
       'B', 'DH'])
# phonemes = np.array(['B', 'ER'])
for i_phone in phonemes:
    for count, patient_id in enumerate(patients_list):
        data_add = datasets_add + patient_id + '/' + 'Results_'+raw_denoised+'/phonems_psd/vis_'+i_phone+'.txt'
        if Path(data_add).is_file():
            in_df = pd.read_csv(data_add,  sep='\t')
            in_df.to_csv(datasets_add + patient_id + '/' + 'Results_'+raw_denoised+'/phonems_psd/vis_'+i_phone+'.node', sep='\t', index=False)
            aa_df = pd.DataFrame()
            aa_df[['x', 'y','z']] = in_df[['x', 'y','z']]
            aa_df['color'] = count+1
            aa_df['size'] = in_df['color']

        if count == 0:
            tot_df= aa_df
        else:
            tot_df = pd.concat([tot_df,aa_df],axis=0)

    tot_df.to_csv(save_result_path+'all_'+i_phone+'.node', sep='\t', index=False)
