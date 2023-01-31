from data_utilities import *
import json

patient_ids = ['DM1005', 'DM1007', 'DM1008', 'DM1012', 'DM1013']
datasets_add = './Datasets/'
phonemes_dict_df = pd.read_csv(datasets_add + 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
### apply two words

#####
phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(),phonemes_dict_df['ids'].to_list()))
pp =0
for patient_id in patient_ids:

    phones_df_p = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-phonemes.tsv',
        sep='\t')
    ''' Replace missing values with NAN'''
    phones_df_p['phoneme'] = phones_df_p['phoneme'].fillna('NAN')
    phones_df_p.phoneme = apply_stress_remove([phones_df_p.phoneme.to_list()])[0]
    phones_df_p = phones_df_p[["phoneme", "trial_id"]]
    phones_df_p.phoneme = phones_df_p.phoneme.replace(' ', 'NAN')
    phones_df_p.phoneme = phones_df_p.phoneme.replace('NA', 'NAN')
    phones_df_p['phoneme'] = phones_df_p['phoneme'].str.upper()

    if pp == 0:
        phones_df = phones_df_p.copy()
    else:
        phones_df_p.trial_id +=  phones_df.trial_id.max()
        phones_df =pd.concat([phones_df, phones_df_p])

    pp +=1

phones_df['phoneme_id'] = 0
phones_df['phoneme_id'] = phones_df['phoneme'].apply(lambda x: phones_code_dic[x])


phones_df.to_csv(datasets_add+'LM/our_phonemes_df.csv')