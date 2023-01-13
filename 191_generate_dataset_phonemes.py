from data_utilities import *
import json


datasets_add = './Datasets/'
phones_df_all = pd.read_csv(datasets_add+'LM/our_phonemes_df.csv')
phonemes_dict_df = pd.read_csv(datasets_add + 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(),phonemes_dict_df['ids'].to_list()))

phones_df_all['ph_temp'] = 1# just for finding maximum length for sentences
max_sentence_L= phones_df_all.groupby(['trial_id']).sum().ph_temp.max()
dataset = np.zeros((phones_df_all.trial_id.max(), max_sentence_L)).astype('int')
for ii in range(phones_df_all.trial_id.max()):
    current_sent = phones_df_all[phones_df_all.trial_id == ii].phoneme_id.to_numpy().astype('int')
    if max_sentence_L != len(current_sent):
        dataset[ii,:] = np.concatenate([current_sent, (phones_code_dic['NAN']*np.ones((max_sentence_L-len(current_sent),)) )], axis=0).astype('int')
    else:
        dataset[ii, :] = current_sent.astype('int')

data_in = dataset
data_out = np.concatenate([dataset,(phones_code_dic['NAN']*np.ones((dataset.shape[0],1)))],axis=1)[:,1:].astype('int')