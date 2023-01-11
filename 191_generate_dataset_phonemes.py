from data_utilities import *
import json


datasets_add = './Datasets/'
phones_df_all = pd.read_csv(datasets_add+'LM/phonemes_df.csv')

phones_code_dic = dict(zip(phones_df_all.phoneme.unique(), np.arange(phones_df_all.phoneme.nunique())))
if 'NAN' in phones_code_dic:
    pass
else:
    phones_code_dic.update({'NAN': len(phones_df_all.phoneme.unique())})
# if phones_df['phoneme'].isin(['nan']):
#     phones_df[phones_df['phoneme'] == 'nan']['phoneme']= 'NAN'
phones_df_all['phoneme_id'] = 0
phones_df_all['phoneme_id'] = phones_df_all['phoneme'].apply(lambda x: phones_code_dic[x])
phones_df_all['ph_temp'] = 1
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