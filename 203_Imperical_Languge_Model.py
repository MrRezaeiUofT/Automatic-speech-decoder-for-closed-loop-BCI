from data_utilities import *
import matplotlib.pyplot as plt
import json
import pickle
from model_utils import get_language_components
patient_id = 'DM1005'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)

N_gram = 2
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')

''' Only consider the onset data'''
non_phoneme_onset = total_data[total_data.phoneme_onset == 0].index.to_numpy()
total_data = total_data.drop(non_phoneme_onset, axis=0)
# sp_index = total_data[total_data.phoneme == 'sp'].index.to_numpy()
# total_data = total_data.drop(sp_index, axis=0)
# nan_index = total_data[total_data.phoneme == 'NAN'].index.to_numpy()
# total_data = total_data.drop(nan_index, axis=0)
''' visualization of the phoneme histograms'''
plt.figure(figsize=(16,8))
total_data.phoneme.hist(log=True, bins=total_data.phoneme_id.nunique())
plt.savefig(save_result_path + 'Phonemes-dist.png')
plt.savefig(save_result_path + 'Phonemes-dist.svg', format='svg')
count_phonemes, bb= np.histogram(total_data.phoneme_id.to_numpy(), total_data.phoneme_id.nunique())

''' get language model'''
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic = get_language_components(total_data, N_gram, save_result_path, datasets_add)

with  open(data_add + 'language_model_data.pkl', "wb") as open_file:
    pickle.dump([pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes ], open_file)
