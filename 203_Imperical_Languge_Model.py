from data_utilities import *
import matplotlib.pyplot as plt
import json
import pickle
from model_utils import get_language_components
patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)

N_gram = 2
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')
''' visualization of the phoneme histograms'''
total_data.phoneme.hist(log=True)

pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic = get_language_components(total_data, N_gram)

with  open(data_add + 'language_model_data.pkl', "wb") as open_file:
    pickle.dump([pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic ], open_file)
