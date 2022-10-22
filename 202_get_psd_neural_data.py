from data_utilities import *
from neural_utils import *

patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')

# '''  remove rows with 'sp' phonemes '''
# total_data =total_data.drop(total_data[total_data.phoneme == 'sp'].index)
# time frequency neural features
list_ECOG_chn = total_data.columns[total_data.columns.str.contains("feature")].to_list()
frequency_bands = [[2, 30],[30,50], [70,200]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': 1000//dataset_info['dt'],
    'freq_stp': 1,
    'L_cut_freq': 2,
    'H_cut_freq': 200,
    'avg_freq_bands': True,
    'smoothing': True,
    'smoothing_window_size': 10,
     }

saving_add = data_add +'trials/'
freqs = get_psd_features(total_data, psd_config, patient_id, saving_add)

psd_config['freqs'] = freqs
with open(saving_add + "neural_features_info.json", "w") as outfile:
    json.dump(dataset_info, outfile)

