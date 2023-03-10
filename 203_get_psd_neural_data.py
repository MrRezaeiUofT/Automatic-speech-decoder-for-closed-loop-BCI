from data_utilities import *
from neural_utils import *

patient_id = 'DM1007'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)
total_data = pd.read_csv(data_add + 'prepro_phoneme_neural_total_v1_'+raw_denoised+'.csv')

# '''  remove rows with 'sp' phonemes '''
# total_data =total_data.drop(total_data[total_data.phoneme == 'sp'].index)
# time frequency neural features
list_ECOG_chn = total_data.columns[total_data.columns.str.contains("ecog")].to_list()
frequency_bands = [[30, 50], [70,110],[130, 170]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': 1000//dataset_info['dt'],
    'margin_length': 500//dataset_info['dt'],

    'freq_stp': 1,
    'L_cut_freq': 30,
    'H_cut_freq': 170,
    'avg_freq_bands': True,
    'smoothing': False,
    'smoothing_window_size': 10,
     }

saving_add = data_add +'/trials_'+raw_denoised+'/'
freqs = get_psd_features(total_data, psd_config, patient_id, saving_add)

psd_config['freqs'] = freqs
with open(saving_add + "neural_features_info.json", "w") as outfile:
    json.dump(dataset_info, outfile)


