from data_utilities import *
import seaborn as sns
patient_id = 'DM1008'
datasets_add = './Datasets/'
feature_id = [0, 128]
dt = 19
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')
# time frequency neural features
list_ECOG_chn = total_data.columns[total_data.columns.str.contains("feature")].to_list()[1:3]
frequency_bands = [[2, 8], [8, 12], [12, 24]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': 1000//dt,
    'freq_stp': 1,
    'L_cut_freq': 2,
    'H_cut_freq': 25,
    'avg_freq_bands': True,
    'smoothing': True,
    'smoothing_window_size': 50,
     }

neural_psd_band, freqs = get_psd_features(total_data, psd_config, patient_id)