from data_utilities import *
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
patient_id = 'DM1008'
datasets_add = './Datasets/'
feature_id = [0, 128]
dt = 2
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')
# time frequency neural features
list_ECOG_chn = total_data.columns[total_data.columns.str.contains("feature")].to_list()
frequency_bands = [[2, 8], [8, 12], [12, 24]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': 1000//dt,
    'freq_stp': 1,
    'L_cut_freq': 2,
    'H_cut_freq': 200,
    'avg_freq_bands': False,
    'smoothing': True,
    'smoothing_window_size': 50,
     }

neural_psd_band, freqs = get_psd_features(total_data, psd_config, patient_id)


corr_matrix = np.zeros((neural_psd_band.shape[0], neural_psd_band.shape[1]))


# for ii in range(neural_psd_band.shape[0]):
#     for jj in range(neural_psd_band.shape[1]):
#         corr_matrix[ii, jj] = pearsonr(neural_psd_band[ii, jj, :].squeeze(), total_data.phoneme_id.to_numpy().squeeze())[0]
#
# fig, ax = plt.subplots()
# im = ax.imshow(corr_matrix.T)
# ax.set_xlabel('channels')
# ax.set_ylabel('frequency')