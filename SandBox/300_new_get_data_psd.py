from data_utilities import *
from neural_utils import *

patient_id = 'DM1008'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
neural_file_name='neural_data_trial1_'+raw_denoised
sentences_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-trials.tsv',
        sep='\t')
electrods_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_electrodes.tsv',
        sep='\t')
list_chn_df = pd.read_csv('./Datasets/' + patient_id + '/' +'CH_labels_'+raw_denoised+'.csv')
list_chn_df = list_chn_df.T[0].reset_index()
list_ECOG_chn = electrods_df.name[electrods_df.name.str.contains("ecog")].to_list()

mat = scipy.io.loadmat('./Datasets/' + patient_id + '/' + neural_file_name+'.mat')
neural_df = pd.DataFrame(np.transpose(mat['dataframe'])[:, :],
                             columns=list_chn_df[0].to_list())

frequency_bands = [[50, 60], [70,110],[130, 170]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': 1000,
    'margin_length': 500,
    'baseline_length': 1000,
    'numb_freq': 100,
    'L_cut_freq': 50,
    'H_cut_freq': 170,
    'avg_freq_bands': True,
    'smoothing': True,
    'smoothing_window_size': 10,
     }
#
saving_add = data_add +'/trials_'+raw_denoised+'/'
freqs = get_psd_features_direct(neural_df[list_ECOG_chn],neural_df.time.to_numpy(), trials_df, sentences_df, psd_config, saving_add)

psd_config['freqs'] = freqs
with open(saving_add + "neural_features_info.json", "w") as outfile:
    json.dump(psd_config, outfile)

