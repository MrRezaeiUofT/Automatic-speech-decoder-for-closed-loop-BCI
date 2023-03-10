from data_utilities import *
import json

patient_id = 'DM1007'
raw_denoised = 'raw'
datasets_add = './Datasets/'

data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
dataset_info = {
    'dt': 1,
    'sampling_freq':1000
}


total_data, neural_df, phonemes_df, new_phonemes_df, trials_df, dt, zero_time, phones_code_dic, feature_list = get_data(patient_id,
                                                                                                          datasets_add,raw_denoised,
                                                                                                          dataset_info['dt'],
                                                                                                          dataset_info['sampling_freq'],
                                                                                                                        'neural_data_trial1_'+raw_denoised,
                                                                                                                        denoised_neural=False)
dataset_info['zero_time'] = zero_time
dataset_info['feature_id'] = feature_list
total_data.to_csv(data_add + 'prepro_phoneme_neural_total_v1_'+raw_denoised+'.csv')
with open(data_add + "dataset_info.json", "w") as outfile:
    json.dump(dataset_info, outfile)