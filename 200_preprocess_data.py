from data_utilities import *


patient_id = 'DM1008'
datasets_add = './Datasets/'
feature_id = [0, 128]
dt = 10

total_data, neural_df, phonemes_df, new_phonemes_df, trials_df, dt, zero_time, phones_code_dic = get_data(patient_id, datasets_add, feature_id, dt)
total_data.to_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')