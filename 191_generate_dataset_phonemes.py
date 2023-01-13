from data_utilities import *

datasets_add = './Datasets/'
data_in, data_out, vocab_size = get_phonems_data(datasets_add,
                     phonemes_add= 'LM/our_phonemes_df.csv',
                     dict_add = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')