
import scipy.io
import numpy as np
import pandas as pd
# from patsy import dmatrix, build_design_matrices
# from neural_utils import  calDesignMatrix_V2, calDesignMatrix_V4
import pickle
import torch
from torch.utils.data import Dataset

def get_data(patient_id, datasets_add,raw_denoised, dt, sampling_freq, file_name,denoised_neural=False):
    '''
    get the neural and phoneme information
    :param patient_id: patient ID
    :param datasets_add: address of the dataset folder
    :param feature_id: selection interval for neural data channels
    :return:
         neural_df: neural features dataframe
         phones_df: phonemes data frame
         trials_df: trials information
         dt: resolution of quantized time
         zero_time: intial time of the all aligned dataframes here
    '''
    mat = scipy.io.loadmat('./Datasets/' + patient_id + '/' + file_name+'.mat')
    list_chn_df = pd.read_csv('./Datasets/' + patient_id + '/' +'CH_labels_'+raw_denoised+'.csv')
    list_chn_df = list_chn_df.T[0].reset_index()
    neural_df = pd.DataFrame(np.transpose(mat['dataframe'])[:, :],
                             columns=list_chn_df[0].to_list())
    '''group by over repeated indexes in denoised signal'''
    # if denoised_neural:
    #     neural_df = neural_df.groupby(by=["time"], as_index=False).mean().reset_index()
    # else:
    #     pass

    neural_df = neural_df.rename(columns={'feature-0': 'time'})
    phones_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-phonemes.tsv',
        sep='\t')
    phones_df.phoneme =  phones_df.phoneme.replace(' ', 'NAN')
    sentences_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
    words_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-words.tsv',
        sep='\t')
    trials_df = pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-trials.tsv', sep='\t')
    ''' upper case all phonems'''
    phones_df['phoneme'] = phones_df['phoneme'].str.upper()
    phones_df['phoneme'] = phones_df['phoneme'].fillna('NAN')
    phones_df.phoneme = apply_stress_remove([phones_df.phoneme.to_list()])[0]
    phones_df.phoneme = phones_df.phoneme.replace('NA', 'NAN')
    ''' drop nans in trial annotations'''
    trials_df = trials_df.dropna()

    ''' identify the zero time and align the dataset with it'''
    zero_time = trials_df.onset.to_numpy()[0]
    trials_df.onset = trials_df.onset - zero_time
    trials_df.itg_onset = trials_df.itg_onset - zero_time

    phones_df.onset = phones_df.onset - zero_time
    sentences_df.onset = sentences_df.onset - zero_time
    words_df.onset = words_df.onset - zero_time
    neural_df.time = neural_df.time - zero_time
    neural_df = neural_df.drop(neural_df[neural_df.time < 0].index)



    ''' quantize time '''
    trials_df.itg_onset = np.floor(trials_df.itg_onset * sampling_freq / dt).astype('int')
    trials_df.itg_duration = np.floor(trials_df.itg_duration * sampling_freq / dt).astype('int')

    phones_df.onset = np.floor(phones_df.onset * sampling_freq / dt).astype('int')
    phones_df.duration = np.floor(phones_df.duration * sampling_freq / dt).astype('int')

    words_df.onset = np.floor(words_df.onset * sampling_freq / dt).astype('int')
    words_df.duration = np.floor(words_df.duration * sampling_freq / dt).astype('int')

    sentences_df.onset = np.floor(sentences_df.onset * sampling_freq / dt).astype('int')
    sentences_df.duration = np.floor(sentences_df.duration * sampling_freq / dt).astype('int')

    trials_df.onset = np.floor(trials_df.onset * sampling_freq / dt).astype('int')
    trials_df.duration = np.floor(trials_df.duration * sampling_freq / dt).astype('int')

    ''' zero step neural data preprocess-> quantize time (by averaging over the neural features in a windo), impute nanas with mean'''
    neural_df.time = np.floor(neural_df.time * sampling_freq / dt).astype('int')
    neural_df = neural_df.groupby(by=["time"], dropna=False).mean()
    neural_df = neural_df.apply(lambda x: x.fillna(0), axis=0)

    ''' synchronize'''
    if neural_df.index[0] !=0:
        temp_df = pd.DataFrame(np.zeros((neural_df.index[0], len(neural_df.columns))), columns=neural_df.columns)
        neural_df=pd.concat([temp_df,neural_df])
    ''' add baseline identifier to neural df'''
    neural_df['baseline_flag'] = 0
    for itr in trials_df.trial_id.unique():

        if itr == 1:
            pass
        else:
            index_start_baseline = trials_df[trials_df.trial_id == itr-1].itg_onset.to_numpy()[0].astype('int')
            length_baseline = trials_df[trials_df.trial_id == itr-1].itg_duration.to_numpy()[0]
            index_baselines = np.arange(index_start_baseline, index_start_baseline+length_baseline)
            if raw_denoised == 'denoised':
                index_baselines = np.intersect1d(index_baselines,neural_df.index.to_numpy())
            neural_df['baseline_flag'][index_baselines] = itr

    ''' re-assign the phoneme ids'''

    phonemes_dict_df = pd.read_csv(datasets_add + 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
    phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df['ids'].to_list()))
    if 'NAN' in phones_code_dic:
        pass
    else:
        phones_code_dic.update({'NAN':len(phones_df.phoneme.unique()) })


    phones_df['phoneme_id'] = 0
    phones_df['phoneme_id'] = phones_df['phoneme'].apply(lambda x: phones_code_dic[x])
    phones_df['phoneme_onset'] = 0
    phoneme_onset_cul_indx = phones_df.columns.get_loc('phoneme_onset')

    ''' resample the phonemes'''
    temp_frame = phones_df.iloc[0].values
    temp_frame[1] = phones_df.onset[0]
    temp_df = temp_frame.repeat(phones_df.onset[0]).reshape([-1, phones_df.onset[0]]).transpose()
    temp_df[0, phoneme_onset_cul_indx] = 1
    new_phones_df = pd.DataFrame(temp_df, columns=phones_df.columns)

    temp_df = phones_df.iloc[0].values.repeat(phones_df.duration[0]).reshape([-1, phones_df.duration[0]]).transpose()
    temp_df[0, phoneme_onset_cul_indx] = 1
    new_phones_df = new_phones_df.append(pd.DataFrame(temp_df, columns=phones_df.columns))
    for ii in range(1, phones_df.shape[0]):
        # print(ii)
        if phones_df.duration[ii - 1] + phones_df.onset[ii - 1] < phones_df.onset[ii]:
            numb_rep = phones_df.onset[ii] - phones_df.duration[ii - 1] - phones_df.onset[ii - 1]
            temp_frame[1] = numb_rep
            temp_frame[0] = phones_df.onset[ii]

            temp_df = phones_df.iloc[ii].values.repeat(numb_rep).reshape(
                    [-1, numb_rep]).transpose()
            temp_df[:, phones_df.columns.get_loc('duration')] = 0
            temp_df[:, phones_df.columns.get_loc('phoneme')] = 'NAN'
            temp_df[:, phones_df.columns.get_loc('phoneme_id')] = phones_code_dic['NAN']
            new_phones_df = new_phones_df.append(pd.DataFrame(temp_df
                , columns=phones_df.columns))
        else:
            pass
        if np.size(phones_df.iloc[ii].values.repeat(phones_df.duration[ii])) != 0 :
            temp_df = phones_df.iloc[ii].values.repeat(phones_df.duration[ii]).reshape(
                    [-1, phones_df.duration[ii]]).transpose()
            temp_df[0, phoneme_onset_cul_indx] = 1
            new_phones_df = new_phones_df.append(pd.DataFrame(temp_df
                , columns=phones_df.columns))

    new_phones_df = new_phones_df.reset_index()

    ''' align neural and phonemes dataframes'''
    neural_df = neural_df.drop(neural_df[neural_df.index > new_phones_df.index[-1]].index)

    ''' build a unified dataset'''
    total_data = neural_df.copy()
    total_data['phoneme_duration'] = new_phones_df.duration
    total_data['phoneme'] = new_phones_df.phoneme
    total_data['phoneme_id'] = new_phones_df.phoneme_id
    total_data[['id_onehot_' + str(iid) for iid in range(len(phones_code_dic))]] = get_one_hot_encodings( total_data,
                                                                                                          len(phones_code_dic))
    total_data['phoneme_onset'] = new_phones_df.phoneme_onset
    total_data['trial_id'] = new_phones_df.trial_id
    return total_data, neural_df, phones_df, new_phones_df, trials_df, dt, zero_time, phones_code_dic, list_chn_df[0].to_list()

def get_one_hot_encodings(dataframe, max_length_code):
    one_hot_codes = np.zeros((dataframe.shape[0],max_length_code))
    phonemes_ids = dataframe['phoneme_id'].to_numpy().astype('int')
    for ii in (np.unique(phonemes_ids)):
        one_hot_codes[np.where(phonemes_ids == ii)[0], ii] = 1

    return one_hot_codes

def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))

# def bspline_window(config_LSSM_MPP):
#     x = np.linspace(0, 1, config_LSSM_MPP['decode_length'])
#     y = dmatrix("bs(x, df=20, degree=10, include_intercept=True) - 1", {"x": x})
#     return y

def get_phonems_data(datasets_add,
                    clustering_id,
                    clustering =True,
                    phonemes_add= 'LM/our_phonemes_df.csv',
                    dict_add = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'):
    '''
    get convert the phonemes sequence to a matrix representations and prepare it for language model training
    :param datasets_add:
    :param phonemes_add:
    :param dict_add:
    :return:
    '''
    phones_df_all = pd.read_csv(datasets_add + phonemes_add)
    phonemes_dict_df = pd.read_csv(datasets_add + dict_add)
    if clustering:
        clr = 'clustering_'+str(clustering_id)
        phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df[clr].to_list()))
        reindexing_dic = dict(zip(phonemes_dict_df['ids'].to_list(), phonemes_dict_df[clr].to_list()))

    else:
        phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df['ids'].to_list()))

    phones_df_all['ph_temp'] = 1  # just for finding maximum length for sentences
    max_sentence_L = phones_df_all.groupby(['trial_id']).sum().ph_temp.max()
    dataset = np.zeros((phones_df_all.trial_id.max(), max_sentence_L+1)).astype('int')
    for ii in range(1,phones_df_all.trial_id.max()):
        current_sent = phones_df_all[phones_df_all.trial_id == ii].phoneme_id.to_numpy().astype('int')
        if len(current_sent) !=0:
            if clustering:
                current_sent = vec_translate(current_sent, reindexing_dic)
        if len(current_sent)< max_sentence_L:
            dataset[ii, 1:] = np.concatenate(
                [current_sent, (phones_code_dic['NAN'] * np.ones((max_sentence_L - len(current_sent),)))],
                axis=0).astype('int')
        else:
            dataset[ii, 1:] = current_sent.astype('int')
    dataset[:,0]=phones_code_dic['NAN']
    data_in = dataset
    data_out = np.concatenate([dataset, (phones_code_dic['NAN'] * np.ones((dataset.shape[0], 1)))], axis=1)[:,
               1:].astype('int')
    vocab_size = len(phones_code_dic)
    return [data_in, data_out,vocab_size]



class prepare_phoneme_dataset(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, data_in,data_out, vocab_size):
        self.data_in = torch.tensor(data_in, dtype=torch.int64)
        self.data_out = torch.tensor(data_out, dtype=torch.int64)
        self.sentence_length = data_out.shape[1]
        self.data_length = data_out.shape[0]
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.sentence_length
    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        return [self.data_in[index], self.data_out[index]]
#
# def get_trial_data(data_add,trial, h_k,f_k, phones_code_dic, raw_denoised,tensor_enable,d_sample=10, del_sp_nan=True):
#     '''
#     get a batch of data and [re[are it for the model
#     :param data_add: data address
#     :param trial: trial id
#     :param h_k: length of history
#     :return:
#     '''
#     file_name = data_add + 'trials_'+raw_denoised+'/trial_' + str(trial) + '.pkl'
#     with open(file_name, "rb") as open_file:
#         data_list_trial = pickle.load(open_file)
#     data_list_trial[1] =data_list_trial[1].reset_index()
#     X_tr = np.swapaxes(data_list_trial[0], 2, 0)
#     # print(X_tr.shape)
#     if tensor_enable:
#         X_tr = X_tr.reshape([X_tr.shape[0], -1])
#
#         XDesign = calDesignMatrix_V2(X_tr, h_k + 1)
#     else:
#
#         XDesign = calDesignMatrix_V4(X_tr, h_k + 1, f_k,d_sample)
#
#     # print(XDesign.shape)
#
#     y_tr = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].to_numpy()
#     non_phoneme_onset = data_list_trial[1][data_list_trial[1].phoneme_onset == 0].index.to_numpy()
#     if del_sp_nan:
#         ''' delete 'sp' and 'NaN'  and non-onset_phonemes from dataset'''
#         sp_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['SP'])[0]
#         nan_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['NAN'])[0]
#         delet_phonemes_indx = np.unique(np.concatenate([nan_index, sp_index, non_phoneme_onset],axis=0))
#         XDesign = np.delete(XDesign, delet_phonemes_indx, 0)
#         y_tr = np.delete(y_tr, delet_phonemes_indx, 0)
#         if tensor_enable:
#             return torch.tensor(XDesign, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
#         else:
#             return XDesign, y_tr
#     else:
#         ''' delete non-onset_phonemes from dataset'''
#
#         non_phoneme_onset = data_list_trial[1][data_list_trial[1].phoneme_onset == 0].index.to_numpy()
#         XDesign = np.delete(XDesign, non_phoneme_onset, 0)
#         y_tr = np.delete(y_tr, non_phoneme_onset, 0)
#         if tensor_enable:
#             return torch.tensor(XDesign, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
#         else:
#             return XDesign, y_tr
#





def apply_stress_remove(input_list):
    output_list = []
    if len(input_list) !=0:
        for item in input_list[0]:
            # print(item)
            output_list.append(item[:2])
    else:
        pass

    return [output_list]


def vec_translate(a, my_dict):
   return np.vectorize(my_dict.__getitem__)(a)

def sort_index(idexes,guid):
    id_orders = np.zeros_like(idexes)+1000
    for ii in range(len(idexes)):
        temp =np.where(guid == idexes[ii])[0]
        if len(temp) == 0:
            pass
        else:
            id_orders[ii] = temp
    sorting_id = np.argsort(np.squeeze(id_orders))
    return idexes[sorting_id], sorting_id