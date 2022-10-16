
import scipy.io
import numpy as np
import pandas as pd
import scipy
from mne import create_info, EpochsArray
from scipy import signal
from mne.time_frequency import (tfr_array_morlet)
from sklearn.preprocessing import MinMaxScaler


def get_data(patient_id, datasets_add, feature_id, dt, sampling_freq):
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
    mat = scipy.io.loadmat('./Datasets/' + patient_id + '/' + 'neural_data_trial3.mat')
    neural_df = pd.DataFrame(np.transpose(mat['dataframe'])[:, feature_id[0]:feature_id[1]],
                             columns=['feature-' + str(ii) for ii in range(feature_id[0], feature_id[1])])
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


    ''' identify the dt for phonemes'''
    # shortest_ph = np.ceil(phones_df.duration.min() * sampling_freq).astype('int')
    # shortest_delay_between_ph = np.ceil(
    #     (phones_df.onset.to_numpy()[1:] - phones_df.onset.to_numpy()[:-1]).min() * sampling_freq).astype('int')
    # dt = np.min([shortest_ph, shortest_delay_between_ph]) -1

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
    neural_df = neural_df.apply(lambda x: x.fillna(x.mean()), axis=0)

    ''' add baseline identifier to neural df'''
    neural_df['baseline_flag'] = 0
    for itr in trials_df.trial_id.unique():
        if itr == 1:
            pass
        else:
            index_start_baseline = trials_df[trials_df.trial_id == itr-1].itg_onset.to_numpy()[0].astype('int')
            length_baseline = trials_df[trials_df.trial_id == itr-1].itg_duration.to_numpy()[0]
            index_baselines = np.arange(index_start_baseline, index_start_baseline+length_baseline)
            neural_df['baseline_flag'][index_baselines] = itr

    ''' re-assign the phoneme ids'''
    phones_code_dic = dict(zip(phones_df.phoneme.unique(), np.arange(phones_df.phoneme.nunique())))
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
    total_data[['id_onehot_' + str(iid) for iid in range(total_data['phoneme_id'].nunique())]] = pd.get_dummies(
        total_data['phoneme_id']).to_numpy()
    total_data['phoneme_onset'] = new_phones_df.phoneme_onset
    total_data['trial_id'] = new_phones_df.trial_id
    return total_data, neural_df, phones_df, new_phones_df, trials_df, dt, zero_time, phones_code_dic


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))

