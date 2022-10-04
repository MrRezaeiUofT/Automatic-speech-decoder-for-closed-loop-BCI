
import scipy.io
import numpy as np
import pandas as pd
import scipy
# from mne import create_info, EpochsArray
from scipy import signal
# from mne.time_frequency import (tfr_array_morlet)
from sklearn.preprocessing import MinMaxScaler


def get_data(patient_id, datasets_add, feature_id, dt):
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
    phones_df.onset = phones_df.onset - zero_time
    sentences_df.onset = sentences_df.onset - zero_time
    words_df.onset = words_df.onset - zero_time
    neural_df.time = neural_df.time - zero_time
    neural_df = neural_df.drop(neural_df[neural_df.time < 0].index)

    ''' identify the dt for phonemes'''
    # shortest_ph = np.ceil(phones_df.duration.min() * 1000).astype('int')
    # shortest_delay_between_ph = np.ceil(
    #     (phones_df.onset.to_numpy()[1:] - phones_df.onset.to_numpy()[:-1]).min() * 1000).astype('int')
    # dt = np.min([shortest_ph, shortest_delay_between_ph]) -1

    ''' quantize time '''
    phones_df.onset = np.floor(phones_df.onset * 1000 / dt).astype('int')
    phones_df.duration = np.floor(phones_df.duration * 1000 / dt).astype('int')

    words_df.onset = np.floor(words_df.onset * 1000 / dt).astype('int')
    words_df.duration = np.floor(words_df.duration * 1000 / dt).astype('int')

    sentences_df.onset = np.floor(sentences_df.onset * 1000 / dt).astype('int')
    sentences_df.duration = np.floor(sentences_df.duration * 1000 / dt).astype('int')

    trials_df.onset = np.floor(trials_df.onset * 1000 / dt).astype('int')
    trials_df.duration = np.floor(trials_df.duration * 1000 / dt).astype('int')
    ''' zero step neural data preprocess-> quantize time (by averaging over the neural features in a windo), impute nanas with mean'''
    neural_df.time = np.floor(neural_df.time * 1000 / dt).astype('int')
    neural_df = neural_df.groupby(by=["time"], dropna=False).mean()
    neural_df = neural_df.apply(lambda x: x.fillna(x.mean()), axis=0)
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
    total_data['phoneme_onset'] = new_phones_df.phoneme_onset
    total_data['trial_id'] = new_phones_df.trial_id
    return total_data, neural_df, phones_df, new_phones_df, trials_df, dt, zero_time, phones_code_dic


# def get_psd_features(total_data_df, psd_config, patient_id):
#
#
#     if patient_id == 'DM1008':
#        pass
#     else:
#         pass
#
#     neural_data = np.expand_dims(np.transpose(total_data_df[psd_config['chnls']].to_numpy()), axis=0)
#     neural_psd = psd_extractor(neural_data, psd_config, type='power')
#     neural_psd_band, freqs = feature_mapping(neural_psd, psd_config)
#     return neural_psd_band, freqs
#
#
# def psd_extractor(neural_data, psd_config, type='power'):
#     freqs = np.arange(psd_config['L_cut_freq'], psd_config['H_cut_freq'], psd_config['freq_stp'])
#     n_cycles = freqs
#     info = create_info(ch_names=psd_config['chnls'], sfreq=psd_config['sampling_freq'], ch_types='misc', verbose=0)
#     epochs = EpochsArray(data=neural_data, info=info, verbose=0)
#     if type == 'power':
#         power = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
#                                  output='avg_power', verbose=0)
#
#     if psd_config['smoothing']:
#         window = signal.windows.gaussian(psd_config['smoothing_window_size'], psd_config['smoothing_window_size'] / 8)
#         # window = signal.windows.hamming(PSD_config['smoothing_window_size'])
#         for ii in range(power.shape[0]):
#             for jj in range(power.shape[1]):
#                 power[ii, jj, :] = np.convolve(power[ii, jj, :], window, 'same')
#
#     return power
#
#
# def feature_mapping(neural_features, psd_config):
#
#     frequency_bands = psd_config['FreqBands']
#     # list of frequencies for extracting power
#     freqs = np.arange(psd_config['L_cut_freq'], psd_config['H_cut_freq'], psd_config['freq_stp'])
#     temp = np.zeros((neural_features.shape[0], len(frequency_bands), neural_features.shape[2]))
#
#     if psd_config["avg_freq_bands"]:
#         for ii in range(len(frequency_bands)):
#             temp[:, ii, :] = np.median(
#                 neural_features[:, np.where((freqs >= frequency_bands[ii][0]) &
#                                             ((freqs < frequency_bands[ii][1])))[0], :],  axis=1)
#         new_neural_features = temp
#         freqs = np.arange(len(frequency_bands))
#     else:
#         new_neural_features= neural_features
#     return new_neural_features, freqs


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))