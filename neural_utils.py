

import numpy as np

from mne import create_info, EpochsArray
from scipy import signal
from mne.time_frequency import (tfr_array_morlet)
from mne.baseline import rescale
import pickle
import json

def get_psd_features(total_data_df, psd_config, patient_id, saving_add):


    if patient_id == 'DM1008':
       pass
    else:
        pass
    for trial in total_data_df.trial_id.unique():
        print('trial=%d'%(trial))
        trial_df = total_data_df.loc[(total_data_df.trial_id == trial) | (total_data_df.baseline_flag == trial)].reset_index()
        indx_baselines = trial_df.loc[trial_df.baseline_flag == trial].index.to_numpy()

        neural_data = np.expand_dims(np.transpose(trial_df[psd_config['chnls']].to_numpy()), axis=0)
        neural_psd = psd_extractor(neural_data, psd_config, indx_baselines, type='power')
        # Baseline the output


        if psd_config['smoothing']:
            window = signal.windows.gaussian(psd_config['smoothing_window_size'],
                                             psd_config['smoothing_window_size'] / 8)
            # window = signal.windows.hamming(PSD_config['smoothing_window_size'])
            for ii in range(neural_psd.shape[0]):
                for jj in range(neural_psd.shape[1]):
                    neural_psd[ii, jj, :] = np.convolve(neural_psd[ii, jj, :], window, 'same')
        neural_psd_band, freqs = feature_mapping(neural_psd, psd_config)
        trial_df = trial_df.drop(indx_baselines)
        neural_psd_band = np.delete(neural_psd_band,indx_baselines, axis=-1)

        file_name = saving_add + 'trial_' + str(trial) + '.pkl'
        with  open(file_name, "wb") as open_file:
            pickle.dump([neural_psd_band, trial_df], open_file)


    return  freqs


def psd_extractor(neural_data, psd_config, index_baselines, type='power'):
    freqs = np.arange(psd_config['L_cut_freq'], psd_config['H_cut_freq'], psd_config['freq_stp'])
    n_cycles = freqs
    info = create_info(ch_names=psd_config['chnls'], sfreq=psd_config['sampling_freq'], ch_types='misc', verbose=0)
    epochs = EpochsArray(data=neural_data, info=info, verbose=0)
    if type == 'power':
        power = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                 output='avg_power', verbose=0)
        if len(index_baselines > 0):
            rescale(power, epochs.times, (np.min(index_baselines) / psd_config['sampling_freq'], np.max(index_baselines) / psd_config['sampling_freq']),
            mode='mean', copy=False, verbose=2)

    return power


def feature_mapping(neural_features, psd_config):

    frequency_bands = psd_config['FreqBands']
    # list of frequencies for extracting power
    freqs = np.arange(psd_config['L_cut_freq'], psd_config['H_cut_freq'], psd_config['freq_stp'])
    temp = np.zeros((neural_features.shape[0], len(frequency_bands), neural_features.shape[2]))

    if psd_config["avg_freq_bands"]:
        for ii in range(len(frequency_bands)):
            temp[:, ii, :] = np.median(
                neural_features[:, np.where((freqs >= frequency_bands[ii][0]) &
                                            ((freqs < frequency_bands[ii][1])))[0], :],  axis=1)
        new_neural_features = temp
        freqs = np.arange(len(frequency_bands))
    else:
        new_neural_features= neural_features
    return new_neural_features, freqs

def calDesignMatrix_V2(X,h):
    '''
 design matrix with concat all features
    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def calDesignMatrix_V3(X,h):
    '''
 design matrix with keep features orders
    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''

    PadX = np.zeros([h , X.shape[1], X.shape[2]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1], X.shape[2]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :, :]= (PadX[i:h+i, :, :])
    return XDsgn