

import numpy as np

from mne import create_info, EpochsArray
from scipy import signal
from mne.time_frequency import (tfr_array_morlet)
from mne.baseline import rescale
import pickle
import json
from neurodsp.filt import filter_signal
from scipy.signal import hilbert, chirp
import numpy as np
import matplotlib.pyplot as plt

def get_psd_features(total_data_df, psd_config, patient_id, saving_add):

    margin_length = psd_config['margin_length']

    for trial in total_data_df.trial_id.unique():
        print('trial=%d of %d'%(trial,len(total_data_df.trial_id.unique())))
        if trial == total_data_df.trial_id.unique()[0]: #
            indx_df = total_data_df.loc[
                (total_data_df.trial_id == trial) | (total_data_df.baseline_flag == trial)].index
            indx_baselines = total_data_df.loc[
                 (total_data_df.baseline_flag == trial)].index

            af_indx = np.arange(indx_df[-1] + 1, indx_df[-1] + 1 + margin_length, 1)

            total_index = np.concatenate([ indx_df, af_indx], axis=0)
            trial_df = total_data_df.loc[total_index].reset_index()

            drop_indx = np.arange(0, len(indx_baselines))
            temp_af = np.arange(trial_df.shape[0]- 1, trial_df.shape[0] - 1 - margin_length, -1)
            drop_indx = np.concatenate([drop_indx, temp_af], axis=0)
            neural_data = np.expand_dims(np.transpose(trial_df[psd_config['chnls']].to_numpy()), axis=0)
            neural_psd, freqs = psd_extractor(neural_data, psd_config, np.arange(0, len(indx_baselines)), type='power')
        elif trial == total_data_df.trial_id.unique()[-1]:
            indx_df = total_data_df.loc[
                (total_data_df.trial_id == trial) | (total_data_df.baseline_flag == trial)].index
            indx_baselines = total_data_df.loc[
                (total_data_df.baseline_flag == trial)].index
            if len(indx_baselines > 0):
                bf_indx = np.arange( indx_baselines[0] - margin_length, indx_baselines[0], 1)
            else:
                bf_indx=[]



            total_index = np.concatenate([bf_indx, indx_df], axis=0)
            trial_df = total_data_df.loc[total_index].reset_index()

            drop_indx = np.arange(margin_length, len(indx_baselines) + margin_length)
            temp_bf = np.arange(0, margin_length)
            drop_indx = np.concatenate([temp_bf, drop_indx], axis=0)
            neural_data = np.expand_dims(np.transpose(trial_df[psd_config['chnls']].to_numpy()), axis=0)
            neural_psd, freqs = psd_extractor(neural_data, psd_config,  np.arange(margin_length, len(indx_baselines) + margin_length), type='power')

        else:
            indx_df = total_data_df.loc[(total_data_df.trial_id == trial) | (total_data_df.baseline_flag == trial)].index
            indx_baselines = total_data_df.loc[
                (total_data_df.baseline_flag == trial)].index
            bf_indx = np.arange(indx_baselines[0]-margin_length,indx_baselines[0], 1)
            af_indx = np.arange(indx_df[-1] + 1, indx_df[-1] + 1 + margin_length, 1)

            total_index = np.concatenate([bf_indx, indx_df, af_indx], axis=0)
            trial_df = total_data_df.loc[total_index].reset_index()

            drop_indx = np.arange(margin_length, len(indx_baselines) + margin_length)
            temp_bf = np.arange(0, margin_length)
            temp_af = np.arange(trial_df.shape[0] - 1, trial_df.shape[0]  - 1 - margin_length, -1)
            drop_indx = np.concatenate([temp_bf, drop_indx, temp_af])

            neural_data = np.expand_dims(np.transpose(trial_df[psd_config['chnls']].to_numpy()), axis=0)
            neural_psd, freqs = psd_extractor(neural_data, psd_config,  np.arange(margin_length, len(indx_baselines) + margin_length), type='power')
            # Baseline the output


        if psd_config['smoothing']:
            window = signal.windows.gaussian(psd_config['smoothing_window_size'],
                                             psd_config['smoothing_window_size'] / 8)
            # window = signal.windows.hamming(PSD_config['smoothing_window_size'])
            for ii in range(neural_psd.shape[0]):
                for jj in range(neural_psd.shape[1]):
                    neural_psd[ii, jj, :] = np.convolve(neural_psd[ii, jj, :], window, 'same')
        if psd_config['avg_freq_bands']:
            neural_psd_band, freqs = feature_mapping(neural_psd, psd_config,freqs)
        else:
            neural_psd_band=neural_psd
            freqs=1
        trial_df = trial_df.drop(drop_indx)
        neural_psd_band = np.delete(neural_psd_band, drop_indx, axis=-1)



        file_name = saving_add + 'trial_' + str(trial) + '.pkl'
        with  open(file_name, "wb") as open_file:
            pickle.dump([neural_psd_band, trial_df], open_file)


    return  freqs


def get_psd_features_direct(neural_df, times, trials_df, sentences_df, psd_config, saving_add):

    margin_length = psd_config['margin_length']
    baseline_length = psd_config['baseline_length']
    sampling_freq = psd_config['sampling_freq']
    times=times - sentences_df.onset[0]
    sentences_df.onset = sentences_df.onset - sentences_df.onset[0]

    for trial in trials_df.trial_id.unique():
        print(trial)
        # start_trial_time=trials_df.onset[trial]
        # duration_trial = trials_df.duration[trial]
        start_sent_time = sentences_df.onset[trial]
        duration_sent = sentences_df.duration[trial]
        # start_itg = trials_df.itg_onset[trial]
        # duration_itg = trials_df.itg_duration[trial]
        t_start=start_sent_time-2*margin_length/sampling_freq-baseline_length/sampling_freq
        t_end=start_sent_time+duration_sent+margin_length/sampling_freq
        # print(t_end-t_start-duration_sent)
        indx_df = neural_df.iloc[np.where((times>=t_start) &
                                          (times<=t_end))[0]].index
        drop_indx = np.arange(0, baseline_length+margin_length)

        trial_df = neural_df.loc[indx_df]
        neural_data = np.expand_dims(np.transpose(trial_df.to_numpy()), axis=0)

        neural_psd, freqs = psd_extractor(neural_data, psd_config,  np.arange(margin_length, baseline_length+margin_length), type='power')
                # Baseline the output


        if psd_config['smoothing']:
                window = signal.windows.gaussian(psd_config['smoothing_window_size'],
                                                 psd_config['smoothing_window_size'] / 8)
                # window = signal.windows.hamming(PSD_config['smoothing_window_size'])
                for ii in range(neural_psd.shape[0]):
                    for jj in range(neural_psd.shape[1]):
                        neural_psd[ii, jj, :] = np.convolve(neural_psd[ii, jj, :], window, 'same')
        if psd_config['avg_freq_bands']:
            neural_psd_band, freqs = feature_mapping(neural_psd, psd_config,freqs)
        else:
            neural_psd_band=neural_psd
            freqs=1

        neural_psd_band = np.delete(neural_psd_band, drop_indx, axis=-1)



        file_name = saving_add + 'trial_' + str(trial) + '.pkl'
        with  open(file_name, "wb") as open_file:
            pickle.dump(neural_psd_band, open_file)


    return  freqs

def psd_extractor(neural_data, psd_config, index_baselines, type='power'):
    freqs = np.round(np.logspace(np.log10(psd_config['L_cut_freq']),np.log10(psd_config['H_cut_freq']),psd_config['numb_freq'])).astype('int')
    n_cycles = freqs#psd_config['L_cut_freq']
    info = create_info(ch_names=psd_config['chnls'], sfreq=psd_config['sampling_freq'], ch_types='misc', verbose=0)
    epochs = EpochsArray(data=neural_data, info=info, verbose=0)
    if type == 'power':
        power = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                 output='power', verbose=0).squeeze()
        if len(index_baselines)>100:
            power=rescale(power, epochs.times, (np.min(index_baselines) / psd_config['sampling_freq'], np.max(index_baselines) / psd_config['sampling_freq']),
            mode='zscore', copy=False, verbose=2)

    return power,freqs

def psd_extractor_v2(neural_data, psd_config, index_baselines, type='power'):
    info = create_info(ch_names=psd_config['chnls'], sfreq=psd_config['sampling_freq'], ch_types='misc', verbose=0)
    epochs = EpochsArray(data=neural_data, info=info, verbose=0)
    for ii_fb in range(len(psd_config['FreqBands'])):
        sig_filt = filter_signal(neural_data.squeeze(), psd_config['sampling_freq'], 'bandpass', [psd_config['FreqBands'][ii_fb][0],
                                                                                psd_config['FreqBands'][ii_fb][1]])
        sig_filt = np.nan_to_num(sig_filt)
        analytic_signal = hilbert(sig_filt)
        amplitude_envelope = np.abs(analytic_signal)
        if ii_fb ==0:
            all=np.expand_dims(amplitude_envelope, axis=1)
        else:
            all=np.concatenate([all,np.expand_dims(amplitude_envelope, axis=1)],axis=1)
    if len(index_baselines )>0:
        all = rescale(all, epochs.times, (
        np.min(index_baselines) / psd_config['sampling_freq'], np.max(index_baselines) / psd_config['sampling_freq']),
                        mode='zscore', copy=False, verbose=2)
    # for ii_chn in range(all.shape[0]):
    #     for ii_band in range(all.shape[1]):
    #         all[ii_chn,ii_band,:]=(all[ii_chn,ii_band,:]-np.nanmean(all[ii_chn,ii_band,index_baselines]))/np.nanstd(all[ii_chn,ii_band,index_baselines])
    return all

def feature_mapping(neural_features, psd_config,freqs):

    frequency_bands = psd_config['FreqBands']
    # list of frequencies for extracting power

    temp = np.zeros((neural_features.shape[0], len(frequency_bands), neural_features.shape[2]))

    if psd_config["avg_freq_bands"]:
        for ii in range(len(frequency_bands)):
            temp[:, ii, :] = np.nanmedian(
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

def calDesignMatrix_V4(X,h, f,d_sample):
    '''
    h history and f future
 design matrix with keep features orders
    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    indx_d=np.arange(0,h+f,d_sample)
    # # window=np.zeros([f+h, X.shape[1], X.shape[2]])
    # for ii in range(X.shape[1]):
    #     for jj in range(X.shape[2]):
    #         X[:,ii,jj]= np.convolve(X[:,ii,jj],signal.windows.gaussian(50, std=10),'same')

    PadX_h = np.zeros([h, X.shape[1], X.shape[2]])
    PadX_f =np.zeros([f, X.shape[1], X.shape[2]])
    PadX =np.concatenate([PadX_h,X, PadX_f],axis=0)
    XDsgn=np.zeros([X.shape[0], len(indx_d), X.shape[1], X.shape[2]])
    # print(PadX.shapepe)
    for i in range(h, XDsgn.shape[0]):

        XDsgn[i-h, :, :, :] = PadX[i-h:f+i, :, :][indx_d]

    return XDsgn