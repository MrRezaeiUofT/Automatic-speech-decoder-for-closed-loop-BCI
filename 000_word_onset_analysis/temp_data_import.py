from speech_utils import *
from GMM_utils import *
import pandas as pd
from sklearn import preprocessing

patient_folder='../Datasets/DM1008/'
phones_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
audio_file = patient_folder+"sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
audio_strt_time = 63623.7147033506

'''audio and text synchronization'''
phones_DF['onset'] = phones_DF['onset']-audio_strt_time
sentences_DF['onset'] = sentences_DF['onset']-audio_strt_time
words_DF['onset'] = words_DF['onset']-audio_strt_time

''' Extracting MFCCs'''
mfccs_features, sr,signal = mfcc_feature_extraction(audio_file, 13)
dt_features = 1/(mfccs_features.shape[1]/(signal.shape[0]/sr))
mfccs_features = preprocessing.scale(mfccs_features, axis=1)
plt.figure(figsize=(25, 10))
out=librosa.display.specshow(mfccs_features,
                         x_axis="time",
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title("delta2_mfccs")
plt.show()
acustic_feature_DF=pd.DataFrame(mfccs_features.transpose(), columns=['f_'+str(i) for i in range(mfccs_features.shape[0])])


''' Sentence specified model'''
Trial_ID=1
words_DF_trial = words_DF[words_DF.trial_id == Trial_ID]



snt_duration = words_DF_trial.duration.sum()
snt_onset = sentences_DF[sentences_DF.trial_id == Trial_ID].onset.to_numpy().squeeze()
snt_onset_indx = int(np.floor(snt_onset // dt_features))
snt_duration_len = int(np.floor(snt_duration // dt_features))
ac_DF_trial = acustic_feature_DF.iloc[snt_onset_indx:snt_onset_indx + snt_duration_len]


wrd_event = np.zeros(ac_DF_trial.shape[0],)
wrd_duration = np.zeros(ac_DF_trial.shape[0],)
delay_btwn_wrd = np.zeros(ac_DF_trial.shape[0],)
wrd_type =  []
last_wrd_time=0
for ii in range(wrd_event.shape[0]):
    time_int = [snt_onset+ii*dt_features,snt_onset+(ii+1)*dt_features]
    identified_wrd = words_DF_trial.word.loc[(words_DF_trial.onset<=time_int[1]) & (words_DF_trial.onset>=time_int[0])]
    identified_wrd_duration = words_DF_trial.duration.loc[(words_DF_trial.onset <= time_int[1]) & (words_DF_trial.onset >= time_int[0])]
    if (len(identified_wrd) == 0) or (identified_wrd.to_list()[0] == 'sp') :
        # No word for the time interval
        wrd_event[ii] = 0
        wrd_duration[ii] = 0
        wrd_type.append('NAN')

    elif len(identified_wrd) == 1:
        # One word for the time interval
        wrd_event[ii] = 1
        wrd_duration[ii] = identified_wrd_duration//dt_features
        delay_btwn_wrd[ii] = ii - last_wrd_time
        last_wrd_time = ii
        wrd_type.append( identified_wrd.to_list()[0])
    else:
        # select the longest word for the time interval
        selected=np.where(identified_wrd_duration == identified_wrd_duration.max())[0]
        wrd_event[ii] = 2
        wrd_duration[ii] = identified_wrd_duration.to_numpy()[selected]//dt_features
        wrd_type.append(identified_wrd.to_list()[selected[0]])
        delay_btwn_wrd[ii] = ii - last_wrd_time
        last_wrd_time = ii
ac_DF_trial['wrd_event']=wrd_event
ac_DF_trial['wrd']=wrd_type
ac_DF_trial['wrd_duration']=wrd_duration
ac_DF_trial['diff_from_last_wrd']=delay_btwn_wrd

''' build the CIF for events'''

M = 51
tau = 1.0
window = signal.windows.exponential(M, tau=tau)
ac_DF_trial['synth_CIF']=np.convolve(ac_DF_trial['wrd_event'],window[:M//2],'same' )
ac_DF_trial['synth_CIF']=ac_DF_trial['synth_CIF']/ac_DF_trial['synth_CIF'].max()
plt.plot(ac_DF_trial['synth_CIF'])

