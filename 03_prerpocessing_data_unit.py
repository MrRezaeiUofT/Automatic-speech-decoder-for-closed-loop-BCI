from speech_utils import *
from GMM_utils import *
import pandas as pd
from sklearn import preprocessing


patient_folder='./Datasets/DM1008/'
phones_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
audio_file = patient_folder+"sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
audio_strt_time = 63623.7147033506
phones_DF.phoneme.hist()

'''audio and text synchronization'''
phones_DF['onset'] = phones_DF['onset']-audio_strt_time
sentences_DF['onset'] = sentences_DF['onset']-audio_strt_time
words_DF['onset'] = words_DF['onset']-audio_strt_time

''' Extracting MFCCs'''
mfccs_features, sr,signal = mfcc_feature_extraction(audio_file, 13)
mfccs_features = preprocessing.scale(mfccs_features, axis=1)
plt.figure(figsize=(25, 10))
out=librosa.display.specshow(mfccs_features,
                         x_axis="time",
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title("delta2_mfccs")
plt.show()
acustic_feature_DF=pd.DataFrame(mfccs_features.transpose(), columns=['f_'+str(i) for i in range(mfccs_features.shape[0])])
dt_features = 1/(mfccs_features.shape[1]/(signal.shape[0]/sr))

''' gathering the information'''
phones_event = np.zeros(mfccs_features.shape[1],)
phones_duration = np.zeros(mfccs_features.shape[1],)
phones_type =  []
for ii in range(phones_event.shape[0]):
    time_int = [ii*dt_features,(ii+1)*dt_features]
    identified_phones = phones_DF.phoneme.loc[(phones_DF.onset<=time_int[1]) & (phones_DF.onset>=time_int[0])]
    identified_phone_duration = phones_DF.duration.loc[(phones_DF.onset <= time_int[1]) & (phones_DF.onset >= time_int[0])]
    if len(identified_phones) == 0:
        # No phoneme for the time interval
        phones_event[ii] = 0
        phones_duration[ii] = 0
        phones_type.append('NAN')

    elif len(identified_phones) == 1:
        # One phoneme for the time interval
        phones_event[ii] = 1
        phones_duration[ii] = identified_phone_duration//dt_features
        phones_type.append( identified_phones.to_list()[0])
    else:
        # select the longest phoneme for the time interval
        selected=np.where(identified_phone_duration == identified_phone_duration.max())[0]
        phones_event[ii] = 2
        phones_duration[ii] = identified_phone_duration.to_numpy()[selected]//dt_features
        phones_type.append(identified_phones.to_list()[selected[0]])
acustic_feature_DF['phone_event']=phones_event
acustic_feature_DF['phone']=phones_type
acustic_feature_DF['phone_duration']=phones_duration

# consider all length of thee phones
for ii in range(len(acustic_feature_DF)):
    print(ii)
    if acustic_feature_DF['phone_event'][ii] !=0:
        number_e=int(acustic_feature_DF['phone_duration'][ii])
        acustic_feature_DF['phone_event'][ii:ii+number_e]=acustic_feature_DF['phone_event'][ii]
        acustic_feature_DF['phone'][ii:ii + number_e] = acustic_feature_DF['phone'][ii]


'''save the result'''
acustic_feature_DF.to_csv(patient_folder+'preprocessed_dataframe.csv',index=False)
acustic_feature_DF.phone.hist()





