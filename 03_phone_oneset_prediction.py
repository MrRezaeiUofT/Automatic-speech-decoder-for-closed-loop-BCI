from speech_utils import *
from GMM_utils import *
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm

N_gram=2
phones_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
audio_file = "./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
audio_strt_time = 63623.7147033506
phones_DF.phoneme.hist()
###

# audio and text synchronization
phones_DF['onset'] = phones_DF['onset']-audio_strt_time
sentences_DF['onset'] = sentences_DF['onset']-audio_strt_time
words_DF['onset'] = words_DF['onset']-audio_strt_time
# Extracting MFCCs


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

# gathering the information

phones_event = np.zeros(mfccs_features.shape[1],)
phones_type =  []
for ii in range(phones_event.shape[0]):
    time_int = [ii*dt_features,(ii+1)*dt_features]
    identified_phones = phones_DF.phoneme.loc[(phones_DF.onset<=time_int[1]) & (phones_DF.onset>=time_int[0])]
    if len(identified_phones) == 0:
        phones_event[ii] = 0
        phones_type.append('NAN')
    else:
        phones_event[ii] = 1
        phones_type.append( identified_phones.to_list()[0])

acustic_feature_DF['phone_even']=phones_event
acustic_feature_DF['phone']=phones_type
acustic_feature_DF.to_csv('./Datasets/DM1008/preprocessed_data.csv')
# phones N-gram model
phones_NgramModel=NgramModel(N_gram)
phones_NgramModel.update(sentence=listToString(acustic_feature_DF['phone'].to_list()), need_tokenize=True)
print(phones_NgramModel.prob(('HH',),'IY1'))
print(phones_NgramModel.map_to_probs(('HH',)))
plt.figure()
acustic_feature_DF.phone.hist()
## GLM binomial GLM fit for events
#
# X=acustic_feature_DF[['f_'+str(i) for i in range(mfccs_features.shape[0])]].to_numpy()
# Y=phones_event
# bin_GLM = sm.GLM(Y, X, family=sm.families.Binomial())
# bin_GLM_results = bin_GLM.fit()
# # print(bin_GLM_results.summary())
# y_hat=bin_GLM_results.predict(X)
# y_hat[y_hat>=.5]= 1
# y_hat[y_hat<.5]= 0
# print('binomial GLM error_rate=%f'%(100*np.sum(np.abs(Y-y_hat))/y_hat.shape[0]))

# acustic model by GMMs
gmm_number_Comp=2
acustic_model={}

for phone_id in acustic_feature_DF.phone.unique():
    X = acustic_feature_DF[['f_'+str(i) for i in range(mfccs_features.shape[0])]][acustic_feature_DF.phone == phone_id]
    if X.shape[0]>1:
        acustic_model[phone_id]=GaussianMixture(n_components=gmm_number_Comp).fit(X)
    else:
        acustic_model[phone_id]= None

# Bayesian filtering

