from speech_utils import *
from neural_utils import *
from GMM_utils import *
import pandas as pd
from sklearn import preprocessing

import statsmodels.api as sm
def calDesignMatrix_V2(X,h):
    '''

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
Use_neural_fea= False
patient_folder='../Datasets/DM1008/'
phones_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
audio_file = patient_folder+"sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
df_neural = get_neural_DM1008(patient_folder + 'neural_data_trial3.mat')
audio_file_res = patient_folder+"sub-DM1008_ses-intraop_task-lombard_run-03_recording-respiration_physio.wav"
audio_strt_time = 63623.7147033506
mfccs_features_len = 39
neural_smaping_rate=1000
hist_len = 5
'''audio and text synchronization'''
phones_DF['onset'] = phones_DF['onset']-audio_strt_time
sentences_DF['onset'] = sentences_DF['onset']-audio_strt_time
words_DF['onset'] = words_DF['onset']-audio_strt_time
df_neural['time'] = df_neural['time']- df_neural['time'][0]
''' Extracting MFCCs'''
mfccs_features, sr,signal_ac = mfcc_feature_extraction(audio_file, 13)
dt_features = 1/(mfccs_features.shape[1]/(signal_ac.shape[0]/sr))
mfccs_features = preprocessing.scale(mfccs_features, axis=1)

# mfccs_features_res, sr_res,signal_ac_res = mfcc_feature_extraction(audio_file_res, 13)
# mfccs_features_res= preprocessing.scale(mfccs_features_res, axis=1)
#
# mfccs_features = np.concatenate([mfccs_features_res,mfccs_features],axis=0)
# plt.figure(figsize=(25, 10))
# out=librosa.display.specshow(mfccs_features,
#                          x_axis="time",
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.title("delta2_mfccs")
# plt.show()
acustic_feature_DF=pd.DataFrame(mfccs_features.transpose(), columns=['f_'+str(i) for i in range(mfccs_features.shape[0])])


''' Sentence specified model'''
Trial_IDs=sentences_DF.trial_id[sentences_DF.sentence == sentences_DF.sentence.unique()[1]].to_list()
Trial_IDs_tr=Trial_IDs[2:]
Trial_IDs_te= Trial_IDs[:2]

X_tr,y_tr,X_te,y_te, ac_DF_trial_tr, ac_DF_trial_te, featurelist, events_times_tr, events_times_te = get_tr_te(words_DF, sentences_DF, acustic_feature_DF, df_neural,
                                                                  dt_features, neural_smaping_rate,Trial_IDs_tr, Trial_IDs_te, mfccs_features_len,
              Use_neural_fea)

XD_tr = calDesignMatrix_V2(X_tr,hist_len).reshape([X_tr.shape[0],-1])
XD_te = calDesignMatrix_V2(X_te,hist_len).reshape([X_te.shape[0],-1])
pos_GLM = sm.GLM(y_tr, XD_tr, family=sm.families.Poisson())
# pos_GLM_results = pos_GLM.fit()
pos_GLM_results = pos_GLM.fit_regularized(L1_wt=.5)
# print(pos_GLM_results.summary())
y_hat_tr = pos_GLM_results.predict(XD_tr)
y_hat_te = pos_GLM_results.predict(XD_te)
# y_hat_te = np.zeros((X_te.shape[0],))
# event_hat_te = np.zeros((X_te.shape[0],))
# numbe_events = ac_DF_trial_tr.wrd.nunique()-1
# history_events = np.zeros((X_te.shape[0],numbe_events))
# history_diff=  np.zeros((X_te.shape[0],))
#
# t_past=0
# X_te_masked=X_te
# X_te_masked[:,-9:]=0
# qq=0
# for t in range(hist_len, X_te_masked.shape[0]):
#         history_diff[t] = (t - t_past)/X_te_masked.shape[0]/2
#         history_events[:t, qq:] = ((np.arange(t) / X_te_masked.shape[0]/2) ** 2).reshape([-1,1])
#
#         X_te_masked[:t,-9] = history_diff[:t]
#         X_te_masked[:t, -8:] = history_events[:t,:]
#         XD_te_new = calDesignMatrix_V2(X_te_masked[:t,:],hist_len)
#         y_hat_te[t] = pos_GLM_results.predict(XD_te_new[-1,:,:].reshape([1,-1]))
#         if y_hat_te[t]>10:
#             y_hat_te[t]=10
#         pred_event=np.random.poisson(lam=y_hat_te[t], size=1)
#
#         if pred_event > 0:
#
#             if (t>= events_times_tr[:,qq].mean() - 4*events_times_tr[:,qq].std()) and (t<= events_times_tr[:,qq].mean() + 4*events_times_tr[:,qq].std()):
#                 event_hat_te[t] =1
#                 qq = qq + 1
#                 t_past = t
#             else:
#                 print('unable to detect')
#

# y_hat/=np.max(y_hat)
plt.figure()
doms_tr =np.arange(0,y_tr.shape[0])*dt_features
plt.plot(doms_tr,y_tr, 'r',label = 'True_val')
plt.plot(doms_tr,y_hat_tr, 'b',label = 'Pred_val')
plt.xlabel('time (s)')
for ii in ac_DF_trial_tr.wrd.unique():
    if ii == 'NAN':
        pass
    else:
        indxes_words =ac_DF_trial_tr[ac_DF_trial_tr.wrd == ii].index.to_numpy()
        for kk in range(len(indxes_words)):
            plt.annotate(ii, (indxes_words[kk] *dt_features,.8),rotation=90)
        # plt.text(ac_DF_trial_tr[ac_DF_trial_tr.wrd == ii].index.to_numpy()[0] ,1, ii)
plt.title(str(ac_DF_trial_tr.wrd.unique().tolist()))
plt.legend()
plt.show()
plt.figure()
hist_cols = [col for col in ac_DF_trial_tr.columns if 'history_events' in col]
plt.plot(ac_DF_trial_tr[hist_cols])
plt.plot(ac_DF_trial_tr.history_diff)

plt.figure()
doms_te =np.arange(0,y_te.shape[0])*dt_features
plt.plot(doms_te,y_te, 'r',label = 'True_val')
plt.plot(doms_te,y_hat_te, 'b',label = 'Pred_val')
plt.xlabel('time (s)')
for ii in ac_DF_trial_te.wrd.unique():
    if ii == 'NAN':
        pass
    else:
        indxes_words =ac_DF_trial_te[ac_DF_trial_te.wrd == ii].index.to_numpy()
        for kk in range(len(indxes_words)):
            plt.annotate(ii, (indxes_words[kk] *dt_features,.8),rotation=90)
        # plt.text(ac_DF_trial_tr[ac_DF_trial_tr.wrd == ii].index.to_numpy()[0] ,1, ii)
plt.title(str(ac_DF_trial_te.wrd.unique().tolist()))
plt.legend()
plt.show()
plt.figure()
plt.plot(ac_DF_trial_te[hist_cols])
plt.plot(ac_DF_trial_te.history_diff)
# plt.figure()
# plt.stem(event_hat_te)

# if Use_neural_fea :
#     pos_GLM_parm_df = pd.DataFrame(pos_GLM_results.params.reshape([1, -1]).squeeze(), columns=['par_val'])
#     pos_GLM_parm_df['significance'] = np.abs(pos_GLM_parm_df.par_val) / pos_GLM_parm_df.par_val.max()
#     pos_GLM_parm_df['parm_name'] = df_neural.columns
#     pos_GLM_parm_df['parm_name'][0] = 'bias'
#     pos_GLM_parm_df = pos_GLM_parm_df.sort_values(by='significance', ascending=False)
#     plt.figure()
#     plt.stem(pos_GLM_parm_df.parm_name, pos_GLM_parm_df.significance)
#     plt.xticks(rotation=90)
# else:
#
#     pos_GLM_parm_df = pd.DataFrame(pos_GLM_results.params.reshape([1, -1]).squeeze(), columns=['par_val'])
#     pos_GLM_parm_df['significance'] = np.abs(pos_GLM_parm_df.par_val) / pos_GLM_parm_df.par_val.max()
#     pos_GLM_parm_df['parm_name'] = res_list = [*['bias'], *featurelist]
#     pos_GLM_parm_df = pos_GLM_parm_df.sort_values(by='significance', ascending=False)
#     plt.figure()
#     plt.stem(pos_GLM_parm_df.parm_name, pos_GLM_parm_df.significance)
#     plt.xticks(rotation=90)

