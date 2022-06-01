from speech_utils import *
from neural_utils import *
from GMM_utils import *
from scipy.stats import norm
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
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
patient_folder='../Datasets/DM1012/'
phones_DF=pd.read_csv(patient_folder+'sub-DM1012_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv(patient_folder+'sub-DM1012_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv(patient_folder+'sub-DM1012_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
audio_file = patient_folder+"sub-DM1012_ses-intraop_task-lombard_run-02_recording-directionalmicaec_physio.wav"
## df_neural = get_neural_DM1008(patient_folder + 'neural_data_trial3.mat')
## audio_file_res = patient_folder+"sub-DM1012_ses-intraop_task-lombard_run-02_recording-respiration_physio.wav"
# patient_folder='../Datasets/DM1008/'
# phones_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
# sentences_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
# words_DF=pd.read_csv(patient_folder+'sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')
# audio_file = patient_folder+"sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
# patient DM1008
# audio_strt_time = 63623.7147033506
# patient DM1012
audio_strt_time = 72949.95086

mfccs_features_len = 39
neural_smaping_rate=1000
hist_len = 1
thr_spike=.02
'''audio and text synchronization'''
phones_DF['onset'] = phones_DF['onset']-audio_strt_time
sentences_DF['onset'] = sentences_DF['onset']-audio_strt_time
words_DF['onset'] = words_DF['onset']-audio_strt_time
# df_neural['time'] = df_neural['time']- df_neural['time'][0]
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
for uniqe_sent in sentences_DF.sentence.unique()[1:2]:
    Trial_IDs=sentences_DF.trial_id[sentences_DF.sentence == uniqe_sent].to_list()
    loo = LeaveOneOut()

    df_acc = pd.DataFrame(np.zeros((len(Trial_IDs),2)),columns=['MAE_te','RMSE_te'])
    trial_ind=0
    for train, test in loo.split(Trial_IDs):
        Trial_IDs_tr = []
        Trial_IDs_te = []
        for iii in train:
            Trial_IDs_tr.append(Trial_IDs[iii])
        Trial_IDs_te.append(Trial_IDs[test[0]])

        [X_tr,y_tr,X_te,y_te, ac_DF_trial_tr,
         ac_DF_trial_te, featurelist, events_times_tr,
         events_times_te ]= get_tr_te(words_DF, sentences_DF, acustic_feature_DF,
                                                                          dt_features,
                                      Trial_IDs_tr, Trial_IDs_te, mfccs_features_len, Use_neural_fea)

        XD_tr = calDesignMatrix_V2(X_tr,hist_len).reshape([X_tr.shape[0],-1])
        XD_te = calDesignMatrix_V2(X_te,hist_len).reshape([X_te.shape[0],-1])
        pos_GLM = sm.GLM(y_tr, XD_tr, family=sm.families.Poisson())
        # pos_GLM_results = pos_GLM.fit()
        pos_GLM_results = pos_GLM.fit_regularized(L1_wt=0)
        # print(pos_GLM_results.summary())
        y_hat_tr = pos_GLM_results.predict(XD_tr)
        # y_hat_te = pos_GLM_results.predict(XD_te)
        y_hat_te = np.zeros((X_te.shape[0],))
        event_hat_te = np.zeros((X_te.shape[0],))
        numbe_events = ac_DF_trial_tr.wrd.nunique()-1
        history_events = np.zeros((X_te.shape[0],numbe_events))
        history_diff=  np.zeros((X_te.shape[0],))

        X_te_masked=X_te
        # X_te_masked[:,-1]=0

        prior_filter_events_times=np.zeros((numbe_events,X_te_masked.shape[0]))
        prior_filter_events_delay=np.zeros((numbe_events,X_te_masked.shape[0]))

        delays = np.zeros(events_times_tr.shape)
        delays[:,0]=events_times_tr[:,0]
        delays[:,1:]=events_times_tr[:,1:]-events_times_tr[:,:-1]
        # delays[:,-1]=X_te_masked.shape[0]-events_times_tr[:,-1]
        for events in range(numbe_events):
            if events_times_tr[:, events].std() == 0:
                std_n = 3
                std_d = 3
            else:
                std_n = 3*events_times_tr[:, events].std()
                std_d = 3 * delays[:, events].std()
            prior_filter_events_times[events,:] = norm.pdf(np.arange(X_te_masked.shape[0]).squeeze(), events_times_tr[:, events].mean(), std_n)
            prior_filter_events_times[events,:] = prior_filter_events_times[events,:]/prior_filter_events_times[events,:].max()
            prior_filter_events_delay[events, :] = norm.pdf(np.arange(X_te_masked.shape[0]).squeeze(),
                                                        delays[:, events].mean(), std_d)
            prior_filter_events_delay[events, :] = prior_filter_events_delay[events, :] / prior_filter_events_delay[events, :].max()


        t_past=0
        event_ID=0
        for t in range(hist_len, X_te_masked.shape[0]):
            if (event_ID <= numbe_events - 1):
                history_diff[t] = (t - t_past)/X_te_masked.shape[0]
                # X_te_masked[:t,-1] = history_diff[:t]

                XD_te_new = calDesignMatrix_V2(X_te_masked[:t,:],hist_len)
                np.random.choice(delays[:, event_ID])

                # if ((t-t_past)>= np.random.normal(delays[:, event_ID].mean(), delays[:, event_ID].std())):
                if ((t - t_past) >= delays[:, event_ID].mean()):
                    y_hat_te[t] = pos_GLM_results.predict(XD_te_new[-1, :, :].reshape([1, -1]))
                    y_hat_te[t] = y_hat_te[t]*prior_filter_events_times[event_ID, t]#* prior_filter_events_delay[qq, t-t_past]


                #
                    if  ( y_hat_te[t]>thr_spike) :
                        # print(qq)
                        event_hat_te[t]=1
                        event_ID = event_ID + 1
                        t_past = t
                #     print( y_hat_te[t])
                # if qq == numbe_events-1:
                #     break
        # y_hat_te=y_hat_te * prior_filter_events_times[3,:]#.mean(axis=0)
        y_hat_te/=np.nanmax(y_hat_te)
        mae_te = np.mean(np.abs(np.where(event_hat_te != 0)[0] - events_times_te))
        rmse_te = np.sqrt(np.mean(((np.where(event_hat_te != 0)[0] - events_times_te)**2)))
        df_acc['MAE_te'][trial_ind] = mae_te*dt_features
        df_acc['RMSE_te'][trial_ind] = rmse_te*dt_features
        trial_ind+=1
        hist_cols = [col for col in ac_DF_trial_tr.columns if 'history_events' in col]
        # plt.figure()
        # doms_tr =np.arange(0,y_tr.shape[0])*dt_features
        # plt.plot(doms_tr,y_tr, 'r',label = 'True_val')
        # plt.plot(doms_tr,y_hat_tr, 'b',label = 'Pred_val')
        # plt.xlabel('time (s)')
        # for ii in ac_DF_trial_tr.wrd.unique():
        #     if ii == 'NAN':
        #         pass
        #     else:
        #         indxes_words =ac_DF_trial_tr[ac_DF_trial_tr.wrd == ii].index.to_numpy()
        #         for kk in range(len(indxes_words)):
        #             plt.annotate(ii, (indxes_words[kk] *dt_features,.8),rotation=90)
        #         # plt.text(ac_DF_trial_tr[ac_DF_trial_tr.wrd == ii].index.to_numpy()[0] ,1, ii)
        # plt.title(str(ac_DF_trial_tr.wrd.unique().tolist()))
        # plt.legend()
        # plt.show()
        # plt.figure()
        #
        # plt.plot(ac_DF_trial_tr[hist_cols])
        # plt.plot(ac_DF_trial_tr.history_diff)
        plt.figure()
        doms_te =np.arange(0,y_te.shape[0])*dt_features
        plt.plot(doms_te,y_te, 'r',label = 'True_val')
        plt.plot(doms_te,y_hat_te, 'b',label = 'Pred_val')
        # plt.plot(doms_te,prior_filter_events_times.transpose())
        plt.stem(doms_te,event_hat_te)
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
        # plt.figure()
        # plt.plot(ac_DF_trial_te[hist_cols])
        # plt.plot(ac_DF_trial_te.history_diff)
        #
        plt.figure()
        plt.plot(prior_filter_events_times.transpose())
        # plt.stem(event_hat_te)
        plt.title('prior_filter_events_times')

        plt.figure()
        plt.plot(prior_filter_events_delay.transpose())
        # plt.stem(event_hat_te)
        plt.title('prior_filter_events_delay')

    plt.figure()
    df_acc.boxplot()
pos_GLM_parm_df = pd.DataFrame(pos_GLM_results.params.reshape([1, -1]).squeeze(), columns=['par_val'])
pos_GLM_parm_df['significance'] = np.abs(pos_GLM_parm_df.par_val) / pos_GLM_parm_df.par_val.max()
pos_GLM_parm_df['parm_name'] = res_list = [*['bias'], *featurelist]
pos_GLM_parm_df = pos_GLM_parm_df.sort_values(by='significance', ascending=False)
plt.figure()
plt.stem(pos_GLM_parm_df.parm_name, pos_GLM_parm_df.significance)
plt.xticks(rotation=90)