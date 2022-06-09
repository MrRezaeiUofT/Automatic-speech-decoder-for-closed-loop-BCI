
import h5py
import scipy.io
import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm
M = 51
tau = 1.0

# patient_folder='./Datasets/DM1008/'
# filepath = patient_folder + 'neural_data_trial3.mat'
def get_neural_DM1008(filepath):
    '''
    get the neural data for the third trial
    :param filepath:
    :return:
    '''
    mat = scipy.io.loadmat(filepath)

    labels=["time",	"ecog_L101",	"ecog_L102",	"ecog_L103",	"ecog_L104",	"ecog_L105","ecog_L106",	"ecog_L107",	"ecog_L108",
        "ecog_L109",	"ecog_L110"	,"ecog_L111"	,"ecog_L112"	,"ecog_L113"	,"ecog_L114"	,"ecog_L115",	"ecog_L116",	"ecog_L117"	,
        "ecog_L118",	"ecog_L119",	"ecog_L120",	"ecog_L121",	"ecog_L122",	"ecog_L123"	,"ecog_L124",	"ecog_L125",	"ecog_L126",
        "ecog_L127"	,"ecog_L128"	,"ecog_L129"	,"ecog_L130"	,"ecog_L131"	,"ecog_L132"	,"ecog_L133",	"ecog_L134",	"ecog_L135",
        "ecog_L136",	"ecog_L137"	,"ecog_L138"	,"ecog_L139"	,"ecog_L140"	,"ecog_L141",	"ecog_L142",	"ecog_L143",	"ecog_L144",
        "ecog_L145",	"ecog_L146"	,"ecog_L147"	,"ecog_L148"	,"ecog_L149"	,"ecog_L150"	,"ecog_L151",	"ecog_L152"	,"ecog_L153"	,
        "ecog_L154",	"ecog_L155",	"ecog_L156",	"ecog_L157",	"ecog_L158",	"ecog_L159",	"ecog_L160"	,"ecog_L161",	"ecog_L162",
        "ecog_L163",	"v0_A232"	,"ecog_L201"	,"ecog_L202"	,"ecog_L203"	,"ecog_L204"	,"ecog_L205",	"ecog_L206",	"ecog_L207"	,
        "ecog_L208"	,"ecog_L209"	,"ecog_L210"	,"ecog_L211"	,"ecog_L212"	,"ecog_L213"	,"ecog_L214",	"ecog_L215",	"ecog_L216"	,
        "ecog_L217",	"ecog_L218"	,"ecog_L219"	,"ecog_L220"	,"ecog_L221"	,"ecog_L222"	,"ecog_L223",	"ecog_L224",	"ecog_L225",
        "ecog_L226"	,"ecog_L227"	,"ecog_L228"	,"ecog_L229"	,"ecog_L230"	,"ecog_L231"	,"ecog_L232",	"ecog_L233",	"ecog_L234",
        "ecog_L235"	,"ecog_L236"	,"ecog_L237"	,"ecog_L238"	,"ecog_L239",	"ecog_L240",	"ecog_L241",	"ecog_L242"	,"ecog_L243"	,
        "ecog_L244"	,"ecog_L245"	,"ecog_L246"	,"ecog_L247"	,"ecog_L248",	"ecog_L249",	"ecog_L250",	"ecog_L251",	"ecog_L252",
        "ecog_L253",	"ecog_L254"	,"ecog_L255"	,"ecog_L256"	,"ecog_L257"	,"ecog_L258",	"ecog_L259",	"ecog_L260",	"ecog_L261",
        "ecog_L262",	"ecog_L263"	,"v0_con8",	"dbs_L1",	"dbs_L2A",	"dbs_L2B",	"dbs_L2C"	,"dbs_L3A"	,"dbs_L3B"	,
        "dbs_L3C"	,"dbs_L4",	"v0_C10",	"dbs_R1",	"dbs_R2A"	,"dbs_R2B",	"dbs_R2C",	"dbs_R3A",	"dbs_R3B"	,
        "dbs_R3C",	"dbs_R4"	,"v0_C11",	"resp_1",	"v0_D102",	"v)_D103",	"pd_1",	"audioR_s",	"audioR_p"	,"audioR_a",
        "micro_Ll" ,"micro_Lc",	"micro_Lp",	"macro_Ll",	"macro_Lc",	"macro_Lp",	"audioAO_s",	"audio_p",	"envaudio_p",
        "audio_s"	,"envaudio_s"]


    df=pd.DataFrame(mat['dataframe'].transpose(), columns=labels)

    # remove unexpected columns
    df_neural = df.loc[:, ~df.columns.isin( ["v0_con8",	"dbs_L1",	"dbs_L2A",	"dbs_L2B",	"dbs_L2C"	,"dbs_L3A"	,"dbs_L3B"	,
        "dbs_L3C"	,"dbs_L4",	"v0_C10",	"dbs_R1",	"dbs_R2A"	,"dbs_R2B",	"dbs_R2C",	"dbs_R3A",	"dbs_R3B"	,
        "dbs_R3C",	"dbs_R4"	,"v0_C11",	"resp_1",	"v0_D102",	"v)_D103",	"pd_1",	"audioR_s",	"audioR_p"	,"audioR_a",
        "micro_Ll" ,"micro_Lc",	"micro_Lp",	"macro_Ll",	"macro_Lc",	"macro_Lp",	"audioAO_s",	"audio_p",	"envaudio_p",
        "audio_s"	,"envaudio_s"])]

    df_neural=df_neural.fillna(df_neural.mean())

    # standardizing data
    df_neural=(df_neural - df_neural.mean()) / df_neural.std()
    df_neural = df_neural.fillna(0)
    return df_neural


# df_neural = get_neural(filepath)

def preprocessed_data(words_DF,sentences_DF,acustic_feature_DF,dt_features,Trial_ID):
    zeropad_lengh=10
    words_DF_trial = words_DF[words_DF.trial_id == Trial_ID]
    # words_DF_trial = words_DF[:100]
    ''''''

    snt_duration = words_DF_trial.duration.sum()
    snt_onset = sentences_DF[sentences_DF.trial_id == Trial_ID].onset.to_numpy().squeeze()
    snt_onset_indx = int(np.floor(snt_onset // dt_features))
    snt_duration_len = int(np.floor(snt_duration // dt_features))
    ac_DF_trial = acustic_feature_DF.iloc[snt_onset_indx:snt_onset_indx + snt_duration_len]

    wrd_event = np.zeros(ac_DF_trial.shape[0], )
    wrd_duration = np.zeros(ac_DF_trial.shape[0], )
    delay_btwn_wrd = np.zeros(ac_DF_trial.shape[0], )
    # neural_feature = np.zeros((ac_DF_trial.shape[0], df_neural.shape[1]))
    wrd_type = []
    last_wrd_time = 0
    for ii in range(wrd_event.shape[0]):
        time_int = [snt_onset + (ii) * dt_features, snt_onset + ((ii)+ 1) * dt_features]
        # neural_feature[ii, :] = df_neural.loc[np.floor(time_int[0] * neural_smaping_rate):np.floor(
        #     time_int[1] * neural_smaping_rate)].mean().to_numpy()
        identified_wrd = words_DF_trial.word.loc[
            (words_DF_trial.onset <= time_int[1]) & (words_DF_trial.onset >= time_int[0])]
        identified_wrd_duration = words_DF_trial.duration.loc[
            (words_DF_trial.onset <= time_int[1]) & (words_DF_trial.onset >= time_int[0])]
        if (len(identified_wrd) == 0) or (identified_wrd.to_list()[0] == 'sp'):
            # No word for the time interval
            wrd_event[ii] = 0
            wrd_duration[ii] = 0
            wrd_type.append('NAN')

        elif len(identified_wrd) == 1:
            # One word for the time interval
            wrd_event[ii] = 1
            wrd_duration[ii] = identified_wrd_duration // dt_features
            delay_btwn_wrd[ii] = ii - last_wrd_time
            last_wrd_time = ii
            wrd_type.append(identified_wrd.to_list()[0])
        else:
            # select the longest word for the time interval
            selected = np.where(identified_wrd_duration == identified_wrd_duration.max())[0]
            wrd_event[ii] = 2
            wrd_duration[ii] = identified_wrd_duration.to_numpy()[selected] // dt_features
            wrd_type.append(identified_wrd.to_list()[selected[0]])
            delay_btwn_wrd[ii] = ii - last_wrd_time
            last_wrd_time = ii
    ac_DF_trial['wrd_event'] = wrd_event
    ac_DF_trial['wrd'] = wrd_type
    ac_DF_trial['wrd_duration'] = wrd_duration
    ac_DF_trial['diff_from_last_wrd'] = delay_btwn_wrd
    ac_DF_trial.reset_index()
    ''' build the CIF for events'''
    numbe_events = int(ac_DF_trial.wrd_event.sum())
    history_events = np.zeros((ac_DF_trial.shape[0],numbe_events))
    history_diff=  np.zeros((ac_DF_trial.shape[0],))
    lambda_diff=  np.zeros((ac_DF_trial.shape[0],))
    events_times = np.zeros((1,numbe_events))
    # print(history_diff.shape)
    qq =0
    t_past = 0
    for t in range(ac_DF_trial.shape[0]):
        # print(ac_DF_trial.head())
        if qq <numbe_events:
            history_diff[t] =( t - t_past)
            lambda_diff[t] = np.exp((t - t_past)/20 - ac_DF_trial.shape[0] )/ac_DF_trial.shape[0]
        if ac_DF_trial.wrd_event.iloc[t]>0:
            events_times[0,qq]=t
            # history_events[:t,qq] = 1*(np.arange( t)/ac_DF_trial.shape[0])
            qq=qq+1
            t_past = t


    history_diff/=ac_DF_trial.shape[0]

    ac_DF_trial['history_diff']= history_diff
    ac_DF_trial['time_ind'] = np.arange(ac_DF_trial.shape[0])/ac_DF_trial.shape[0]

    ac_DF_trial[['history_events'+str(i) for i in range(numbe_events)]] = history_events
    window = signal.windows.exponential(M, tau=tau)
    # 1 * np.convolve(ac_DF_trial.wrd_event, window, 'same')
    ac_DF_trial['synth_CIF'] =lambda_diff#+history_events[:,-1].squeeze()
    ac_DF_trial['synth_CIF'] = ac_DF_trial['synth_CIF'] / ac_DF_trial['synth_CIF'].max()
    # plt.plot(ac_DF_trial['synth_CIF'])

    events_times=events_times+zeropad_lengh
    df_zer=pd.DataFrame(np.zeros((zeropad_lengh,len(ac_DF_trial.columns))),columns=ac_DF_trial.columns)
    df_zer['wrd'] = 'NAN'
    ac_DF_trial= pd.concat([df_zer,ac_DF_trial])

    ac_DF_trial=ac_DF_trial.reset_index()
    return ac_DF_trial, events_times


def get_tr_te(words_DF, sentences_DF, acustic_feature_DF,
                                                                  dt_features,Trial_IDs_tr, Trial_IDs_te, mfccs_features_len,
              Use_neural_fea):
    qq = 0
    for trials_id in Trial_IDs_tr:
        if qq == 0:
            ac_DF_trial_tr, events_times_tr = preprocessed_data(words_DF, sentences_DF, acustic_feature_DF,
                                                                  dt_features, trials_id)
        else:
            ac_DF_trial, events_times = preprocessed_data(words_DF, sentences_DF, acustic_feature_DF,
                                                            dt_features, trials_id)

            events_times_tr = np.concatenate([events_times, events_times_tr], axis=0)
            ac_DF_trial_tr = pd.concat([ac_DF_trial, ac_DF_trial_tr])
            # neural_feature_tr = np.concatenate([neural_feature, neural_feature_tr], axis=0)
        qq += 1

    qq = 0
    for trials_id in Trial_IDs_te:
        if qq == 0:
            ac_DF_trial_te, events_times_te = preprocessed_data(words_DF, sentences_DF, acustic_feature_DF,
                                                                  dt_features, trials_id)
        else:
            ac_DF_trial, events_times = preprocessed_data(words_DF, sentences_DF, acustic_feature_DF,
                                                            dt_features, trials_id)
            ac_DF_trial_te = pd.concat([ac_DF_trial, ac_DF_trial_te])
            events_times_te = np.concatenate([events_times, events_times_te], axis=0)
            # neural_feature_te = np.concatenate([neural_feature, neural_feature_te], axis=0)
        qq += 1

    ''' decoding model with GLM'''
    ac_DF_trial_tr = ac_DF_trial_tr.reset_index()
    ac_DF_trial_te = ac_DF_trial_te.reset_index()
    # select from either neural or acustic features
    if Use_neural_fea:
        pass
        # X_tr = neural_feature_tr[:, 1:]
        # X_te = neural_feature_te[:, 1:]
    else:
        featurelist=[]
        featurelist.extend(['f_' + str(i) for i in range(mfccs_features_len)])
        # featurelist.extend(['history_diff'])
        # featurelist.extend(['time_ind'])
        # featurelist.extend(['history_events'+str(i) for i in range(ac_DF_trial_tr.wrd.nunique()-1)])
        print(featurelist)
        X_tr = ac_DF_trial_tr[featurelist].to_numpy()
        X_te = ac_DF_trial_te[featurelist].to_numpy()

    X_tr = sm.add_constant(X_tr)
    X_te = sm.add_constant(X_te)
    y_tr = ac_DF_trial_tr['synth_CIF'].to_numpy()
    y_te = ac_DF_trial_te['synth_CIF'].to_numpy()

    return X_tr,y_tr,X_te,y_te,  ac_DF_trial_tr, ac_DF_trial_te, featurelist,events_times_tr, events_times_te

