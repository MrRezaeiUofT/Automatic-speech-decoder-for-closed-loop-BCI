from GMM_utils import *
import pandas as pd

N_gram=2
mfccs_features_len = 39
gmm_number_Comp=2
patient_folder='./Datasets/DM1008/'
acustic_feature_DF=pd.read_csv(patient_folder+'preprocessed_dataframe.csv')
acustic_feature_DF.phone.hist()

''' assign id to phones'''
phones_code_dic = dict(zip(acustic_feature_DF.phone.unique(), np.arange(acustic_feature_DF.phone.nunique())))
acustic_feature_DF['phone_id']=0
acustic_feature_DF['phone_id']= acustic_feature_DF['phone'].apply(lambda x: phones_code_dic[x])


''' phones N-gram model'''
# phones_NgramModel=NgramModel(N_gram)
# phones_NgramModel.update(sentence=listToString(acustic_feature_DF['phone'].to_list()), need_tokenize=True)
# print(phones_NgramModel.prob(('HH',),'IY1'))
# print(phones_NgramModel.map_to_probs(('HH',)))


'''GLM binomial GLM fit for events'''
# X=acustic_feature_DF[['f_'+str(i) for i in range(mfccs_features.shape[0])]].to_numpy()
# Y=phones_event
# bin_GLM = sm.GLM(Y, X, family=sm.families.Binomial())
# bin_GLM_results = bin_GLM.fit()
# # print(bin_GLM_results.summary())
# y_hat=bin_GLM_results.predict(X)
# y_hat[y_hat>=.5]= 1
# y_hat[y_hat<.5]= 0
# print('binomial GLM error_rate=%f'%(100*np.sum(np.abs(Y-y_hat))/y_hat.shape[0]))

'''acustic model by GMMs'''
# acustic_model={}
# for phone_id in acustic_feature_DF.phone.unique():
#     X = acustic_feature_DF[['f_'+str(i) for i in range(mfccs_features_len)]][acustic_feature_DF.phone == phone_id]
#     if X.shape[0]>1:
#         acustic_model[phone_id]=GaussianMixture(n_components=gmm_number_Comp).fit(X)
#     else:
#         acustic_model[phone_id]= None

'''Bayesian filtering'''

