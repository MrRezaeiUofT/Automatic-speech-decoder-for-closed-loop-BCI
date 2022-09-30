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