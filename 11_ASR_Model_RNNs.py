from GMM_utils import *
from DDD_utils import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from speech_utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

N_gram=2
mfccs_features_len = 39
gmm_number_Comp=2
patient_folder='./Datasets/DM1008/'
acustic_feature_DF=pd.read_csv(patient_folder+'preprocessed_dataframe.csv')
# acustic_feature_DF.phone.hist()

''' change phoneme  ' ' to 'DDD' '''
acustic_feature_DF['phone'][acustic_feature_DF['phone'] == ' ']= 'DDD'
''' assign id to phones'''
phones_code_dic = dict(zip(acustic_feature_DF.phone.unique(), np.arange(acustic_feature_DF.phone.nunique())))
phones_code_dic_inver = {v: k for k, v in phones_code_dic.items()}
acustic_feature_DF['phone_id']=0
acustic_feature_DF['phone_id']= acustic_feature_DF['phone'].apply(lambda x: phones_code_dic[x])


''' phones N-gram model'''
phones_NgramModel=NgramModel(N_gram)
phones_NgramModel.update(sentence=listToString(acustic_feature_DF['phone'].to_list()), need_tokenize=True)
# print(phones_NgramModel.prob(('HH',),'IY1'))
# print(phones_NgramModel.map_to_probs(('HH',)))

''' logistic regression model as discriminative model'''
X=acustic_feature_DF[['f_'+str(i) for i in range(mfccs_features_len)]].to_numpy()

y=acustic_feature_DF['phone_id'].to_numpy()

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_y = onehot_encoder.fit_transform(y.reshape([-1,1]))
for hk in range(20):
    XDesign=calDesignMatrix_V2(X,hk+1)
    RNN_model= get_model(XDesign,onehot_encoded_y)
    XDesign_tr, XDesign_te,onehot_encoded_y_tr, onehot_encoded_y_te,y_tr, y_te = train_test_split(XDesign,onehot_encoded_y,y, test_size=.3)
    RNN_model.fit(XDesign_tr,onehot_encoded_y_tr,batch_size=1000, epochs=150,verbose=0)
    y_hat_prob_tr = RNN_model.predict(XDesign_tr)
    y_hat_tr = np.argmax(y_hat_prob_tr, axis=-1)

    y_hat_prob=RNN_model.predict(XDesign_te)
    y_hat = np.argmax(y_hat_prob,axis=-1)

    ''' DDD filtering'''

    p_wXT = np.zeros((y_hat_prob.shape[0],y_hat_prob.shape[1]))
    pwtwt1 = get_state_transition_p_bigram(phones_code_dic,phones_NgramModel)#.transpose()
    # pwtwt1= pwtwt1/(pwtwt1.shape[0])
    # plt.imshow(pwtwt1)

    for ii in range(p_wXT.shape[0]):

        if ii == 0:
            p_wXT[ii, :]=y_hat_prob[0]
        else:

            p_prev= pwtwt1.dot(y_hat_prob[ii-1])
            p_prev/=p_prev.sum()
            one_step_pred = pwtwt1.dot( p_wXT[ii-1, :])
            one_step_pred/=one_step_pred.sum()
            p_wXT[ii,:]=(y_hat_prob[ii]) *one_step_pred


    y_hat_DDD=np.argmax(p_wXT, axis=-1)
    print('hk =%d----prediction_acc=%f------------DDD_acc=%f-percent, train-acc=%f-\n'%(hk,accuracy_score(y_te,y_hat),
                                                                                        accuracy_score(y_te,y_hat_DDD),
                                                                                        accuracy_score(y_tr,y_hat_tr)))

matrix_show(confusion_matrix(y_te,y_hat), phones_code_dic.keys(), '4D model result-test')
matrix_show(confusion_matrix(y_te,y_hat_DDD), phones_code_dic.keys(), 'DNN model result-test')
matrix_show(confusion_matrix(y_tr,y_hat_tr), phones_code_dic.keys(), 'DNN model result-train')