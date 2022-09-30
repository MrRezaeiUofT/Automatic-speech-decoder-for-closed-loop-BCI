from GMM_utils import *
from DDD_utils import *
import pandas as pd
from speech_utils import *

from sklearn.linear_model import LogisticRegression
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
model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs')

for hk in range(3):
    XDesign=calDesignMatrix_V2(X,hk+1).reshape([X.shape[0],-1])
    model_LR.fit(XDesign, y)
    y_hat=model_LR.predict(XDesign)
    # print('acc=%f-percent\n'%(accuracy_score(y,y_hat)))
    y_hat_prob=model_LR.predict_proba(XDesign)

    ''' DDD filtering'''

    p_wXT = np.zeros((y_hat_prob.shape[0],y_hat_prob.shape[1]))
    pwtwt1 = get_state_transition_p_bigram(phones_code_dic,phones_NgramModel)
    # pwtwt1= pwtwt1/(pwtwt1.shape[0])
    # plt.imshow(pwtwt1)

    for ii in range(p_wXT.shape[0]):

        if ii == 0:
            p_wXT[ii, :]=y_hat_prob[0]
        else:

            p_prev= pwtwt1.dot(y_hat_prob[ii-1])
            p_prev /= p_prev.sum()
            one_step_pred = pwtwt1.dot( p_wXT[ii-1, :])
            one_step_pred /= one_step_pred.sum()
            p_wXT[ii,:]=(y_hat_prob[ii]) *one_step_pred


    y_hat_DDD=np.argmax(p_wXT, axis=-1)
    print('hk =%d----prediction_acc=%f------------DDD_acc=%f-percent\n'%(hk,accuracy_score(y,y_hat),accuracy_score(y,y_hat_DDD)))