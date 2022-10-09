from data_utilities import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from neural_utils import  calDesignMatrix_V2
import pickle
patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic = language_data

''' logistic regression model as discriminative model'''

model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs')


hk = 1

for trial in range(1,2):
    file_name = data_add + 'trials/trial_' + str(trial) + '.pkl'
    with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)
    X = np.swapaxes(data_list_trial[0], 2, 0)
    X = X.reshape([X.shape[0], -1])
    y = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].to_numpy()
    XDesign = calDesignMatrix_V2(X,hk+1).reshape([X.shape[0], -1])
    model_LR.fit(XDesign, y)
    y_hat=model_LR.predict(XDesign)
    # print('acc=%f-percent\n'%(accuracy_score(y,y_hat)))
    y_hat_prob = model_LR.predict_proba(XDesign)

    ''' DDD filtering'''

    p_wXT = np.zeros((y_hat_prob.shape[0],y_hat_prob.shape[1]))
    y_hat_DDD=np.argmax(p_wXT, axis=-1)
    print('trial = %d, hk =%d----prediction_acc=%f\n'%(trial, hk,accuracy_score(y,y_hat)))
