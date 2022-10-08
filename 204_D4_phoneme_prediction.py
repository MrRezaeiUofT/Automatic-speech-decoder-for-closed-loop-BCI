from data_utilities import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from model_utils import get_language_components
from neural_utils import get_psd_features, calDesignMatrix_V2
patient_id = 'DM1008'
datasets_add = './Datasets/'
# Opening JSON file
with open(datasets_add + patient_id + '/' + 'Preprocessed_data/' + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)

N_gram = 2
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')

''' get language model'''
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic = get_language_components(total_data, N_gram)

''' get neural features'''
list_ECOG_chn = total_data.columns[total_data.columns.str.contains("feature")].to_list()
frequency_bands = [[2, 8], [8, 12], [12, 24]]

psd_config={
    'chnls': list_ECOG_chn,
    'FreqBands': frequency_bands,
    'sampling_freq': dataset_info['sampling_freq']//dataset_info['dt'],
    'freq_stp': 2,
    'L_cut_freq': 60,
    'H_cut_freq': 150,
    'avg_freq_bands': False,
    'smoothing': False,
    'smoothing_window_size': 50,
     }
data_list, freqs = get_psd_features(total_data, psd_config, patient_id)

''' logistic regression model as discriminative model'''

model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs')


hk = 1

for trial in range(1):
    X = np.swapaxes(data_list[trial][0], 2, 0)
    X = X.reshape([X.shape[0], -1])

    y = data_list[trial][1].phoneme_id.to_numpy()
    XDesign = calDesignMatrix_V2(X,hk+1).reshape([X.shape[0], -1])
    model_LR.fit(XDesign, y)
    y_hat=model_LR.predict(XDesign)
    # print('acc=%f-percent\n'%(accuracy_score(y,y_hat)))
    y_hat_prob = model_LR.predict_proba(XDesign)

    ''' DDD filtering'''

    p_wXT = np.zeros((y_hat_prob.shape[0],y_hat_prob.shape[1]))
    y_hat_DDD=np.argmax(p_wXT, axis=-1)

    print('trial = %d, hk =%d----prediction_acc=%f\n'%(trial, hk,accuracy_score(y,y_hat)))
