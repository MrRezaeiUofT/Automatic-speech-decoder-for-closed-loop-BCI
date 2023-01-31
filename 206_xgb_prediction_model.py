from data_utilities import *
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import torch
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

''' constants'''
h_k = 120
f_k=25
n_components = 2
epsilon = 1e-5
config_bs = {
        'decode_length': h_k+1+f_k,
        'bsp_degree':100,
    }
kernel_pca_comp = 10
patient_id = 'DM1012'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/' +'phonems_psd/'


''' get the data '''
trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()

''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes = language_data

''' gather all features for the phonemes and generate the dataset'''
for trial in trials_id:
    if trial == 1:
        XDesign_total, y_tr_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id_total = np.argmax(y_tr_total, axis=-1).reshape([-1,1])
    else:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id = np.argmax(y_tr, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        y_tr_total = np.concatenate([y_tr_total, y_tr], axis=0)

X = np.swapaxes(XDesign_total, -3, -1).squeeze() ##
y_onehot = y_tr_total

'''dimensional reduction for features '''
X = np.nan_to_num(X)
bsp_w = bspline_window(config_bs)[:,1:-1]
X=X.dot(bsp_w).reshape([X.shape[0], -1])

Kernel_pca = KernelPCA(n_components=kernel_pca_comp, kernel="rbf")
X = Kernel_pca.fit_transform(X)
y = np.argmax(y_onehot,axis=-1)

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(y)
y_reindexed = np.zeros_like(y)
for count, value in enumerate(unique_vals_y):
    y_reindexed[np.where(y == value)] = count

''' train and test split'''
X_train, X_test, y_train, y_test = train_test_split(X, y_reindexed, test_size = 0.2, random_state = 0)
''' XG-boost training '''

xgb_classifier = xgb.XGBClassifier(n_estimators=20,
                                   learning_rate=.01,
                                   max_features=10,
                                   max_depth=3,
                                   reg_lambda=1,
                                   reg_alpha=0,
                                   random_state=0) # https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_classifier.fit(X_train,y_train)
xgb_classifier.fit(X_train,y_train)
predictions_xgb = xgb_classifier.predict(X_test)
predic_probs_xgb = xgb_classifier.predict_proba(X_test)
predictions_xgb_train = xgb_classifier.predict(X_train)


''' visualize xgboost result test'''
indx_ph_arr = np.array([37, 1, 6, 4, 9, 18, 28, 24, 15, 33, 22, 11, 29, 31, 35, 34, 27, 20, 8, 36, 21, 3, 17, 32, 2, 23,
                        7, 5, 10, 19, 25, 26, 30, 13, 0, 14]) # Shenoy ppr
uniques_te = np.array(np.unique(np.concatenate([predictions_xgb,y_test], axis=0)))
indx_ph_arr_te =indx_ph_arr[indx_ph_arr<uniques_te.max()]
conf_matrix_test = confusion_matrix(y_test, predictions_xgb)
conf_matrix_test = conf_matrix_test[indx_ph_arr_te, :]
conf_matrix_test = conf_matrix_test[:,  indx_ph_arr_te]
disp=ConfusionMatrixDisplay(conf_matrix_test, display_labels=np.array(list(phones_code_dic.keys()))[indx_ph_arr_te])
disp.plot()
plt.title('test_result XGboost, acc='+str(100*accuracy_score(y_test, predictions_xgb))+'%')
plt.savefig(save_result_path+'test_result_XG_boost.png')
plt.savefig(save_result_path+'test_result_XG_boos.svg',  format='svg')
print("Test-Accuracy of XG-boost Model::",accuracy_score(y_test, predictions_xgb))

uniques_tr = np.unique(np.concatenate([y_train,predictions_xgb_train], axis=0))
indx_ph_arr_tr = indx_ph_arr[indx_ph_arr<uniques_tr.max()]
conf_matrix_train = confusion_matrix(y_train,predictions_xgb_train )
conf_matrix_train = conf_matrix_train[indx_ph_arr_tr, :]
conf_matrix_train = conf_matrix_train[:, indx_ph_arr_tr]
disp=ConfusionMatrixDisplay(conf_matrix_train, display_labels=np.array(list(phones_code_dic.keys()))[indx_ph_arr_tr])
disp.plot()
plt.title('train_result XGboost, acc='+str(100*accuracy_score(y_train, predictions_xgb_train))+'%')
plt.savefig(save_result_path+'train_result_XG_boost.png')
plt.savefig(save_result_path+'train_result_XG_boos.svg',  format='svg')
print("Train Accuracy of XG-boost Model::",accuracy_score(y_train, predictions_xgb_train))