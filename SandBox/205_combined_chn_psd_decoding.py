from data_utilities import *

from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
h_k = 200
f_k=200
d_sample = 1
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': len(np.arange(0,h_k+1+f_k,d_sample)),
    }
kernel_pca_comp = 150
patient_id = 'DM1005'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
phonemes_dataset_address = 'LM/our_phonemes_df.csv'
clustering_phonemes = True
clustering_phonemes_id = 1

do_sample_langmodel = True
trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()
phonemes_dict_df = pd.read_csv(datasets_add + phonemes_dic_address)
if clustering_phonemes:
    clr = 'clustering_' + str(clustering_phonemes_id)
    phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df[clr].to_list()))
else:
    phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df['ids'].to_list()))


''' gather all features for the phonemes and generate the dataset'''
for trial in trials_id[1:]:
    print(trial)
    if trial == 2:
        XDesign_total, y_tri_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic,raw_denoised, d_sample=d_sample, tensor_enable=False)
        first_y_tri = y_tri_total[0].reshape([1,-1])
        phonemes_id_total = np.argmax(y_tri_total, axis=-1).reshape([-1,1])
        sent_ids = trial*np.ones((phonemes_id_total.shape[0],1))
    else:
        XDesign, y_tri = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic,raw_denoised, d_sample=d_sample, tensor_enable=False)
        phonemes_id = np.argmax(y_tri, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        sent_ids =  np.concatenate([sent_ids,trial * np.ones((phonemes_id.shape[0], 1))], axis=0)
        first_y_tri = np.concatenate([first_y_tri,y_tri[0].reshape([1,-1])], axis=0)
        y_tri_total = np.concatenate([y_tri_total, y_tri], axis=0)



X = np.mean(np.swapaxes(XDesign_total, -3, -1).squeeze() [:,:,-2:,:],axis=2)##
X = np.nan_to_num(X)
bsp_w = bspline_window(config_bs)[:,1:-1]
y_true = np.argmax(y_tri_total,axis=-1)
''' clustering the neural features indexes according to phonemes clusters  '''
if clustering_phonemes:
    clr = 'clustering_' + str(clustering_phonemes_id)
    reindexing_dic = dict(zip(phonemes_dict_df['ids'].to_list(), phonemes_dict_df[clr].to_list()))
    y_true = vec_translate(y_true, reindexing_dic)
else:
    pass
''' delete infrequent phonemes'''
uniques_y_true, counts_y_true = np.unique(y_true, return_counts=True)
phoneme_id_to_delete = uniques_y_true[np.where(counts_y_true<count_phoneme_thr)[0]]
if len(phoneme_id_to_delete)>0:
    for ii_di in range(len(phoneme_id_to_delete)):
        if ii_di == 0:
            index_to_delete = np.where(y_true == phoneme_id_to_delete[ii_di])[0]
        else:
            index_to_delete = np.concatenate([index_to_delete, np.where(y_true == phoneme_id_to_delete[ii_di])[0]], axis=0)
    y_true = np.delete(y_true,index_to_delete, axis=0)
    X = np.delete(X,index_to_delete, axis=0)
''' Feature normalization'''


''' Get mni positions'''
chn_df = pd.read_csv( datasets_add + patient_id + '/sub-'+patient_id+'_electrodes.tsv', sep='\t')
position_mni = chn_df[chn_df.name.str.contains("ecog")][['mni_x', 'mni_y', 'mni_z']].to_numpy()
position_mni = np.arange(position_mni.shape[0])
position_mni=np.repeat(np.expand_dims(position_mni,0),X.shape[0],0)
y_true=np.repeat(np.expand_dims(y_true,1),position_mni.shape[1],1)


''' reshape X, y_tru and positions'''
X_new=X.reshape([-1,X.shape[-1]])
position_mni=position_mni.reshape([-1,1])
y_true=y_true.reshape([-1,1])
kernel_pca_comp = 10

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(y_true)
y_reindexed = np.zeros_like(y_true)
for count, value in enumerate(unique_vals_y):
    y_reindexed[np.where(y_true == value)[0]] = count
''''kernel'''
Kernel_pca = PCA(n_components=30)
X_l = Kernel_pca.fit_transform(X_new)
X_l=np.concatenate([X_new,position_mni],axis=1)
''' train and test split'''
X_train, X_test, y_train, y_test = train_test_split(X_l, y_reindexed,test_size = 0.5,
                                                                  random_state = 0, shuffle=False)

xgb_classifier = xgb.XGBClassifier(n_estimators=50,
                                   random_state=0) # https://xgboost.readthedocs.io/en/stable/parameter.html
from imblearn.under_sampling import RandomUnderSampler
# oversample = SMOTE()
undersample = RandomUnderSampler()
# X_train_over, y_train_over =oversample.fit_resample(X_train,y_train)
X_train_over, y_train_over =undersample.fit_resample(X_train, y_train )
xgb_classifier.fit(X_train_over,y_train_over)

predictions_xgb_te = xgb_classifier.predict(X_test)
predic_probs_xgb_te = xgb_classifier.predict_proba(X_test)
predictions_xgb_tr = xgb_classifier.predict(X_train)
predic_probs_xgb_tr = xgb_classifier.predict_proba(X_train)
print("xgboost_result_tr", accuracy_score(y_train, predictions_xgb_tr))
print("xgboost_result_te", accuracy_score(y_test, predictions_xgb_te))

'''visualization'''
predictions_xgb_convert_back_te= np.zeros((predictions_xgb_te.shape[0],1)).astype('int')
predictions_xgb_convert_back_tr= np.zeros((predictions_xgb_tr.shape[0],1)).astype('int')
y_test_re= np.zeros((y_test.shape[0],1)).astype('int')
y_train_re= np.zeros((y_train.shape[0],1)).astype('int')

import matplotlib.pyplot as plt
for count, value in enumerate(unique_vals_y):
    predictions_xgb_convert_back_te[np.where(predictions_xgb_te == count)] = value
    predictions_xgb_convert_back_tr[np.where(predictions_xgb_tr == count)] = value
    y_test_re[np.where(y_test == count)] = value
    y_train_re[np.where(y_train == count)] = value


def visul_result_shenoy_order(y_true, y_hat, label):

    indx_ph_arr = np.array([37, 1, 6, 4, 9, 18, 28, 24, 15, 33, 22, 11, 29, 31, 35, 34, 27, 20, 8, 36, 21, 3, 17, 32, 2, 23,
                            7, 5, 10, 19, 25, 26, 30, 13, 0, 14]) # Shenoy ppr
    uniques = np.array(np.unique(np.concatenate([y_true,y_hat], axis=0)))
    new_indx, sorting_id= sort_index(uniques, indx_ph_arr)
    conf_matrix = confusion_matrix(y_true, y_hat)
    # conf_matrix = conf_matrix[sorting_id, :]
    # conf_matrix = conf_matrix[:,  sorting_id]

    disp=ConfusionMatrixDisplay((conf_matrix), display_labels=np.array(list(phones_code_dic.keys()))[new_indx])
    disp.plot()
    plt.title(label + '-acc=' + str(accuracy_score(y_true, y_hat)))
    plt.savefig(save_result_path + 'max-mean_encoding_Mixed_Chn_'+label+'_.png')
    plt.savefig(save_result_path + 'max-mean_encoding_Mixed_Chn_'+label+'_.svg', format='svg')


visul_result_shenoy_order(y_train_re, predictions_xgb_convert_back_tr, 'train')
visul_result_shenoy_order(y_test_re, predictions_xgb_convert_back_te, 'test')