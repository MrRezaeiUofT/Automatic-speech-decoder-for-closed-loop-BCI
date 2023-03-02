
from sklearn.decomposition import KernelPCA
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from data_utilities import *
from imblearn.over_sampling import SMOTE
import torch
from mingpt.utils import set_seed
set_seed(3407)
from mingpt.model import GPT
import matplotlib.pyplot as plt

''' constants'''
h_k = 500
f_k=500
d_sample=10
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': len(np.arange(0,h_k+1+f_k,d_sample)),
    }
kernel_pca_comp = 50
patient_id = 'DM1012'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
phonemes_dataset_address = 'LM/phonemes_df_harvard_dataset.csv'
clustering_phonemes = True
clustering_phonemes_id = 1
num_samples_langmodel = 30
do_sample_langmodel = True
####################################### Get Neural Data ################################################################
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
        XDesign_total, y_tri_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic,raw_denoised,d_sample=d_sample, tensor_enable=False)
        first_y_tri = y_tri_total[0].reshape([1,-1])
        phonemes_id_total = np.argmax(y_tri_total, axis=-1).reshape([-1,1])
        sent_ids = trial*np.ones((phonemes_id_total.shape[0],1))
    else:
        XDesign, y_tri = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic,raw_denoised,d_sample=d_sample, tensor_enable=False)
        phonemes_id = np.argmax(y_tri, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        sent_ids =  np.concatenate([sent_ids,trial * np.ones((phonemes_id.shape[0], 1))], axis=0)
        first_y_tri = np.concatenate([first_y_tri,y_tri[0].reshape([1,-1])], axis=0)
        y_tri_total = np.concatenate([y_tri_total, y_tri], axis=0)

X = np.swapaxes(XDesign_total, -3, -1).squeeze() [:,:,-1,:]##
y_onehot=  y_tri_total
######################################## Prediction Process ############################################################
'''dimensional reduction for features '''
X = np.nan_to_num(X)
bsp_w = bspline_window(config_bs)[:,1:-1]
X=X.dot(bsp_w).reshape([X.shape[0], -1])
Kernel_pca = KernelPCA(n_components=kernel_pca_comp, kernel="rbf")
X = Kernel_pca.fit_transform(X)
y_true = np.argmax(y_onehot, axis=-1)
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
for ii_di in range(len(phoneme_id_to_delete)):
    if ii_di == 0:
        index_to_delete = np.where(y_true == phoneme_id_to_delete[ii_di])[0]
    else:
        index_to_delete = np.concatenate([index_to_delete, np.where(y_true == phoneme_id_to_delete[ii_di])[0]], axis=0)
y_true = np.delete(y_true,index_to_delete, axis=0)
X =  np.delete(X,index_to_delete, axis=0)
sent_ids_n =  np.delete(sent_ids,index_to_delete, axis=0)

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(y_true)
y_reindexed = np.zeros_like(y_true)
for count, value in enumerate(unique_vals_y):
    y_reindexed[np.where(y_true == value)[0]] = count


''' train and test split'''
X_train, X_test, y_train, y_test, st_tr, st_te, y_org_tr, y_org_te, index_tr, index_te = train_test_split(X,
                                                                                      y_reindexed,
                                                                                      sent_ids_n,
                                                                                      y_true,np.arange(y_true.shape[0]),
                                                                  test_size = 0.3,
                                                                  random_state = 0, shuffle=False)
''' XG-boost training '''
from imblearn.under_sampling import RandomUnderSampler
oversample = SMOTE()
undersample = RandomUnderSampler()
X_train_over, y_train_over =oversample.fit_resample(X_train,y_train)
# X_train_over, y_train_over =undersample.fit_resample(X_train, y_train )
xgb_classifier = xgb.XGBClassifier(n_estimators=2*kernel_pca_comp,
                                   random_state=0) # https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_classifier.fit(X_train_over,y_train_over)

predictions_xgb_te = xgb_classifier.predict(X_test)
predic_probs_xgb_te = xgb_classifier.predict_proba(X_test)
predictions_xgb_tr = xgb_classifier.predict(X_train)
predic_probs_xgb_tr = xgb_classifier.predict_proba(X_train)
print("xgboost_result_tr", accuracy_score(y_train, predictions_xgb_tr))
print("xgboost_result_te", accuracy_score(y_test, predictions_xgb_te))
''' convert back the indexes to general indexing for all datasets '''

predictions_xgb_convert_back_te= np.zeros((predictions_xgb_te.shape[0],y_onehot.shape[1] )).astype('int')
predic_probs_xgb_convert_back_te= np.zeros_like(predictions_xgb_convert_back_te).astype('float32')

predictions_xgb_convert_back_tr= np.zeros((predictions_xgb_tr.shape[0],y_onehot.shape[1] )).astype('int')
predic_probs_xgb_convert_back_tr= np.zeros_like(predictions_xgb_convert_back_tr).astype('float32')
''' reindex back to global indexes'''
for count, value in enumerate(unique_vals_y):
    predictions_xgb_convert_back_te[np.where(predictions_xgb_te == count),value] = 1
    predic_probs_xgb_convert_back_te[:, value] = predic_probs_xgb_te[:, count]

    predictions_xgb_convert_back_tr[np.where(predictions_xgb_tr == count), value] = 1
    predic_probs_xgb_convert_back_tr[:,value] = predic_probs_xgb_tr[:,count]
########################################################################################################################
######################################### Language Model ###############################################################
data_in, data_out, vocab_size = get_phonems_data(datasets_add, clustering_id=clustering_phonemes_id,
                                                 clustering=clustering_phonemes,
                     phonemes_add= phonemes_dataset_address ,
                     dict_add = phonemes_dic_address)


train_dataset = prepare_phoneme_dataset(data_in, data_out, vocab_size=vocab_size)
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 1e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 200
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
# now let's perform some evaluation
model.eval()
#

prediction_result = np.zeros((500,2))
p=0
for ii in np.random.randint(0,data_in.shape[0],prediction_result.shape[0]):
    input_length =  np.random.randint(1,15)
    x = torch.tensor(data_in[ii][:input_length], dtype=torch.long).to(trainer.device)
    x = x.expand(10, -1)

    y, probs = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40)
    prediction_result[p, 0] = data_in[ii][input_length ]
    prediction_result[p, 1] = (y.detach().cpu()[0, -1])
    # print('-' * 80)
    p =p+1
print('accuracy= %f', accuracy_score(prediction_result[:,0],prediction_result[:,1])*100)
############################################# D4 Model ######################################################



def visul_result_shenoy_order(y_org, P_all, P_p_all, P_lang_all, label):
    print("D4_result", accuracy_score(y_org, np.argmax(P_all, axis=1)))
    print("prediction process_result", accuracy_score(y_org, np.argmax(P_p_all, axis=1)))
    print("language model_result", accuracy_score(y_org, np.argmax(P_lang_all, axis=1)))


    indx_ph_arr = np.array([37, 1, 6, 4, 9, 18, 28, 24, 15, 33, 22, 11, 29, 31, 35, 34, 27, 20, 8, 36, 21, 3, 17, 32, 2, 23,
                            7, 5, 10, 19, 25, 26, 30, 13, 0, 14]) # Shenoy ppr
    uniques = np.array(np.unique(np.concatenate([y_org, np.argmax(P_all, axis=1)], axis=0)))
    new_indx, sorting_id= sort_index(uniques, indx_ph_arr)
    conf_matrix = confusion_matrix(y_org, np.argmax(P_all, axis=1))
    conf_matrix = conf_matrix[sorting_id, :]
    conf_matrix = conf_matrix[:,  sorting_id]

    disp=ConfusionMatrixDisplay((conf_matrix), display_labels=np.array(list(phones_code_dic.keys()))[new_indx])
    disp.plot()
    plt.savefig(save_result_path + 'max-mean_encoding_weight_summary_'+label+'_.png')
    plt.savefig(save_result_path + 'max-mean_encoding_weight_summary_'+label+'_.svg', format='svg')

def get_D4_result(model, st,y_true,y_predic_probs, PL0 ):
    steps_L = 2
    do_sample_L = True
    for cc, ii_snt in enumerate(np.unique(st).astype('int')):
        sent_indexes = np.where(st == ii_snt)[0]
        y_sent= y_true[sent_indexes]
        y_probs_sent = y_predic_probs[sent_indexes]

        P_pp = y_probs_sent
        P_lang = np.zeros_like(P_pp)
        P_total = np.zeros_like(P_pp)

        # PL0=np.ones_like((len(P_pp[0]),))/len(P_pp[0])
        # x = torch.tensor(np.argmax(P_pp, axis=1), dtype=torch.long).to(trainer.device)
        # x = x.expand(num_samples_langmodel, -1)
        # y, probs = model.generate(x, max_new_tokens=steps_L, do_sample=do_sample_L, top_k=40)
        # P_lang = probs.mean(axis=0)
        # P_total = np.multiply(P_lang, P_pp)

        for ii_wrd in range(len(y_sent)):
            input_length_L = ii_wrd
            if ii_wrd == 0:
                P_total[ii_wrd,:] = np.multiply( PL0, P_pp[0,:])
                P_lang[ii_wrd,:] =  np.multiply( PL0, P_pp[0,:])

            elif ii_wrd ==1:
                x = torch.tensor(np.argmax(P_total[ii_wrd-1, :].reshape([1,-1]), axis=1), dtype=torch.long).to(trainer.device)
                x = x.expand(num_samples_langmodel, -1)
                y, probs = model.generate(x, max_new_tokens=steps_L, do_sample=do_sample_L, top_k=40)
                P_lang[ii_wrd, :] = probs.mean(axis=0)[0, :]
                P_total[ii_wrd,:] = np.multiply( P_lang[ii_wrd, :], P_pp[ii_wrd,:])
            else:

                x = torch.tensor(np.argmax(P_total[ii_wrd-input_length_L:ii_wrd, :], axis=1), dtype=torch.long).to(trainer.device)
                x = x.expand(num_samples_langmodel, -1)
                y, probs = model.generate(x, max_new_tokens=steps_L, do_sample=do_sample_L, top_k=40)
                P_lang[ii_wrd, :] = probs.mean(axis=0)[0, :]
                P_total[ii_wrd, :] = np.multiply( P_lang[ii_wrd, :], P_pp[ii_wrd, :])
        if cc == 0:
            P_all = P_total
            y_all = y_sent
            P_p_all = P_pp
            P_lang_all = P_lang
        else:

            P_all = np.concatenate([P_all,P_total],axis=0)
            y_all = np.concatenate([y_all,y_sent],axis=0)
            P_p_all = np.concatenate([P_p_all,P_pp],axis=0)
            P_lang_all = np.concatenate([P_lang_all, P_lang], axis=0)


    return P_all, y_all, P_p_all, P_lang_all

PL0 = first_y_tri.mean(axis=0)/first_y_tri.sum()
P_all,y_all,P_p_all,P_lang_all = get_D4_result(model, st_tr,y_true,predic_probs_xgb_convert_back_tr, PL0)
visul_result_shenoy_order(y_org_tr, P_all, P_p_all, P_lang_all,'train')

P_all,y_all,P_p_all,P_lang_all = get_D4_result(model, st_te,y_true,predic_probs_xgb_convert_back_te, PL0)
visul_result_shenoy_order(y_org_te, P_all, P_p_all, P_lang_all, 'test')
