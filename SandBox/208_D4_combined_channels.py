from data_utilities import *
import matplotlib.pyplot as plt
import torch
from mingpt.utils import set_seed
set_seed(3407)
from mingpt.model import GPT
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
num_samples_langmodel = 30
from sklearn.preprocessing import MinMaxScaler, StandardScaler
h_k = 200
f_k=200
d_sample = 1
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': len(np.arange(0,h_k+1+f_k,d_sample)),
    }
kernel_pca_comp = 30
patient_id = 'DM1005'
raw_denoised = 'denoised'
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
# bsp_w = bspline_window(config_bs)[:,1:-1]
# X=X.dot(bsp_w)
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
                                                                  test_size = 0.5,
                                                                  random_state = 0, shuffle=True)


''' Get mni positions'''
chn_weights=np.load(datasets_add + patient_id+'/Results_'+raw_denoised+'/phonems_psd/channel_encoding_weights.npy')[:,:,0]
chn_weights/=np.expand_dims(chn_weights.sum(axis=1),1)
position_mni_org = np.arange(chn_weights.shape[1])
position_mni_tr=np.repeat(np.expand_dims(position_mni_org,0),X_train.shape[0],0)
y_train_new=np.repeat(np.expand_dims(y_train,1),chn_weights.shape[1],1)

''' reshape X, y_tru and positions'''
X_train_new=np.zeros((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
for ii in range(X_train.shape[2]):
    X_train_new[:,ii]=X_train[:,:,ii].reshape([-1,])

position_mni_tr=position_mni_tr.reshape([-1,1])
y_train_new=y_train_new.reshape([-1,1])
X_train_new=np.concatenate([X_train_new,position_mni_tr],axis=1)
''' XG-boost training '''

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler()
X_train_new_over, y_train_over =undersample.fit_resample(X_train_new, y_train_new )
xgb_classifier = xgb.XGBClassifier(n_estimators=50,
                                   random_state=0) # https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_classifier.fit(X_train_new_over,y_train_over)
predictions_xgb_tr = xgb_classifier.predict(X_train_new)
print("xgboost_result_tr", accuracy_score(y_train_new, predictions_xgb_tr))



position_mni_te=np.repeat(np.expand_dims(position_mni_org,0),X_test.shape[0],0)
y_test_new=np.repeat(np.expand_dims(y_test,1),chn_weights.shape[1],1)
X_test_new=np.zeros((X_test.shape[0]*X_test.shape[1],X_test.shape[2]))
for ii in range(X_test.shape[2]):
    X_test_new[:,ii]=X_test[:,:,ii].reshape([-1,])
position_mni_te=position_mni_te.reshape([-1,1])
y_test_new=y_test_new.reshape([-1,1])
X_test_new=np.concatenate([X_test_new,position_mni_te],axis=1)
predictions_xgb_te = xgb_classifier.predict(X_test_new)
print("xgboost_result_te", accuracy_score(y_test_new, predictions_xgb_te))
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

def get_D4_result(model, st,y_true,X, chn_weights, PL0, obser_model,position_mni_org, unique_vals_y, dic_length):
    steps_L = 2
    do_sample_L = True

    for cc, ii_snt in enumerate(np.unique(st).astype('int')):
        sent_indexes = np.where(st == ii_snt)[0]
        y_sent= y_true[sent_indexes]
        X_sent = X[sent_indexes]
        y_probs_sent_reduced=np.zeros((y_sent.shape[0],len(unique_vals_y)))
        for ii_ph in range(len(sent_indexes)):

            # X_in =  input_encoder.transform(X_sent[ii_ph].squeeze())
            X_in = X_sent[ii_ph].squeeze()
            X_in = np.concatenate([X_in,position_mni_org.reshape([-1,1])], axis=1)
            y_temp= obser_model.predict_proba(X_in)

            y_probs_sent_reduced[ii_ph, :]=np.nanmean(y_temp*chn_weights.transpose(),axis=0).squeeze()

        y_probs_sent = np.zeros((y_sent.shape[0],dic_length )).astype('float32')

        ''' reindex back to global indexes'''
        for count, value in enumerate(unique_vals_y):
            y_probs_sent[:, value] = y_probs_sent_reduced[:, count]


        P_pp = y_probs_sent
        P_lang = np.zeros_like(P_pp)
        P_total = np.zeros_like(P_pp)


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
P_all_tr,y_all_tr,P_p_all_tr,P_lang_all_tr = get_D4_result(model, st_tr,y_train,X_train, chn_weights, PL0,
                                                           xgb_classifier,position_mni_org,unique_vals_y,
                                                           y_tri_total.shape[1])
visul_result_shenoy_order(y_org_tr, P_all_tr, P_p_all_tr, P_lang_all_tr,'train')

P_all_te,y_all_te,P_p_all_te,P_lang_all_te = get_D4_result(model, st_te,y_test,X_test, chn_weights, PL0,
                                                      xgb_classifier,position_mni_org,unique_vals_y,y_tri_total.shape[1]
                                                           )
visul_result_shenoy_order(y_org_te, P_all_te, P_p_all_te, P_lang_all_te,'test')