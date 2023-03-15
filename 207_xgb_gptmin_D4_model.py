
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
device = torch.device( "cpu")
to_t = lambda array: torch.tensor(array, device=device)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

''' constants'''
h_k = 200
f_k=200
d_sample=1
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': len(np.arange(0,h_k+1+f_k,d_sample)),
    }
kernel_pca_comp = 1
patient_id = 'DM1007'
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

''' read channles information'''
import json
saving_add = data_add +'/trials_'+raw_denoised+'/'
f = open(saving_add + "neural_features_info.json")
psd_config = json.load(f)
list_ECOG_chn= psd_config['chnls']
''' mask non informative chnl'''
saved_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
non_inf_chnnel_id = np.load(saved_result_path + '/all_chn/no_inf_chnl.npy')

X = np.mean(np.swapaxes(XDesign_total, -3, -1).squeeze() [:,:,-2:,:],axis=2)##
X[:,non_inf_chnnel_id,:]=0
y_onehot=  y_tri_total
######################################## Prediction Process ############################################################
'''dimensional reduction for features '''
X = np.nan_to_num(X)

X_new=np.zeros((X.shape[0],X.shape[1],kernel_pca_comp))
pcas=[]
for ii_ch in range(X.shape[1]):
    if ~np.isin(ii_ch, non_inf_chnnel_id):
        Kernel_pca = KernelPCA(n_components=kernel_pca_comp, kernel="linear")
        pcas.append(Kernel_pca)
        temp_x=(X[:,ii_ch,:]-np.nanmean(X[:,ii_ch,:]))/(np.nanstd(X[:,ii_ch,:])+1e-4)
        X_new[:,ii_ch,:] = Kernel_pca.fit_transform(temp_x)
    else:
        pass
X=X_new.reshape([X_new.shape[0],X_new.shape[1]*X_new.shape[2]])

Kernel_pca_f = KernelPCA(n_components=kernel_pca_comp*1, kernel="linear")
X=Kernel_pca_f.fit_transform(X)
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
if len(phoneme_id_to_delete) !=0:
    for ii_di in range(len(phoneme_id_to_delete)):
        if ii_di == 0:
            index_to_delete = np.where(y_true == phoneme_id_to_delete[ii_di])[0]
        else:
            index_to_delete = np.concatenate([index_to_delete, np.where(y_true == phoneme_id_to_delete[ii_di])[0]], axis=0)
    y_true = np.delete(y_true,index_to_delete, axis=0)
    X =  np.delete(X,index_to_delete, axis=0)
    sent_ids_n =  np.delete(sent_ids,index_to_delete, axis=0)
else:
    sent_ids_n=sent_ids

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(y_true)
y_reindexed = np.zeros_like(y_true)
for count, value in enumerate(unique_vals_y):
    y_reindexed[np.where(y_true == value)[0]] = count


''' shuffling according sentences'''
def shuffle_sentences(X, y_reindexed, sent_ids_n, y_true, test_size):
    '''
    A function to shuffle data with keeping the sentences structure
    :param X:
    :param y_reindexed:
    :param sent_ids_n:
    :param y_true:
    :param test_size:
    :return:
    '''

    sent_ids_uniques=np.unique(sent_ids_n)
    np.random.shuffle(sent_ids_uniques)

    for count, value in enumerate(sent_ids_uniques):
        index=np.where(sent_ids_n == value )[0]

        if count ==0:
            X_new=X[index,:]
            y_reindexed_new = y_reindexed[index,:]
            sent_ids_n_new = sent_ids_n[index,:]
            y_true_new =y_true[index,:]
        else:
            X_new=np.concatenate([X_new,X[index,:]], axis=0)
            y_reindexed_new = np.concatenate([y_reindexed_new, y_reindexed[index,:]], axis=0)
            sent_ids_n_new = np.concatenate([sent_ids_n_new, sent_ids_n[index,:]], axis=0)
            y_true_new = np.concatenate([y_true_new, y_true[index,:]], axis=0)
    X_train, X_test, y_train, y_test, st_tr, st_te, y_org_tr, y_org_te, index_tr, index_te = train_test_split(X_new,
                                                                                                                      y_reindexed_new,
                                                                                                                      sent_ids_n_new,
                                                                                                                      y_true_new,
                                                                                                                      np.arange(
                                                                                                                          y_true_new.shape[
                                                                                                                              0]),
                                                                                                                      test_size=test_size,
                                                                                                                      random_state=0,
                                                                                                                      shuffle=False)

    return X_train, X_test, y_train, y_test, st_tr, st_te, y_org_tr, y_org_te, index_tr, index_te



X_train, X_test, y_train_pp, y_test_pp, st_tr, st_te, y_org_tr, y_org_te, index_tr, index_te = shuffle_sentences(X,
                                                                                                           np.expand_dims(y_reindexed, axis=1),
                                                                                                           sent_ids_n,
                                                                                                           np.expand_dims(y_true, axis=1), .3)


''' XG-boost training '''
from imblearn.under_sampling import RandomUnderSampler
oversample = SMOTE()
undersample = RandomUnderSampler()
# X_train_over, y_train_over =oversample.fit_resample(X_train,y_train)
X_train_over, y_train_over =undersample.fit_resample(X_train, y_train_pp )
xgb_classifier = xgb.XGBClassifier(n_estimators=3*kernel_pca_comp,
                                   reg_alpha=.5,
                                   max_depth=5,
                                   random_state=0) # https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_classifier.fit(X_train,y_train_pp)

predictions_xgb_te = xgb_classifier.predict(X_test)
predic_probs_xgb_te = xgb_classifier.predict_proba(X_test)
predictions_xgb_tr = xgb_classifier.predict(X_train)
predic_probs_xgb_tr = xgb_classifier.predict_proba(X_train)
print("xgboost_result_tr", accuracy_score(y_train_pp, predictions_xgb_tr))
print("xgboost_result_te", accuracy_score(y_test_pp, predictions_xgb_te))
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

''' test reindexing'''
# print("xgboost_result_tr", accuracy_score(y_org_tr, np.argmax(predic_probs_xgb_convert_back_tr,axis=1)))
# print("xgboost_result_te", accuracy_score(y_org_te, np.argmax(predic_probs_xgb_convert_back_te, axis=1)))
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
gpt_model = GPT(model_config)

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 1e-4 # the gpt_model we're using is so small that we can go a bit faster
train_config.max_iters = 10000
train_config.num_workers = 0
trainer = Trainer(train_config, gpt_model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
# now let's perform some evaluation
gpt_model.eval()
#

prediction_result = np.zeros((500,2))
p=0
for ii in np.random.randint(0,data_in.shape[0],prediction_result.shape[0]):
    input_length =  np.random.randint(1,15)
    x = torch.tensor(data_in[ii][:input_length], dtype=torch.long).to(trainer.device)
    x = x.expand(10, -1)

    y, probs = gpt_model.generate(x, max_new_tokens=1, do_sample=True, top_k=40)
    prediction_result[p, 0] = data_in[ii][input_length ]
    prediction_result[p, 1] = (y.detach().cpu()[0, -1])
    # print('-' * 80)
    p =p+1
print('accuracy= %f', accuracy_score(prediction_result[:,0],prediction_result[:,1])*100)
############################################# D4 Model ######################################################


PL0 = first_y_tri.mean(axis=0)/vocab_size
# PL0=np.ones((vocab_size,))/vocab_size
from torch import nn
class State_model(nn.Module):

    def __init__(self, max_sent_length,steps, embedding_dim, vocab_size):
        super(State_model, self).__init__()
        self.steps=steps
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embeddings = nn.Embedding(max_sent_length, embedding_dim)
        self.linear1 = nn.Linear(vocab_size * steps, embedding_dim)

        self.linear2 = nn.Linear(2*embedding_dim+vocab_size, embedding_dim)
        self.linear3 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        p, k, pp = inputs
        k_embeds = self.embeddings(k)

        x=torch.flatten(p, start_dim=1)

        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.cat((x, k_embeds,pp), axis=-1)
        out = torch.nn.functional.relu(self.linear2(x))
        out=torch.nn.functional.relu(self.linear3(out))
        probs = torch.nn.functional.softmax(out, dim=-1)
        return probs
state_model= State_model(max_sent_length=data_in.shape[1],steps=1, embedding_dim=5, vocab_size=vocab_size)



def D4_model(state_model, st,y_true,y_predic_probs, PL0, epochs,training ):

    optim = torch.optim.Adam(state_model.parameters(),
                             lr=1e-3, weight_decay=1e-6)
    PL0=torch.Tensor(PL0)
    # PL0=torch.ones((vocab_size,))/vocab_size
    uniques, counts = np.unique(y_true, return_counts=True)
    counts=counts/counts.sum()
    weights=np.zeros((vocab_size,))
    weights[uniques]=1/counts
    weights=weights/weights.sum()
    loss = nn.CrossEntropyLoss(torch.tensor(weights))
    total_loss = []
    if training:
        for ii_ep in range(epochs):
            for cc, ii_snt in enumerate(np.unique(st).astype('int')):
                sent_indexes = np.where(st == ii_snt)[0]
                y_sent = y_true[sent_indexes]
                y_probs_sent = y_predic_probs[sent_indexes]

                P_pp = y_probs_sent
                P_lang = np.zeros_like(P_pp)
                P_pp = to_t(P_pp)
                P_lang = to_t(P_lang)
                y_sent = to_t(y_sent)
                y_sent_one_hot=torch.nn.functional.one_hot(y_sent.squeeze().to(torch.int64), state_model.vocab_size).to(torch.float32)

                temp=torch.zeros((len(y_sent),state_model.steps,P_pp.shape[1]))
                P_pp_pad=torch.cat([PL0.repeat(state_model.steps,1),P_pp],axis=0)

                for ii_wrd in range(len(y_sent)):
                        temp[ii_wrd,:,:]=P_pp_pad[ii_wrd :ii_wrd+state_model.steps, :]

                optim.zero_grad()
                P_total=state_model([temp,torch.tensor(np.arange(len(y_sent)), dtype=torch.int),P_pp])

                loss_state= loss(P_total,y_sent_one_hot)

                loss_state.backward()
                optim.step()

                total_loss.append(loss_state.detach().numpy())
                if cc == 0:
                    P_all = P_total.detach().numpy()
                    y_all = y_sent_one_hot.detach().numpy()
                    P_p_all = P_pp.detach().numpy()
                    P_lang_all = P_lang.detach().numpy()
                else:

                    P_all = np.concatenate([P_all, P_total.detach().numpy()], axis=0)
                    y_all = np.concatenate([y_all, y_sent_one_hot.detach().numpy()], axis=0)
                    P_p_all = np.concatenate([P_p_all, P_pp.detach().numpy()], axis=0)
                    P_lang_all = np.concatenate([P_lang_all, P_lang.detach().numpy()], axis=0)
    else:

        for cc, ii_snt in enumerate(np.unique(st).astype('int')):
                sent_indexes = np.where(st == ii_snt)[0]
                y_sent = y_true[sent_indexes]
                y_probs_sent = y_predic_probs[sent_indexes]

                P_pp = y_probs_sent
                P_lang = np.zeros_like(P_pp)
                P_pp = to_t(P_pp)
                P_lang = to_t(P_lang)
                y_sent = to_t(y_sent)
                temp=torch.zeros((len(y_sent),state_model.steps,P_pp.shape[1]))
                P_pp_pad=torch.cat([PL0.repeat(state_model.steps,1),P_pp],axis=0)

                for ii_wrd in range(len(y_sent)):
                        temp[ii_wrd,:,:]=P_pp_pad[ii_wrd :ii_wrd+state_model.steps, :]

                with torch.no_grad():
                    P_total=state_model([temp,torch.tensor(np.arange(len(y_sent)), dtype=torch.int),P_pp])


                if cc == 0:
                    P_all = P_total.detach().numpy()
                    y_all = y_sent.detach().numpy()
                    P_p_all = P_pp.detach().numpy()
                    P_lang_all = P_lang.detach().numpy()
                else:

                    P_all = np.concatenate([P_all, P_total.detach().numpy()], axis=0)
                    y_all = np.concatenate([y_all, y_sent.detach().numpy()], axis=0)
                    P_p_all = np.concatenate([P_p_all, P_pp.detach().numpy()], axis=0)
                    P_lang_all = np.concatenate([P_lang_all, P_lang.detach().numpy()], axis=0)

    return P_all, y_all, P_p_all, P_lang_all, state_model, total_loss
P_all_D4_tr,y_all_D4_tr,P_p_all_D4_tr,P_lang_all_D4_tr, state_model, total_loss = D4_model(state_model, st_tr,
                                                                  y_org_tr,predic_probs_xgb_convert_back_tr,
                                                                  PL0 ,epochs=100,training=True)
plt.figure()
plt.plot(total_loss)
P_all_D4_te,y_all_D4_te,P_p_all_D4_te,P_lang_all_D4_te, state_model, total_loss = D4_model(state_model, st_te,
                                                                  y_org_te,predic_probs_xgb_convert_back_te,
                                                                  PL0 ,epochs=0, training=False)

######################################### Finalize Result #######
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
def correct_D4_result_gpt(model, st,y_true,y_predic_probs, PL0 ):
    steps_L = 2
    do_sample_L = True
    for cc, ii_snt in enumerate(np.unique(st).astype('int')):
        sent_indexes = np.where(st == ii_snt)[0]

        y_sent= y_true[sent_indexes]
        y_probs_sent = y_predic_probs[sent_indexes]

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
P_all_tr,y_all_tr,P_p_all_tr,P_lang_all_tr = correct_D4_result_gpt(gpt_model,
                                                                   st_tr,
                                                                   y_org_tr,
                                                                   P_all_D4_tr,
                                                                   PL0)
visul_result_shenoy_order(y_all_tr.squeeze(), P_all_tr, P_p_all_tr, P_lang_all_tr,'train')

P_all_te,y_all_te,P_p_all_te,P_lang_all_te = correct_D4_result_gpt(gpt_model,
                                                                   st_te,
                                                                   y_org_te,
                                                                   P_all_D4_te,
                                                                   PL0)
visul_result_shenoy_order(y_all_te.squeeze(), P_all_te, P_p_all_te, P_lang_all_te, 'test')
