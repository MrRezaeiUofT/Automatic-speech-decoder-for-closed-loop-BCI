from data_utilities import *

import seaborn as sns
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
h_k = 100
f_k=25
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': h_k+1+f_k,
    }
kernel_pca_comp = 100
patient_id = 'DM1005'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
phonemes_dataset_address = 'LM/our_phonemes_df.csv'
clustering_phonemes = True
clustering_phonemes_id = 1
num_samples_langmodel = 30
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
for trial in trials_id:
    if trial == 1:
        XDesign_total, y_tri_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        first_y_tri = y_tri_total[0].reshape([1,-1])
        phonemes_id_total = np.argmax(y_tri_total, axis=-1).reshape([-1,1])
        sent_ids = trial*np.ones((phonemes_id_total.shape[0],1))
    else:
        XDesign, y_tri = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id = np.argmax(y_tri, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        sent_ids =  np.concatenate([sent_ids,trial * np.ones((phonemes_id.shape[0], 1))], axis=0)
        first_y_tri = np.concatenate([first_y_tri,y_tri[0].reshape([1,-1])], axis=0)
        y_tri_total = np.concatenate([y_tri_total, y_tri], axis=0)



X = np.swapaxes(XDesign_total, -3, -1).squeeze() [:,:,-1,:]##
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
for ii_di in range(len(phoneme_id_to_delete)):
    if ii_di == 0:
        index_to_delete = np.where(y_true == phoneme_id_to_delete[ii_di])[0]
    else:
        index_to_delete = np.concatenate([index_to_delete, np.where(y_true == phoneme_id_to_delete[ii_di])[0]], axis=0)
y_true = np.delete(y_true,index_to_delete, axis=0)
X = np.delete(X,index_to_delete, axis=0)

corr = np.zeros((np.unique(y_true).shape[0],X.shape[-2],3))

for jj in range(corr.shape[1]):
    inputs = X[:,jj,:].dot(bsp_w).reshape([X.shape[0], -1])
    # Kernel_pca = KernelPCA(n_components=kernel_pca_comp, kernel="rbf")
    # inputs = Kernel_pca.fit_transform(inputs)

    clf = LogisticRegression(penalty = 'l2', solver='liblinear',class_weight='balanced', random_state=0).fit(inputs, y_true)
    y_hat = clf.predict_proba(inputs)
    corr[:,jj,0] = np.mean(y_hat,axis=0)
    corr[:, jj,1] = np.max(y_hat, axis=0)
    corr[:, jj, 2] = np.max(np.abs(clf.coef_), axis=1).squeeze()

''' sort LFP channels'''
chn_df = pd.read_csv( datasets_add + patient_id + '/sub-'+patient_id+'_electrodes.tsv', sep='\t')
chn_df = chn_df.sort_values(by=['HCPMMP1_label_1'])[chn_df.name.str.contains("ecog")]
indx_ch_arr = chn_df.index
''' sort phonemes channels'''
indx_ph_arr = np.arange(corr.shape[0])
indx_ph_arr = np.array([37,1,6,4,9,18,28,24,15,33,22,11,29,31,35,34,27,20,8,36,21,3,17,32,2,23,7,5,10,19,25,26,30,13,0,14]) # Shenoy ppr
plt.figure(figsize=(20,10))
new_indx, sorting_id = sort_index(np.unique(y_true), indx_ph_arr)

rearranged_cov = corr[:,indx_ch_arr]
rearranged_cov = rearranged_cov[sorting_id,:]
sns.heatmap((rearranged_cov[:,:,0]), annot=False, cmap='Blues')
labels_ytick = np.array(list(phones_code_dic.keys()))[new_indx]
labels_xtick = chn_df.HCPMMP1_label_1.to_list()
plt.title('Pred-mean-patient'+patient_id)
plt.yticks(ticks=np.arange(len(labels_ytick)), labels=labels_ytick, rotation=0)
plt.xticks(ticks=np.arange(len(labels_xtick)), labels=np.array(labels_xtick), rotation=90)
plt.savefig(save_result_path+'predic-phonemes-mean.png')
plt.savefig(save_result_path+'predic-phonemes-mean.svg',  format='svg')

""" saving for visualization"""
for i_phonemes in range(corr.shape[0]):
    vis_df= pd.DataFrame()
    vis_df['x'] = chn_df['native_x']
    vis_df['y'] = chn_df['native_y']
    vis_df['z'] = chn_df['native_z']
    vis_df['size'] = corr[i_phonemes,:,2]
    vis_df['color'] = corr[i_phonemes,:,2]
    vis_df.to_csv(save_result_path+'/vis_'+labels_ytick[i_phonemes]+'.txt', sep='\t', index=False)

""""""


plt.figure(figsize=(20,10))
sns.heatmap((rearranged_cov[:,:,1]), annot=False, cmap='Blues')
plt.title('Pred-max-patient'+patient_id)
plt.yticks(ticks=np.arange(len(labels_ytick)), labels=labels_ytick, rotation=0)
plt.xticks(ticks=np.arange(len(labels_xtick)), labels=np.array(labels_xtick), rotation=90)
plt.savefig(save_result_path+'predic-phonemes-max.png')
plt.savefig(save_result_path+'predic-phonemes-max.svg',  format='svg')

plt.figure(figsize=(20,10))
sns.heatmap((rearranged_cov[:,:,2]/rearranged_cov[:,:,2].max()), annot=False, cmap='Blues')
plt.title('Encoding-max-patient'+patient_id)
plt.yticks(ticks=np.arange(len(labels_ytick)), labels=labels_ytick, rotation=0)
plt.xticks(ticks=np.arange(len(labels_xtick)), labels=np.array(labels_xtick), rotation=90)
plt.savefig(save_result_path+'max_encoding_weight.png')
plt.savefig(save_result_path+'max_encoding_weight.svg',  format='svg')
''' Summerized Maximum Encoding'''
data_arr = rearranged_cov[:,:,0]#/rearranged_cov[:,:,0].max()
summ_df = pd.DataFrame(data_arr, columns=labels_xtick)
summ_df['phonemes'] = labels_ytick
summ_df_melt = pd.melt(summ_df,id_vars=['phonemes'])
labels_xtick_summary = summ_df_melt[summ_df_melt['phonemes']==labels_ytick[0]].groupby(['variable']).max().index.to_list()
summary_data = np.zeros((len(labels_ytick), len(labels_xtick_summary)))
qq = 0
for phone_ii in labels_ytick:
    summ_df_melt_gb= summ_df_melt[summ_df_melt['phonemes']==phone_ii].groupby(['variable']).max()
    summary_data[qq,:] = summ_df_melt_gb.value.to_numpy()

    qq+=1

plt.figure(figsize=(8,10))
sns.heatmap(summary_data, annot=False, cmap='Blues')
plt.title('Encoding-max-mean-patient-summary'+patient_id)
plt.yticks(ticks=np.arange(len(labels_ytick)), labels=labels_ytick, rotation=0)
plt.xticks(ticks=np.arange(len(labels_xtick_summary)), labels=np.array(labels_xtick_summary), rotation=90)
plt.savefig(save_result_path+'max-mean_encoding_weight_summary.png')
plt.savefig(save_result_path+'max-mean_encoding_weight_summary.svg',  format='svg')