from data_utilities import *

import seaborn as sns
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

h_k = 200
f_k=200
d_sample = 1
count_phoneme_thr = 50
epsilon = 1e-5
kernel_pca_comp = 15
config_bs = {
        'decode_length': len(np.arange(0,h_k+1+f_k,d_sample)),
    }

patient_id = 'DM1013'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
phonemes_dataset_address = 'LM/our_phonemes_df.csv'
clustering_phonemes = True
clustering_phonemes_id = 1
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
# X =np.swapaxes(XDesign_total, -3, -1).squeeze() [:,:,-1,:]
# X = np.nan_to_num(X)
# bsp_w = bspline_window(config_bs)[:,1:-1]
y_true = np.argmax(y_tri_total,axis=-1)
# plt.figure()
# plt.plot(bsp_w[:,1:-1])
# plt.savefig(save_result_path+'spline_basis.png')
# plt.savefig(save_result_path+'spline_basis.svg',  format='svg')

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

''' reindex the target values to local indexes'''
from sklearn.metrics import classification_report
unique_vals_y= np.unique(y_true)
y_reindexed = np.zeros_like(y_true)
for count, value in enumerate(unique_vals_y):
    y_reindexed[np.where(y_true == value)[0]] = count
y_true=y_reindexed
corr = np.zeros((np.unique(y_true).shape[0],X.shape[-2],3))
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

for jj in range(corr.shape[1]):
    if ~np.isin(jj, non_inf_chnnel_id):
        print(jj)

        inputs = X[:,jj,:].squeeze()


        Kernel_pca = KernelPCA(n_components=kernel_pca_comp, kernel="rbf")
        inputs = Kernel_pca.fit_transform(inputs)


        clf = LogisticRegression( solver='lbfgs',class_weight='balanced', random_state=0).fit(inputs, y_true)
        y_hat_p = clf.predict_proba(inputs)
        y_hat=clf.predict(inputs)
        temp=np.float64(confusion_matrix(y_true, y_hat))

        temp_max=np.max(y_hat_p, axis=0)
        temp_max[temp_max<= 1/corr.shape[0]] = 0
        corr[:,jj,0] = temp.diagonal()/temp.sum(axis=1)
        corr[:, jj,1] = temp_max
        corr[:, jj, 2] = np.max(np.abs(clf.coef_), axis=1).squeeze()
    else:
        pass

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

""" saving for visualization with Surf Ice"""
for i_phonemes in range(corr.shape[0]):
    vis_df= pd.DataFrame()
    vis_df['x'] = chn_df['mni_x']
    vis_df['y'] = chn_df['mni_y']
    vis_df['z'] = chn_df['mni_z']
    vis_df['color'] = corr[i_phonemes, :, 0]
    vis_df['color'][vis_df['color'] < 1 / corr.shape[0]] = np.nan
    vis_df['size'] = corr[i_phonemes,:,0]
    vis_df['size'][vis_df['size'] < 1/corr.shape[0]] = np.nan
    vis_df.to_csv(save_result_path + '/vis_' + labels_ytick[i_phonemes] + '.txt', sep='\t', index=False)
    vis_df.to_csv(save_result_path+'/vis_'+labels_ytick[i_phonemes]+'.node', sep='\t', index=False)
    if i_phonemes==0:
        vis_df_all = vis_df.copy()
        vis_df_all['color'] = i_phonemes+1
    else:
        vis_df['color'] = i_phonemes + 1
        vis_df_all=vis_df_all.append(vis_df, ignore_index=True)


vis_df_all.to_csv(save_result_path + '/vis_all.txt', sep='\t', index=False)
vis_df_all.to_csv(save_result_path+'/vis_all.node', sep='\t', index=False)

""" prediction result save visualization"""


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

''' Record weights of phonemes encoding for each channel'''
np.save(save_result_path+'channel_encoding_weights.npy', corr)
'''find the most representetive channels and use it for psd analysis'''
y_true_uniques=np.unique(y_true)
for phoneme_id in range(labels_ytick.shape[0]):
        max_encoding=corr[phoneme_id, :, 0].max()
        if max_encoding > 1/labels_ytick.shape[0]:

            chan_id = np.where(corr[phoneme_id, :, 0] == max_encoding)[0][0]
            # for chan_id in range(corr.shape[1]):
            if True:
                phone_ins=np.where(y_true == y_true_uniques[phoneme_id])[0]
                all_sign= X[phone_ins,chan_id, : ].squeeze()

                # all_sign = (all_sign -np.nanmin(all_sign,axis=0)) / (np.nanmax(all_sign,axis=0) - np.nanmin(all_sign,axis=0))
                mean_sig = all_sign.mean(axis=0)
                std_sig = all_sign.std(axis=0)
                plt.figure()
                xs_id=np.arange(-h_k,f_k+1, d_sample)
                plt.fill_between(xs_id, (mean_sig - 1/np.sqrt(len(phone_ins)) * np.sqrt(std_sig)).squeeze(),
                                 (mean_sig + 1/np.sqrt(len(phone_ins)) * np.sqrt(std_sig)).squeeze(),
                                 color='k',
                                 label='HI-DGD 95%', alpha=.5)
                plt.plot(xs_id, mean_sig, 'k', label='mean')
                # plt.plot(xs_id, all_sign[np.random.randint(0,all_sign.shape[0],5),:].transpose())
                plt.axvline(x=xs_id[h_k//d_sample], color='r', label='onset')
                plt.title('Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'-'+ patient_id+'-maxEncoding-'+str(max_encoding))
                # plt.legend()

                plt.savefig(save_result_path + 'Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.png')
                plt.savefig(save_result_path + 'Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.svg', format='svg')
                plt.close()

''' Difference for all channels'''
for phoneme_id in range(labels_ytick.shape[0]):
        fig, ax =plt.subplots(labels_ytick.shape[0]//2+1, 2,figsize=(10,16), sharex='col')
        max_encoding=corr[phoneme_id, :, 0].max()
        if max_encoding > 1/labels_ytick.shape[0]:

            chan_id = np.where(corr[phoneme_id, :, 0] == max_encoding)[0][0]
            phone_ins = np.where(y_true == y_true_uniques[phoneme_id])[0]
            all_sign_org = X[phone_ins, chan_id, :].squeeze()
            mean_sig_org = all_sign_org.mean(axis=0)
            std_sig_org = all_sign_org.std(axis=0)

            for phoneme_id_new in range(labels_ytick.shape[0]):
                phone_ins_new=np.where(y_true == y_true_uniques[phoneme_id_new])[0]
                all_sign= X[phone_ins_new,chan_id, : ].squeeze()
                mean_sig = all_sign.mean(axis=0)
                std_sig = all_sign.std(axis=0)
                xs_id=np.arange(-h_k,f_k+1, d_sample)
                ax[phoneme_id_new//2,phoneme_id_new%2].axvline(x=xs_id[h_k // d_sample], color='r', label='onset')
                if phoneme_id_new != phoneme_id:
                    ax[phoneme_id_new // 2, phoneme_id_new % 2].plot(xs_id, mean_sig, 'k', alpha=.2)
                    ax[phoneme_id_new // 2, phoneme_id_new % 2].fill_between(xs_id, (mean_sig - 1 / np.sqrt(len(phone_ins_new)) * np.sqrt(std_sig)).squeeze(),
                                     (mean_sig + 1 / np.sqrt(len(phone_ins_new)) * np.sqrt(std_sig)).squeeze(),
                                     color='k',
                                     label='HI-DGD 95%', alpha=.5)

                ax[phoneme_id_new // 2, phoneme_id_new % 2].fill_between(xs_id, (mean_sig_org - 1 / np.sqrt(len(phone_ins)) * np.sqrt(std_sig_org)).squeeze(),
                                     (mean_sig_org + 1 / np.sqrt(len(phone_ins)) * np.sqrt(std_sig_org)).squeeze(),
                                     color='b',
                                     label='HI-DGD 95%', alpha=.5)
                ax[phoneme_id_new // 2, phoneme_id_new % 2].plot(xs_id, mean_sig_org, 'b', label='mean')
                ax[phoneme_id_new // 2, phoneme_id_new % 2].set_title(labels_ytick[phoneme_id_new])




        plt.savefig(save_result_path + 'Cross-Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.png')
        plt.savefig(save_result_path + 'Cross-Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.svg', format='svg')
        plt.close()

''' Difference for all channels PCA'''
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, KernelPCA
# for phoneme_id in range(labels_ytick.shape[0]):
#         max_encoding=corr[phoneme_id, :, 0].max()
#         if max_encoding > 1/labels_ytick.shape[0]:
#
#             chan_id = np.where(corr[phoneme_id, :, 0] == max_encoding)[0][0]
#             X_embedded_ch = KernelPCA(n_components=10, kernel='linear').fit_transform(X[:,chan_id,:])
#             plt.figure()
#             for phoneme_id_new in range(labels_ytick.shape[0]):
#                 phone_ins=np.where(y_true == y_true_uniques[phoneme_id_new])[0]
#                 all_sign= X_embedded_ch[phone_ins, : ].squeeze()
#
#                 if phoneme_id_new != phoneme_id:
#                     plt.scatter(all_sign[:,0].squeeze(), all_sign[:,1].squeeze(), marker='o', c='k', alpha=.5)
#                 else:
#                     plt.scatter(all_sign[:,0].squeeze(), all_sign[:,1].squeeze(), marker='*', c='b', label='mean')
#
#
#
#             plt.title('TSNE-Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'-'+ patient_id+'-maxEncoding-'+str(max_encoding))
#             # plt.legend()
#
#             plt.savefig(save_result_path + 'TSNE-Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.png')
#             plt.savefig(save_result_path + 'TSNE-Encoding-chn-'+list_ECOG_chn[chan_id]+'-'+ labels_ytick[phoneme_id]+'.svg', format='svg')
#             plt.close()


''' visualize all ECOGS psd for all phonemes'''
# for chan_id in range(corr.shape[1]):
#     fig, ax = plt.subplots(7, 2, figsize=(10, 16), sharex='col')
#     for phoneme_id_new in range(labels_ytick.shape[0]):
#
#         all_sign = X[np.where(y_true == y_true_uniques[phoneme_id_new])[0], chan_id, :].squeeze()
#         mean_sig = all_sign.mean(axis=0)
#         std_sig = all_sign.std(axis=0)
#         xs_id = np.arange(-h_k, f_k + 1, d_sample)
#         ax[phoneme_id_new // 2, phoneme_id_new % 2].axvline(x=xs_id[h_k // d_sample], color='r', label='onset')
#
#         ax[phoneme_id_new // 2, phoneme_id_new % 2].fill_between(xs_id, (
#                             mean_sig - 1 / np.sqrt(mean_sig.shape[0]) * np.sqrt(std_sig)).squeeze(),
#                                                                          (mean_sig + 1 / np.sqrt(
#                                                                              mean_sig.shape[0]) * np.sqrt(
#                                                                              std_sig)).squeeze(),
#                                                                          color='b',
#                                                                          label='HI-DGD 95%', alpha=.5)
#         ax[phoneme_id_new // 2, phoneme_id_new % 2].plot(xs_id, mean_sig, 'b', label='mean')
#         ax[phoneme_id_new // 2, phoneme_id_new % 2].set_title(labels_ytick[phoneme_id_new]+'acc='+str(np.round(100* corr[phoneme_id_new,chan_id,0])))
#
#     plt.savefig(save_result_path+'/all_chn/' + 'Cross-Encoding-chn-' + list_ECOG_chn[chan_id] + '-' + labels_ytick[phoneme_id] + '.png')
#     plt.savefig(save_result_path+'/all_chn/' + 'Cross-Encoding-chn-' + list_ECOG_chn[chan_id] + '-' + labels_ytick[phoneme_id] + '.svg',
#                     format='svg')
#     plt.close()