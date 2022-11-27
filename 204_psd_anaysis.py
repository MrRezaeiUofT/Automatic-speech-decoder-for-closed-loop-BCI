from data_utilities import *
import torch
from deep_models import get_trial_data
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

h_k = 120
f_k=25
number_trials = 80
n_components = 2
epsilon = 1e-5
trials = np.arange(1,number_trials)
patient_id = 'DM1012'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/' +'phonems_psd/'
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes = language_data
keys= list(phones_code_dic.keys())
sp_id = phones_code_dic['sp']
# if 'NAN' in phones_code_dic:
#     pass
# else:
#     phones_code_dic.update({'NAN': len(phones_code_dic)})
nan_id = phones_code_dic['NAN']


''' gather all features for the phonemes'''
for trial in trials:
    if trial == 1:
        XDesign_total, y_tr_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id_total = np.argmax(y_tr_total, axis=-1).reshape([-1,1])


    else:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id = np.argmax(y_tr, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        y_tr_total = np.concatenate([y_tr_total, y_tr], axis=0)


''' visualize mean and std features'''
# f, axes = plt.subplots(3, 3, figsize=(18,18), sharey=True)
# for ph_id in np.unique(phonemes_id_total):
#     index_similar = np.where(phonemes_id_total == ph_id)[0]
#     XDesign_means = np.nanmean(XDesign_total[index_similar, :, :, :], axis=0).squeeze().transpose()
#     XDesign_stds= np.std(XDesign_total[index_similar, :, :, :], axis=0).squeeze().transpose()
#     XDesign_stds [XDesign_stds == 0] = np.nan
    # axes[0, 0].matshow(XDesign_means[:, 0, :])
    # axes[0, 0].set_xlabel('time-step')
    # axes[0, 0].set_ylabel('ECoGs-ch')
    # axes[0, 0].set_title('gamma_1' + 'ph-' + str(list(phones_code_dic.keys())[ph_id]))
    # axes[0, 1].matshow(XDesign_means[:, 1, :])
    # axes[0, 1].set_title('gamma_2')
    # axes[0, 2].matshow(XDesign_means[:, 2, :])
    # axes[0, 2].set_title('gamma_3')
    #
    # axes[1, 0].matshow(XDesign_stds[:, 0, :])
    # axes[1, 1].matshow(XDesign_stds[:, 1, :])
    # axes[1, 2].matshow(XDesign_stds[:, 2, :])
    #
    # SNRs = np.log(np.divide(XDesign_means[:, :, :]**2, XDesign_stds[:, :, :]**2))
    # # SNRs[XDesign_stds < 0.01] = 0
    # axes[2, 0].matshow((SNRs[:,0,:]))
    # axes[2, 1].matshow((SNRs[:,1,:]))
    # axes[2, 2].matshow((SNRs[:,2,:]))
    # plt.savefig(save_result_path+'psd-Hk=' + str(h_k)+ 'ph-' + str(list(phones_code_dic.keys())[ph_id]) + '.png')


''' TNSE visualization'''

# X = np.divide(XDesign_total-XDesign_means.transpose(), XDesign_stds.transpose())[:,:,-1,:]
# X = X.reshape([X.shape[0], -1])
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(X)
# finalDf = pd.DataFrame(data = tsne_results, columns = [['pc'+ str(ii) for ii in range(1,n_components+1)]])
# finalDf['target'] = phonemes_id_total
# f, axes = plt.subplots(7, 6, figsize=(32, 32), sharey=True, sharex=True)
# for ii in range(7):
#     for jj in range(6):
#         phoneme_id = ii*(6)+jj
#         df = finalDf.loc[np.where(finalDf.target == phoneme_id)[0]]
#         axes[ii, jj].scatter(df.pc1, df.pc2)
#         axes[ii, jj].set_title(str(list(phones_code_dic.keys())[phoneme_id]))
#         axes[ii, jj].set_xlabel('pc1')
#         axes[ii, jj].set_ylabel('pc2')
# plt.savefig(save_result_path+'tsne-phonemes.png')
''' re assign phonemes id with deleting 'NAN' and 'sp' '''
for ii in range(len(keys)):
    if (ii< sp_id) and (ii< nan_id):
        pass
    elif (ii< sp_id) and (ii> nan_id):
        phones_code_dic[keys[ii]] -=1
    elif (ii > sp_id) and (ii < nan_id):
        phones_code_dic[keys[ii]] -= 1
    elif (ii > sp_id) and (ii > nan_id):
        phones_code_dic[keys[ii]] -= 2
    elif (ii == sp_id):
        del phones_code_dic[keys[ii]]
    elif  (ii == nan_id):
        del phones_code_dic[keys[ii]]
# pwtwt1_new = np.delete(pwtwt1, (sp_id), axis=0)
# pwtwt1_new = np.delete(pwtwt1_new, (nan_id), axis=0)
# pwtwt1_new = np.delete(pwtwt1_new, (sp_id), axis=1)
# pwtwt1_new = np.delete(pwtwt1_new, (nan_id), axis=1)
# pwtwt1_new=pwtwt1_new/(pwtwt1_new.sum(axis=1))
''' define baselines'''
config_bs = {
        'decode_length': h_k+1+f_k,
        'bsp_degree':10,
    }
bsp_w = bspline_window(config_bs)
''' simple classification'''
number_comp = 7
input_type= 'spline'
features= [-1] # high-gamma=-1, low-gamma =-3, med-gamma=-2
X = XDesign_total[:,:,features,:] ##
y_onehot=  y_tr_total
corr = np.zeros((y_onehot.shape[1]-2,X.shape[-1],2))
# count_phonemes = np.delete(count_phonemes,[sp_id,nan_id])
for jj in range(corr.shape[1]):
    print(jj)
    if input_type == 'simple':
        inputs=X[:,:,:,jj].mean(axis =1).reshape([X.shape[0], -1])
    elif input_type == 'spline':
        inputs = np.swapaxes(X, 1, -1)[:,jj,:,:].dot(bsp_w).reshape([X.shape[0], -1])
    elif input_type == 'kernel_pca':
        temp = X[:, :, :, jj].squeeze().reshape([X.shape[0], -1])
        Kernel_pca = KernelPCA(n_components=number_comp, kernel="rbf")
        inputs = Kernel_pca.fit_transform(temp)

    outputs = np.argmax(y_onehot, axis=1)
    weights = 1/(count_phonemes[outputs]+1)
    weights /=weights.max()
    clf = LogisticRegression(penalty = 'l2', solver='liblinear',class_weight='balanced', random_state=0).fit(inputs, outputs)
    y_hat = clf.predict_proba(inputs)
    # GLM_model = sm.GLM(y_onehot, inputs, family=sm.families.)
    # GLM_model_results = GLM_model.fit_regularized(L1_wt=1, refit=False)
    # y_hat = GLM_model.predict(GLM_model_results.params, inputs)
    corr[:,jj,0] = np.mean(y_hat,axis=0)
    corr[:, jj,1] = np.max(y_hat, axis=0)

''' sort LFP channels'''
chn_df = pd.read_csv( datasets_add + patient_id + '/sub-DM1008_electrodes.tsv', sep='\t')
chn_df = chn_df.sort_values(by=['HCPMMP1_label_2'])[chn_df.name.str.contains("ecog")]
indx_ch_arr = chn_df.index
''' sort phonemes channels'''
# indx_ph_arr = np.arange(corr.shape[0])
indx_ph_arr = np.array([0, 3, 5,6,7,10,11,12,14,15,17,18,19,21,23, 24,26,27,28,29,31,35, 36, 38, 1, 2,4,8,9,13,16,20,22,25,30,32,33,34,37,39]) # vows and con
# indx_ph_arr = np.array([11, 35, 21, 24, 27, 18, 17, 10, 5, 19, 3, 14, 31,6, 29, 12, 23, 26, 28, 22, 39, 4, 16, 2, 8, 9, 36, 34, 33, 30, 20, 1, 38]) # Edward cheng
plt.figure()
rearranged_cov = corr[:,indx_ch_arr]
rearranged_cov = rearranged_cov[indx_ph_arr,:]
sns.heatmap((rearranged_cov[:,:,0]), annot=False, cmap='Blues')
plt.title('Encoding-mean\n\n')
plt.yticks(ticks=np.arange(len(np.array(list(phones_code_dic.keys()))[indx_ph_arr])), labels=np.array(list(phones_code_dic.keys()))[indx_ph_arr], rotation=0)
plt.xticks(ticks=np.arange(len(chn_df.HCPMMP1_label_1.to_list())), labels=np.array(chn_df.HCPMMP1_label_1.to_list()), rotation=90)
plt.savefig(save_result_path+'predic-phonemes-mean.png')

plt.figure()
sns.heatmap((rearranged_cov[:,:,1]), annot=False, cmap='Blues')
plt.title('Encoding-max\n\n')
plt.yticks(ticks=np.arange(len(np.array(list(phones_code_dic.keys()))[indx_ph_arr])), labels=np.array(list(phones_code_dic.keys()))[indx_ph_arr], rotation=0)
plt.xticks(ticks=np.arange(len(chn_df.HCPMMP1_label_1.to_list())), labels=np.array(chn_df.HCPMMP1_label_1.to_list()), rotation=90)
plt.savefig(save_result_path+'predic-phonemes-max.png')