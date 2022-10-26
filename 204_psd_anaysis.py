from data_utilities import *
import torch
from deep_models import get_trial_data
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
h_k = 100
f_k=25
number_trials = 80
n_components = 2
epsilon = 1e-5
patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/' +'phonems_psd/'
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)

''' calculate the unbalanced weight for the classifier '''
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes = language_data
keys= list(phones_code_dic.keys())
sp_id = phones_code_dic['sp']
nan_id = phones_code_dic['NAN']
trials = np.arange(1,number_trials)

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



f, axes = plt.subplots(3, 3, figsize=(18,18), sharey=True)
for ph_id in np.unique(phonemes_id_total):
    index_similar = np.where(phonemes_id_total == ph_id)[0]
    XDesign_means = np.nanmean(XDesign_total[index_similar, :, :, :], axis=0).squeeze().transpose()
    XDesign_stds= np.std(XDesign_total[index_similar, :, :, :], axis=0).squeeze().transpose()
    XDesign_stds [XDesign_stds == 0] = np.nan
    axes[0, 0].matshow(XDesign_means[:, 0, :])
    axes[0, 0].set_xlabel('time-step')
    axes[0, 0].set_ylabel('ECoGs-ch')
    axes[0, 0].set_title('gamma_1' + 'ph-' + str(list(phones_code_dic.keys())[ph_id]))
    axes[0, 1].matshow(XDesign_means[:, 1, :])
    axes[0, 1].set_title('gamma_2')
    axes[0, 2].matshow(XDesign_means[:, 2, :])
    axes[0, 2].set_title('gamma_3')

    axes[1, 0].matshow(XDesign_stds[:, 0, :])
    axes[1, 1].matshow(XDesign_stds[:, 1, :])
    axes[1, 2].matshow(XDesign_stds[:, 2, :])

    SNRs = np.log(np.divide(XDesign_means[:, :, :]**2, XDesign_stds[:, :, :]**2))
    # SNRs[XDesign_stds < 0.01] = 0
    axes[2, 0].matshow((SNRs[:,0,:]))
    axes[2, 1].matshow((SNRs[:,1,:]))
    axes[2, 2].matshow((SNRs[:,2,:]))
    plt.savefig(save_result_path+'psd-Hk=' + str(h_k)+ 'ph-' + str(list(phones_code_dic.keys())[ph_id]) + '.png')


''' TNSE visualization'''

X = np.divide(XDesign_total-XDesign_means.transpose(), XDesign_stds.transpose())[:,:,-1,:]
X = X.reshape([X.shape[0], -1])
scaler = StandardScaler()
X = scaler.fit_transform(X)
tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
finalDf = pd.DataFrame(data = tsne_results, columns = [['pc'+ str(ii) for ii in range(1,n_components+1)]])
finalDf['target'] = phonemes_id_total
f, axes = plt.subplots(7, 6, figsize=(32, 32), sharey=True, sharex=True)
for ii in range(7):
    for jj in range(6):
        phoneme_id = ii*(6)+jj
        df = finalDf.loc[np.where(finalDf.target == phoneme_id)[0]]
        axes[ii, jj].scatter(df.pc1, df.pc2)
        axes[ii, jj].set_title(str(list(phones_code_dic.keys())[phoneme_id]))
        axes[ii, jj].set_xlabel('pc1')
        axes[ii, jj].set_ylabel('pc2')
plt.savefig(save_result_path+'tsne-phonemes.png')


''' simple classification'''

X = np.divide(XDesign_total-XDesign_means.transpose(), XDesign_stds.transpose())[:,:,-1:,:]
y_onehot=  y_tr_total
corr = np.zeros((y_onehot.shape[1]-2,X.shape[-1]))
# count_phonemes = np.delete(count_phonemes,[sp_id,nan_id])
for jj in range(corr.shape[1]):
    print(jj)
    inputs=X[:,:,:,jj].mean(axis =1).reshape([X.shape[0], -1])
    outputs = np.argmax(y_onehot, axis=1)
    weights = 1/(count_phonemes[outputs]+1)
    weights /=weights.sum()
    clf = LogisticRegression(random_state=0).fit(inputs, outputs,weights)
    y_hat = clf.predict_proba(inputs)
    corr[:,jj] = y_hat.max(axis=0)
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

indx_arr = np.arange(corr.shape[0])
# indx_arr = np.array([11, 35, 21, 24, 27, 18, 11, 17, 10, 5, 19, 3, 14, 31,6, 29, 12, 23, 26, 28, 22, 39, 4, 16, 2, 8, 9, 36, 34, 33, 30, 20, 1, 38])
plt.figure()
sns.heatmap((corr[indx_arr,:]), annot=False, cmap='Blues')
plt.title('Encoding\n\n')
plt.yticks(ticks=np.arange(len(np.array(list(phones_code_dic.keys()))[indx_arr])), labels=np.array(list(phones_code_dic.keys()))[indx_arr], rotation=0)
plt.savefig(save_result_path+'predic-phonemes.png')