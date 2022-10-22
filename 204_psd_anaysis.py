from data_utilities import *
import torch
from deep_models import get_trial_data
import pickle
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




h_k = 100
f_k=25
number_trials = 80
trials = np.arange(1,number_trials)

''' gather all features for the phonemes'''
for trial in trials:
    if trial == 1:
        XDesign_total, y_tr = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id_total = np.argmax(y_tr, axis=-1).reshape([-1,1])

    else:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id = np.argmax(y_tr, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)


import matplotlib.pyplot as plt
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
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
X= np.divide(XDesign_total-XDesign_means.transpose(), XDesign_stds.transpose())
X= X.reshape([X.shape[0],-1])
scaler = StandardScaler()
X =scaler.fit_transform(X)
y= phonemes_id_total
n_components = 2
tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
finalDf = pd.DataFrame(data = tsne_results, columns = [['pc'+ str(ii) for ii in range(1,n_components+1)]])
finalDf['target'] = y
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

from sklearn.linear_model import LogisticRegression
X = np.divide(XDesign_total-XDesign_means.transpose(), XDesign_stds.transpose())
y_onehot=  pd.get_dummies(
        finalDf['target'].to_numpy().squeeze()).to_numpy()
corr = np.zeros((y_onehot.shape[1],X.shape[-1]))
for jj in range(corr.shape[1]):
    print(jj)
    inputs=X[:,:,:,jj].mean(axis =1).reshape([X.shape[0], -1])
    clf = LogisticRegression(random_state=0).fit(inputs, y)
    y_hat = clf.predict_proba(inputs)
    corr[:,jj] = y_hat.max(axis=0)
# indx_arr = np.array([13,37,23,26,29,20,19,12,6,21,4,16,37,8,31,14,25,28,30, 9, 24, 41,  5,  18, 3, 10])
plt.figure()
sns.heatmap((corr), annot=False, cmap='Blues')
plt.title('Encoding\n\n')
plt.yticks(ticks=np.arange(len(list(phones_code_dic.keys()))), labels=list(phones_code_dic.keys()), rotation=0)
