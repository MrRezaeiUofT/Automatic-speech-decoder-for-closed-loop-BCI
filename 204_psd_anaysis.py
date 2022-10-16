from data_utilities import *
import torch
from deep_models import get_trial_data
import pickle


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




h_k = 200
number_trials = 80
trials = np.arange(1,number_trials)

''' gather all features for the phonemes'''
for trial in trials:
    if trial == 1:
        XDesign_total, y_tr = get_trial_data(data_add, trial, h_k, phones_code_dic, tensor_enable=False)
        XDesign_total -= XDesign_total.mean(axis=0)
        XDesign_total /= (1+XDesign_total.std(axis=0))
        phonemes_id_total = np.argmax(y_tr, axis=-1).reshape([-1,1])

    else:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k, phones_code_dic, tensor_enable=False)
        XDesign -= XDesign.mean(axis=0)
        XDesign /= (XDesign.std(axis=0)+1)
        phonemes_id = np.argmax(y_tr, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)


import matplotlib.pyplot as plt

f, axes = plt.subplots(1, 3, figsize=(32,18) , sharey=True)
for ph_id in np.unique(phonemes_id_total):

    index_similar = np.where(phonemes_id_total == ph_id)[0]
    axes[0].imshow(np.nanmean(XDesign_total[index_similar,:,0,:],axis=0).squeeze().transpose())
    axes[0].set_xlabel('time-step')
    axes[0].set_ylabel('ECoGs-ch')
    axes[0].set_title('gamma_1')
    axes[1].imshow(np.nanmean(XDesign_total[index_similar, :, 1, :],axis=0).squeeze().transpose())
    axes[1].set_title('gamma_2')
    axes[2].imshow(np.nanmean(XDesign_total[index_similar, :, 2, :],axis=0).squeeze().transpose())
    axes[2].set_title('gamma_3')

    plt.savefig(save_result_path+'psd-Hk=' + str(h_k)+ 'ph-' + str(list(phones_code_dic.keys())[ph_id]) + '.png')

