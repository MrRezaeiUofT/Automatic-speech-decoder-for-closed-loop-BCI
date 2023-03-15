from data_utilities import *
import matplotlib.pyplot as plt
patient_id = 'DM1013'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'

trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()

saving_add = data_add +'/trials_'+raw_denoised+'/imgs_psd/'
''' gather all features for the phonemes and generate the dataset'''

margin_bf=500
window_after=1000

for trial in trials_id:
    print(trial)

    file_name = data_add + 'trials_' + raw_denoised + '/trial_' + str(trial) + '.pkl'
    with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)
    data_list_trial[1] = data_list_trial[1].reset_index()
    temp =np.where(data_list_trial[1].phoneme_onset.to_numpy() != 0)[0]
    temp_phoneme=data_list_trial[1].phoneme[temp].to_numpy()
    speach_onset=temp[np.where((temp_phoneme !='SP') & (temp_phoneme !='NAN'))][0]
    X = np.swapaxes(data_list_trial[0], 2, 0)
    X = np.mean(X[ :, -2:, :], axis=1).squeeze()  ##
    if trial==1:
        X_new = np.zeros((len(trials_id), window_after + margin_bf,X.shape[-1]))
        X_new[trial-1,:,:]=X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze()
    else:
        X_new[trial-1, :, :] = X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze()

list_ECOG_chn = data_list_trial[1].columns[data_list_trial[1].columns.str.contains("ecog")].to_list()
chnl_number=len(list_ECOG_chn)
window_inf=800
non_informative_chn_ids=[]
for ii_chd in range(chnl_number):
    mean_sig = X_new[:,:,ii_chd].mean(axis=0)
    std_sig = X_new[:,:,ii_chd].std(axis=0)
    xs_id = np.arange(-margin_bf, window_after, 1)
    plt.figure()

    plt.axvline(x=xs_id[margin_bf], color='r', label='onset')

    plt.fill_between(xs_id, (
            mean_sig - 1 / np.sqrt(chnl_number) * np.sqrt(std_sig)).squeeze(),
                                                             (mean_sig + 1 / np.sqrt(
                                                                 chnl_number) * np.sqrt(
                                                                 std_sig)).squeeze(),
                                                             color='b',
                                                             label='HI-DGD 95%', alpha=.5)
    plt.plot(xs_id, mean_sig, 'b', label='mean')

    if ((np.nanmean(mean_sig[margin_bf:])<np.nanmean(mean_sig[:margin_bf]))
            or (np.nanmax(np.abs(mean_sig))>200)
            or (np.nanmin(np.abs(mean_sig))<1)):
        non_informative_chn_ids.append(ii_chd)
        plt.savefig(
            save_result_path + '/all_chn/' + 'Speach_onset_lock_gamma-' + list_ECOG_chn[ii_chd] +'non-inf'+ '.png')
        plt.savefig(
            save_result_path + '/all_chn/' + 'Speach_onset_lock_gamma-' + list_ECOG_chn[ii_chd] + 'non-inf' + '.svg',
            format='svg')
    else:
        plt.savefig(
            save_result_path + '/all_chn/' + 'Speach_onset_lock_gamma-' + list_ECOG_chn[ii_chd] + 'inf'+'.png')
        plt.savefig(
            save_result_path + '/all_chn/' + 'Speach_onset_lock_gamma-' + list_ECOG_chn[ii_chd] + 'inf'+'.svg',
            format='svg')
    plt.close()

np.save(save_result_path + '/all_chn/''no_inf_chnl.npy',np.array(non_informative_chn_ids))