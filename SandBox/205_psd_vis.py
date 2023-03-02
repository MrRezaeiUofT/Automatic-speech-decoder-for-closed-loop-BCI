from data_utilities import *


import matplotlib.pyplot as plt

h_k = 100
f_k=100
count_phoneme_thr = 50
epsilon = 1e-5
config_bs = {
        'decode_length': h_k+1+f_k,
    }
kernel_pca_comp = 100
patient_id = 'DM1005'
raw_denoised = 'all_freq_raw'  ## 'raw', 'denoised', '_all_freq_raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
phonemes_dataset_address = 'LM/our_phonemes_df.csv'

trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()
phonemes_dict_df = pd.read_csv(datasets_add + phonemes_dic_address)

phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df['ids'].to_list()))

chn_df = pd.read_csv( datasets_add + patient_id + '/sub-'+patient_id+'_electrodes.tsv', sep='\t')
chn_df = chn_df.sort_values(by=['HCPMMP1_label_1'])[chn_df.name.str.contains("ecog")]
indx_ch_arr = chn_df.index
labels_xtick = chn_df.HCPMMP1_label_1.to_list()
''' gather all features for the phonemes and generate the dataset'''
for trial in [1]:
    for chn in indx_ch_arr:
      XDesign_total, y_tri_total, X_tr = get_trial_data_per_chan(data_add, trial,chn, h_k, f_k, phones_code_dic,raw_denoised, tensor_enable=False)
      first_y_tri = y_tri_total[0].reshape([1,-1])
      phonemes_id_total = np.argmax(y_tri_total, axis=-1).reshape([-1, 1]).squeeze()


      for ph_id in range(len(phonemes_id_total)):
        im2 = plt.imshow(np.log(XDesign_total[ph_id, :, :].squeeze().transpose()), cmap=plt.cm.viridis, alpha=.9,
                       interpolation='bilinear')
        plt.axvline(x=h_k, color='r', label='axvline - full height')

        plt.savefig(save_result_path + 'psd_chn_'+labels_xtick[chn]+'_ph_'+list(phones_code_dic.keys())[phonemes_id_total[ph_id]]+'.png')

        plt.savefig(save_result_path + 'psd_chn_'+labels_xtick[chn]+'_ph_'+list(phones_code_dic.keys())[phonemes_id_total[ph_id]]+'.svg', format='svg')






