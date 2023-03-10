from data_utilities import *
from neural_utils import *

patient_id = 'DM1005'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)
total_data = pd.read_csv(data_add + 'prepro_phoneme_neural_total_v1_'+raw_denoised+'.csv')


list_ECOG_chn = total_data.columns[total_data.columns.str.contains("ecog")].to_list()
ECogs_df=total_data[list_ECOG_chn]


margin_length=500
saving_add = data_add +'/trials_'+raw_denoised+'/imgs_raw/'
for trial_id in total_data.trial_id.unique():
    indx_df=ECogs_df.loc[
                (total_data.trial_id == trial_id) | (total_data.baseline_flag == trial_id)].index
    indx_baselines = ECogs_df.loc[
    (total_data.baseline_flag == trial_id)].index

    if trial_id==total_data.trial_id.unique()[0]:
        af_indx = np.arange(indx_df[-1] + 1, indx_df[-1] + 1 + margin_length, 1)
        total_index = np.concatenate([ indx_df, af_indx], axis=0)

    elif trial_id == total_data.trial_id.unique()[-1]:
            bf_indx = np.arange(indx_baselines[0] - margin_length, indx_baselines[0], 1)
            total_index = np.concatenate([bf_indx,indx_df], axis=0)
    else:

        bf_indx = np.arange(indx_baselines[0] - margin_length, indx_baselines[0], 1)
        af_indx = np.arange(indx_df[-1] + 1, indx_df[-1] + 1 + margin_length, 1)
        total_index = np.concatenate([bf_indx,indx_df, af_indx], axis=0)
    pp=0
    for chn_ecog in list_ECOG_chn:
        signal_t=ECogs_df[chn_ecog].to_numpy()[total_index]
        signal_o = ECogs_df[chn_ecog].to_numpy()[indx_df]
        baseline = ECogs_df[chn_ecog].to_numpy()[indx_baselines]
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(total_index,signal_t, 'k')
        ax1.plot(indx_df, signal_o, 'r', linewidth=2)
        ax2.plot(indx_baselines,baseline)
        plt.savefig(saving_add + 'trial_' + str(trial_id)+chn_ecog+'-chnid-'+str(pp)+'.png')
        plt.close()
        pp+=1