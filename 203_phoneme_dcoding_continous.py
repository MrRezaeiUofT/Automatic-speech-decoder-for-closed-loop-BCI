
import torch.optim as optim
import pickle
import pandas as pd
import math
import time
# import matplotlib.pyplot as plt
# from data_utilities import *
from  deep_models import *
from scipy.stats import mode
from sklearn.model_selection import train_test_split
downsampling_rate=25
margin_bf=100
patient_id = 'DM1005'
raw_denoised = 'raw'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()
clustering_phonemes = True
clustering_phonemes_id = 3
phonemes_dict_df = pd.read_csv(datasets_add + phonemes_dic_address)
if clustering_phonemes:
    clr = 'clustering_' + str(clustering_phonemes_id)
    phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df[clr].to_list()))
else:
    phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(), phonemes_dict_df['ids'].to_list()))

saving_add = data_add +'/trials_'+raw_denoised+'/imgs_psd/'
''' gather all features for the phonemes and generate the dataset'''

window_after=np.floor(trials_info_df.duration.to_numpy()*1000).max().astype('int')

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
        phonemes_info = np.zeros((len(trials_id), window_after + margin_bf,3)) ## id, onset, duration

        if X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze().shape[0] == margin_bf+window_after:
            X_new[trial-1,:,:]=X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze()
            phonemes_info[trial - 1, :, :] = data_list_trial[1][
                                                 ['phoneme_id', 'phoneme_onset', 'phoneme_duration']].to_numpy()[
                                             speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
        else:
            aa_diff=margin_bf+window_after-X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze().shape[0]
            temp=np.concatenate([X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze(), np.zeros((aa_diff,X.shape[1]))],axis=0)
            X_new[trial - 1, :, :]=temp

            temp_ph = phonemes_info[trial - 1, :, :] = data_list_trial[1][['phoneme_id', 'phoneme_onset',
                                                                           'phoneme_duration']].to_numpy()[
                                                       speach_onset - margin_bf:speach_onset + window_after,
                                                       :].squeeze()
            temp_add=np.zeros((aa_diff, 3))
            temp_add[:,0]=40 ## nan phonemes
            temp_ph = np.concatenate([temp_ph, temp_add],axis=0)
            phonemes_info[trial - 1, :, :] =temp_ph
    else:
        if X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze().shape[0] == margin_bf + window_after:
            X_new[trial - 1, :, :] = X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
            phonemes_info[trial - 1, :, :] = data_list_trial[1][
                                                 ['phoneme_id', 'phoneme_onset', 'phoneme_duration']].to_numpy()[
                                             speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
        else:
            aa_diff = margin_bf + window_after - \
                      X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze().shape[0]
            temp = np.concatenate(
                [X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze(), np.zeros((aa_diff, X.shape[1]))],
                axis=0)
            X_new[trial - 1, :, :] = temp

            temp_ph =  data_list_trial[1][['phoneme_id', 'phoneme_onset',
                                                                           'phoneme_duration']].to_numpy()[
                                                       speach_onset - margin_bf:speach_onset + window_after,
                                                       :].squeeze()
            temp_add = np.zeros((aa_diff, 3))
            temp_add[:, 0] = 40  ## nan phonemes
            temp_ph = np.concatenate([temp_ph, temp_add], axis=0)
            phonemes_info[trial - 1, :, :] = temp_ph

''' GEt position channels'''
list_ECOG_chn = data_list_trial[1].columns[data_list_trial[1].columns.str.contains("ecog")].to_list()
chn_df = pd.read_csv( datasets_add + patient_id + '/sub-'+patient_id+'_electrodes.tsv', sep='\t')
position_chn = np.arange(len(list_ECOG_chn))
position_chn=np.repeat(np.expand_dims(position_chn,0),X_new.shape[0],0)
''' Mask non informative channels with zero'''

saved_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
non_inf_chnnel_id = np.load(saved_result_path + '/all_chn/no_inf_chnl.npy')
X_new[:,:,non_inf_chnnel_id]=0
X_new= np.swapaxes(X_new, -2, -1)
for ii in range(X_new.shape[1]):
    for jj in range(X_new.shape[0]):
       if ~np.isin(ii,non_inf_chnnel_id):
           X_new[jj,ii,:]=(X_new[jj,ii,:] -X_new[jj,ii,:].mean())/(X_new[jj,ii,:].std())
''' train test trial sep'''

trial_train, trial_test = train_test_split(np.arange(X_new.shape[0]),test_size=.3,random_state=0,shuffle=True)
inputs_train=X_new[trial_train]
inputs_train=np.swapaxes(inputs_train,-1,-2)

pos_train=position_chn[trial_train]
out_train=phonemes_info[trial_train][:,:,0]

inputs_test=X_new[trial_test]
inputs_test=np.swapaxes(inputs_test,-1,-2)
pos_test=position_chn[trial_test]
out_test=phonemes_info[trial_test][:,:,0]

''' clustering the neural features indexes according to phonemes clusters  '''
if clustering_phonemes:
    clr = 'clustering_' + str(clustering_phonemes_id)
    reindexing_dic = dict(zip(phonemes_dict_df['ids'].to_list(), phonemes_dict_df[clr].to_list()))
    out_train = vec_translate(out_train, reindexing_dic)
    out_test = vec_translate(out_test, reindexing_dic)
else:
    pass

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(np.concatenate([out_train,out_test],axis=0))
out_train_reindexed = np.zeros_like(out_train)
out_test_reindexed = np.zeros_like(out_test)
for count, value in enumerate(unique_vals_y):
    for ii in range(out_train.shape[0]):
        out_train_reindexed[ii,np.where(out_train[ii,:] == value)[0]] = count
    for ii in range(out_test.shape[0]):
        out_test_reindexed[ii,np.where(out_test[ii,:] == value)[0]] = count


''' Resampling'''

indx_cent=np.arange(downsampling_rate,out_test_reindexed.shape[1],downsampling_rate)
input_down_tr=np.zeros((inputs_train.shape[0],len(indx_cent), inputs_train.shape[2]))
out_down_tr=np.zeros((out_train_reindexed.shape[0],len(indx_cent)))

input_down_te=np.zeros((inputs_test.shape[0],len(indx_cent),inputs_test.shape[2]))
out_down_te=np.zeros((out_test_reindexed.shape[0],len(indx_cent)))


for count,ii_indx in enumerate(indx_cent):
    input_down_tr[:, count,:] = inputs_train[:,
                              ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2,:].mean(axis=1)
    out_down_tr[:, count] = \
    mode(out_train_reindexed[:, ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2], axis=1)[
        0].squeeze()

    input_down_te[:, count,:] = inputs_test[:,
                              ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2,:].mean(axis=1)
    out_down_te[:, count] = \
    mode(out_test_reindexed[:, ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2], axis=1)[
        0].squeeze()

vocab_size=np.int(len(np.unique(out_down_tr)))
chn_size=len(list_ECOG_chn)
time_length = len(indx_cent)
""" combine channels"""
# out_down_te=np.expand_dims(out_down_te,-1)
# out_down_te = np.repeat(out_down_te, chn_size,2)
# out_down_te = np.swapaxes(out_down_te,-1,-2)
# out_down_te=out_down_te.reshape([out_down_te.shape[0]*out_down_te.shape[1], out_down_te.shape[2]])
# input_down_te = np.swapaxes(input_down_te,-1,-2)
# input_down_te=input_down_te.reshape([input_down_te.shape[0]*input_down_te.shape[1], input_down_te.shape[2]])
# pos_test=np.expand_dims(pos_test,-1)
# pos_test = np.repeat(pos_test, time_length,2)
# pos_test = pos_test.reshape([pos_test.shape[0]*pos_test.shape[1], pos_test.shape[2]])
# input_down_te=np.expand_dims(input_down_te,2)
# input_down_te=np.concatenate([input_down_te,np.expand_dims(pos_test,2)], axis=-1)
#
# out_down_tr=np.expand_dims(out_down_tr,-1)
# out_down_tr = np.repeat(out_down_tr, chn_size,2)
# out_down_tr = np.swapaxes(out_down_tr,-1,-2)
# out_down_tr=out_down_tr.reshape([out_down_tr.shape[0]*out_down_tr.shape[1], out_down_tr.shape[2]])
# input_down_tr = np.swapaxes(input_down_tr,-1,-2)
# input_down_tr=input_down_tr.reshape([input_down_tr.shape[0]*input_down_tr.shape[1], input_down_tr.shape[2]])
# pos_train=np.expand_dims(pos_train,-1)
# pos_train = np.repeat(pos_train, time_length,2)
# pos_train = pos_train.reshape([pos_train.shape[0]*pos_train.shape[1], pos_train.shape[2]])
# input_down_tr=np.expand_dims(input_down_tr,2)
# input_down_tr=np.concatenate([input_down_tr,np.expand_dims(pos_train,2)], axis=-1)

''' visualize data'''
# import matplotlib.pyplot as plt
# select_trial=8
# select_chn=55
# fig, ax =plt.subplots( 4,1, sharex=False)
#
# ax[0].plot(input_down_tr[select_trial,:,select_chn], 'b')
# ax[1].plot(inputs_train[select_trial,:,select_chn], 'k')
# ax[2].stem(out_down_tr[select_trial], 'b')
# ax[3].stem(out_train[select_trial], 'k')

''' build model'''

INPUT_DIM = chn_size
OUTPUT_DIM = vocab_size
ENC_EMB_DIM = 20
DEC_EMB_DIM = 20
HID_DIM = 200
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = chn_size
N_EPOCHS = 100
CLIP = 100

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-2)
''' balance weights'''
uniques,counts=np.unique(out_down_tr.astype('int'),return_counts=True)
counts=counts/counts.sum()
weights=np.zeros((vocab_size,))
weights[uniques]=1/(counts)
weights[-1]=0## weight nan class equal to zero
weights=weights/weights.sum()
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device)) ## I should make it automatic
criterion = nn.CrossEntropyLoss(ignore_index = 40) ## I should make it automatic

Dataset_phonemes_train = get_dataset(input_down_tr, out_down_tr, device)
Dataset_phonemes_test = get_dataset(input_down_te, out_down_te, device)
from torch.utils.data import DataLoader
train_data_loader = DataLoader(Dataset_phonemes_train, batch_size=BATCH_SIZE,shuffle=True)
test_data_loader = DataLoader(Dataset_phonemes_test, batch_size=BATCH_SIZE,shuffle=False)
best_valid_loss = float('inf')
best_valid_acc = 0
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_conf = train(model, train_data_loader, optimizer, criterion, CLIP, vocab_size)
    valid_loss, valid_acc, val_conf = evaluate(model, test_data_loader, criterion, vocab_size)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_acc < best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    print(f'\tTrain Acc: {train_acc*100:.3f} |')
    print(f'\t Val. Acc: {valid_acc*100:.3f} |')


model.load_state_dict(torch.load('tut1-model.pt'))
test_loss, test_acc, test_conf = evaluate(model, test_data_loader, criterion, vocab_size)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(f'| Test Acc: {test_acc*100:.3f} |')

''' visualize confusion matrix'''
import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
# plt.figure
# disp=ConfusionMatrixDisplay((train_conf), display_labels=np.arange(vocab_size))
# disp.plot()
#
# plt.figure
# disp=ConfusionMatrixDisplay((val_conf), display_labels=np.arange(vocab_size))
# disp.plot()

def plot_sample_evaluate(model, iterator, sample_size, vocab_size):
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src ,trg= batch
            src = torch.swapaxes(src, 0, 1)
            trg = torch.swapaxes(trg, 0, 1)
            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
        for ii in np.random.randint(1,src.shape[1],sample_size):
            fig, ax =plt.subplots( 2,1, sharex='col')

            ax[0].stem(trg[:,ii].detach().cpu().numpy().squeeze(), 'b')
            ax[1].stem(output.detach().cpu().numpy().argmax(axis=-1)[:, ii].squeeze(), 'r')

plot_sample_evaluate(model, train_data_loader, 3, vocab_size)

