import torch.optim as optim
import pickle
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib.pyplot as plt
from  deep_models import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_utilities import *

''''''
def plot_sample_evaluate(model, iterator, sample_size, label):
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src ,trg= batch
            src = torch.swapaxes(src, 0, 1)
            trg = torch.swapaxes(trg, 0, 1)
            output = model(src, trg, 0,False)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
        for ii in np.random.randint(1,src.shape[1],sample_size):
            fig, ax =plt.subplots( 2,1, sharex=True, sharey=True)

            ax[0].stem(trg[:,ii].detach().cpu().numpy().squeeze(), 'b')
            ax[0].set_title('true sentence sample-'+str(ii)+'-result for ' + label)
            ax[1].stem(output.detach().cpu().numpy().argmax(axis=-1)[:, ii].squeeze(), 'r')
            ax[1].set_title('predicted sentence sample'+str(ii)+' result for ' + label)

def visualize_confMatrix(data,labels, title ):
    df_cm = pd.DataFrame(np.log(data+1), columns=labels, index=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=False, annot_kws={"size": 16})
    plt.title(title)
''' Constants'''
downsampling_rate=25
margin_bf=100
clustering_phonemes = True
clustering_phonemes_id = 3
device = torch.device( 'cpu')
patient_id = 'DM1007'
raw_denoised = 'raw'
datasets_add = './Datasets/'
phonemes_dataset_address = 'LM/phonemes_df_harvard_dataset.csv' # 'LM/our_phonemes_df.csv' or 'LM/phonemes_df_harvard_dataset.csv'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
saved_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
save_result_path = datasets_add + patient_id + '/Results_'+raw_denoised+'/' +'phonems_psd/'
phonemes_dic_address = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv'
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
        # phonemes_info = np.zeros((len(trials_id), window_after + margin_bf,3)) ## id, onset, duration

        if X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze().shape[0] == margin_bf+window_after:
            X_new[trial-1,:,:]=X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze()
            # phonemes_info[trial - 1, :, :] = data_list_trial[1][
            #                                      ['phoneme_id', 'phoneme_onset', 'phoneme_duration']].to_numpy()[
            #                                  speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
        else:
            aa_diff=margin_bf+window_after-X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze().shape[0]
            temp_X=np.concatenate([X[speach_onset-margin_bf:speach_onset+window_after,:].squeeze(), np.zeros((aa_diff,X.shape[1]))],axis=0)
            X_new[trial - 1, :, :]=temp_X


    else:
        if X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze().shape[0] == margin_bf + window_after:
            X_new[trial - 1, :, :] = X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
            # phonemes_info[trial - 1, :, :] = data_list_trial[1][
            #                                      ['phoneme_id', 'phoneme_onset', 'phoneme_duration']].to_numpy()[
            #                                  speach_onset - margin_bf:speach_onset + window_after, :].squeeze()
        else:
            aa_diff = margin_bf + window_after - \
                      X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze().shape[0]
            temp_X = np.concatenate(
                [X[speach_onset - margin_bf:speach_onset + window_after, :].squeeze(), np.zeros((aa_diff, X.shape[1]))],
                axis=0)
            X_new[trial - 1, :, :] = temp_X


## X_new [trial, time, chn]
''' Get phonemes'''
phones_df = pd.read_csv(
    datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-phonemes.tsv',
    sep='\t')
phones_df.phoneme =  phones_df.phoneme.replace(' ', 'NAN')
phones_df['phoneme'] = phones_df['phoneme'].str.upper()
phones_df['phoneme'] = phones_df['phoneme'].fillna('NAN')
phones_df.phoneme = apply_stress_remove([phones_df.phoneme.to_list()])[0]
phones_df.phoneme = phones_df.phoneme.replace('NA', 'NAN')
phones_df['phoneme_id'] = 0
phones_df['phoneme_id'] = phones_df['phoneme'].apply(lambda x: phones_code_dic[x])
phones_df['count_trial']=1

max_phonemes_seq=phones_df.groupby(by="trial_id",dropna=False).sum().count_trial.max()
phonemes_info=np.ones((len(trials_id),max_phonemes_seq))*phones_code_dic['NAN']
for trial in trials_id:
    temp_phonemes = phones_df.loc[phones_df.trial_id == trial].phoneme_id.to_numpy()
    if len(temp_phonemes) < max_phonemes_seq:
        temp_phonemes=np.concatenate([temp_phonemes,np.ones((max_phonemes_seq-len(temp_phonemes),))*phones_code_dic['NAN'] ])
        phonemes_info[trial-1, :] = temp_phonemes
    else:
        phonemes_info[trial-1,:] = temp_phonemes


## phonemes_info [trial, phoneme_ids],

''' GEt mni position fro ECOGs channels'''
list_ECOG_chn = data_list_trial[1].columns[data_list_trial[1].columns.str.contains("ecog")].to_list()
chn_df = pd.read_csv( datasets_add + patient_id + '/sub-'+patient_id+'_electrodes.tsv', sep='\t')
position_chn = chn_df[['mni_x', 'mni_y', 'mni_z']].loc[chn_df.name.str.contains("ecog")].to_numpy()
position_chn=np.repeat(np.expand_dims(position_chn,0),X_new.shape[0],0)

''' exclude non informative channel manually'''
non_inf_chnnel_id = np.load(saved_result_path + '/all_chn/no_inf_chnl.npy')


''' exclude non informative channels according to noise level Latane analysis'''

## TODO
''' exclude non informative channels according to task modulation'''

## TODO

X_new=np.delete(X_new,non_inf_chnnel_id, axis=-1)
position_chn=np.delete(position_chn,non_inf_chnnel_id, axis=1)

''' Preprocessing ECOGs data per channel'''
# for ii in range(X_new.shape[2]):
#     for jj in range(X_new.shape[0]):
#       X_new[jj,:,ii]=(X_new[jj,:,ii] -X_new[jj,:,ii].mean())/(X_new[jj,:,ii].std()+.5)

''' train test trial split'''
trial_train, trial_test = train_test_split(np.arange(X_new.shape[0]),test_size=.3,random_state=0,shuffle=True)
inputs_train=X_new[trial_train]
pos_train=position_chn[trial_train]
out_train=phonemes_info[trial_train]

inputs_test=X_new[trial_test]
pos_test=position_chn[trial_test]
out_test=phonemes_info[trial_test]

''' clustering the neural features indexes according to phonemes clusters  '''
if clustering_phonemes:
    clr = 'clustering_' + str(clustering_phonemes_id)
    reindexing_dic = dict(zip(phonemes_dict_df['ids'].to_list(), phonemes_dict_df[clr].to_list()))
    out_train = vec_translate(out_train, reindexing_dic)
    out_test = vec_translate(out_test, reindexing_dic)
else:
    pass
''' load large dataset_for pre_train'''

data_in_pre, data_out_pre, _ = get_phonems_data(datasets_add, clustering_id=clustering_phonemes_id,
                                                 clustering=clustering_phonemes,
                     phonemes_add= phonemes_dataset_address ,
                     dict_add = phonemes_dic_address)
data_in_pre=data_in_pre[:,:max_phonemes_seq]
data_out_pre=data_in_pre

''' reindex the target values to local indexes'''
unique_vals_y= np.unique(np.concatenate([out_train,out_test],axis=0))
labels_classes_reindex=np.array(list(phones_code_dic.keys()))[unique_vals_y.astype('int')]
out_train_reindexed = np.zeros_like(out_train)
out_test_reindexed = np.zeros_like(out_test)
for count, value in enumerate(unique_vals_y):
    for ii in range(out_train.shape[0]):
        out_train_reindexed[ii,np.where(out_train[ii,:] == value)[0]] = count
    for ii in range(out_test.shape[0]):
        out_test_reindexed[ii,np.where(out_test[ii,:] == value)[0]] = count

data_in_pre_reindexed = np.zeros_like(data_in_pre)
data_out_pre_reindexd = np.zeros_like(data_out_pre)
for count, value in enumerate(unique_vals_y):
    for ii in range(data_in_pre.shape[0]):
        data_in_pre_reindexed[ii,np.where(data_in_pre[ii,:] == value)[0]] = count
        data_out_pre_reindexd[ii,np.where(data_out_pre[ii,:] == value)[0]] = count

''' Resampling the sequences '''
indx_cent=np.arange(downsampling_rate,inputs_train.shape[1],downsampling_rate)
input_down_tr=np.zeros((inputs_train.shape[0],len(indx_cent), inputs_train.shape[2]))
out_down_tr=out_train_reindexed
input_down_te=np.zeros((inputs_test.shape[0],len(indx_cent),inputs_test.shape[2]))
out_down_te=out_test_reindexed
for count,ii_indx in enumerate(indx_cent):
    input_down_tr[:, count,:] = inputs_train[:,
                              ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2,:].mean(axis=1)


    input_down_te[:, count,:] = inputs_test[:,
                              ii_indx - downsampling_rate // 2:ii_indx + downsampling_rate // 2,:].mean(axis=1)

vocab_size=np.int(len(np.unique(out_down_tr)))
chn_size=len(list_ECOG_chn)-len(non_inf_chnnel_id)
time_length = len(indx_cent)


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
ENC_EMB_DIM = 100
DEC_EMB_DIM = 100
HID_DIM = 50
N_LAYERS = 2
BIDIRECTIONAL = True
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

BATCH_SIZE = 100
N_EPOCHS = 500
CLIP = 1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,BIDIRECTIONAL)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,BIDIRECTIONAL)
model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
print(f'The Encoder model has {count_parameters(enc):,} trainable parameters')
print(f'The Decoder model has {count_parameters(dec):,} trainable parameters')
print(f'The mixed model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=5e-3)



'''Pre training '''
uniques, counts = np.unique(data_out_pre_reindexd, return_counts=True)
counts=counts/counts.sum()
weights=np.zeros((vocab_size,))
weights[uniques]=1/counts
weights=weights/weights.sum()
# weights[-1]=0 # last phone is either NAN or SP which we try to ignore in our analysis

criterion = nn.CrossEntropyLoss( weight=torch.tensor(weights,dtype=torch.float,device=device),
                                 ignore_index = np.arange(vocab_size)[-1])
data_in_tr_pre,data_in_val_pre,data_out_tr_pre, data_out_val_pre= train_test_split(data_in_pre_reindexed,
                                                                                   data_out_pre_reindexd,
                                                                                   test_size=.3,
                                                                                   shuffle=True)
Dataset_phonemes_pre_train = get_dataset(data_in_tr_pre, data_out_tr_pre, device)
Dataset_phonemes_pre_val = get_dataset(data_in_val_pre, data_out_val_pre, device)
pre_train_data_loader_train = DataLoader(Dataset_phonemes_pre_train, batch_size=data_in_tr_pre.shape[0],shuffle=True)
pre_val_data_loader_valid = DataLoader(Dataset_phonemes_pre_val, batch_size=data_in_val_pre.shape[0],shuffle=True)
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_conf = train(model,
                                              pre_train_data_loader_train,
                                              optimizer,
                                              criterion,
                                              CLIP,
                                              vocab_size,
                                              decoder_pretraining=True)
    valid_loss, valid_acc, val_conf = evaluate(model,
                                               pre_val_data_loader_valid,
                                               criterion,
                                               vocab_size,
                                               decoder_pretraining=True)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'pre_model_best_valid_loss.pt')
    print(f'PRE-Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tPRE-Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t PRE-Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    print(f'\tPRE-Train Acc: {train_acc*100:.3f} |')
    print(f'\t PRE-Val. Acc: {valid_acc*100:.3f} |')



# model.load_state_dict(torch.load('pre_model_best_valid_loss.pt'))
test_loss, test_acc, test_conf = evaluate(model,
                                          pre_val_data_loader_valid,
                                          criterion, vocab_size,
                                          decoder_pretraining=True)
visualize_confMatrix(train_conf, labels_classes_reindex, 'pre-train')
visualize_confMatrix(val_conf, labels_classes_reindex, 'pre-test')
print(f'| PRETest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(f'| PRETest Acc: {test_acc*100:.3f} |')


'''Actual training'''
uniques, counts = np.unique(out_down_tr, return_counts=True)
counts=counts/counts.sum()
weights=np.zeros((vocab_size,))
weights[uniques]=1/counts
weights=weights/weights.sum()
criterion = nn.CrossEntropyLoss( weight=torch.tensor(weights,dtype=torch.float,device=device),ignore_index = np.arange(vocab_size)[-1])
Dataset_phonemes_train = get_dataset(input_down_tr, out_down_tr, device)
Dataset_phonemes_test = get_dataset(input_down_te, out_down_te, device)
train_data_loader = DataLoader(Dataset_phonemes_train, batch_size=BATCH_SIZE,shuffle=True)
test_data_loader = DataLoader(Dataset_phonemes_test, batch_size=BATCH_SIZE,shuffle=False)
best_valid_loss = float('inf')
best_valid_acc = 0
model.load_state_dict(torch.load('pre_model_best_valid_loss.pt'))
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_conf = train(model, train_data_loader, optimizer, criterion, CLIP, vocab_size)
    valid_loss, valid_acc, val_conf = evaluate(model, test_data_loader, criterion, vocab_size)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'final_model_best_val_acc.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    print(f'\tTrain Acc: {train_acc*100:.3f} |')
    print(f'\t Val. Acc: {valid_acc*100:.3f} |')

model.load_state_dict(torch.load('final_model_best_val_acc.pt'))
test_loss, test_acc, test_conf = evaluate(model, test_data_loader, criterion, vocab_size,decoder_pretraining=False)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(f'| Test Acc: {test_acc*100:.3f} |')

''' visualize confusion matrix of the result'''
visualize_confMatrix(train_conf, labels_classes_reindex, 'train')
visualize_confMatrix(val_conf, labels_classes_reindex, 'test')
''' visualize samples of prediction result for train and test'''
plot_sample_evaluate(model, train_data_loader, 3, label='train')
plot_sample_evaluate(model, test_data_loader, 3, label='test')
#
