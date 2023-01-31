from data_utilities import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
from deep_models import SimpleClassifier, RNN_Classifier, CNN_Classifier
import pickle


patient_id = 'DM1013'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)


''' calculate the unbalanced weight for the classifier '''
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes = language_data



''' logistic regression model as discriminative model'''
file_name = data_add + 'trials/trial_' + str(1) + '.pkl'
with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)

h_k = 100
f_k = 25
hidden_dim = 50
output_size = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].shape[-1]

''' simple model'''
# Input_size = (data_list_trial[0].shape[0]*data_list_trial[0].shape[1])*(h_k+1)
# model = SimpleClassifier(Input_size,output_size, hidden_dim )

''' RNN model'''
# Input_size = (data_list_trial[0].shape[0]*data_list_trial[0].shape[1])
# n_layers =2
# model = RNN_Classifier(Input_size, h_k, output_size, hidden_dim,n_layers)
''' CNN model'''

model = CNN_Classifier()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


LOSS_tr = []
ACC_tr = []
LOSS_te = []
ACC_te = []
number_trials = 80
trials_te = np.random.randint(1,number_trials,10)
trials_tr = np.arange(number_trials)+1
trials_tr = np.delete(trials_tr, trials_te-1)

''' training and test'''
for epochs in range(20):
    print('epoch=%d'%(epochs))
    ''' train'''
    acc_total_tr = 0
    loss_total_tr = 0
    cf_matrix_tr = np.zeros((output_size, output_size))
    acc_total_te = 0
    loss_total_te = 0
    cf_matrix_te = np.zeros((output_size, output_size))

    pp = 0
    for trial in trials_tr:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k,f_k,phones_code_dic, tensor_enable=False)
        XDesign = np.swapaxes(XDesign, 1, 2)
        XDesign = torch.tensor(XDesign, dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.float32)
        y_hat = model(XDesign)
        loss = criterion(y_hat, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_hat = y_hat.detach().numpy()
        ''' calculate acc and loss'''
        acc_total_tr += 100 * torch.sum(torch.argmax(torch.tensor(y_hat), dim=1) == torch.argmax(y_tr, dim=1)).item() / y_hat.shape[0]/len(trials_tr)
        loss_total_tr += loss.detach().numpy()/len(trials_tr)
        if pp == 0:
            y_hat_total = y_hat
            y_tr_total = y_tr.detach().numpy()
        else:
            y_hat_total = np.concatenate([y_hat_total, y_hat], axis=0)
            y_tr_total = np.concatenate([y_tr_total, y_tr.detach().numpy()], axis=0)
        pp +=1
    ''' gather acc, and loss for all trials'''
    cf_matrix_tr = confusion_matrix(np.argmax(y_hat_total, axis=1),
                                    np.argmax(y_tr_total, axis=1))
    ACC_tr.append( acc_total_tr)
    LOSS_tr.append(loss_total_tr)

    ''' test'''

    pp = 0
    for trial in trials_te:
        XDesign, y_te = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        XDesign = np.swapaxes(XDesign, 1, 2)
        XDesign = torch.tensor(XDesign, dtype=torch.float32)
        y_te = torch.tensor(y_te, dtype=torch.float32)

        y_hat = model(XDesign)
        loss = criterion(y_hat, y_te)
        y_hat = y_hat.detach().numpy()
        ''' apply the language model'''

        ''' calculate acc and loss'''
        loss_total_te += loss.detach().numpy() / len(trials_te)
        acc_total_te += 100 * torch.sum(torch.argmax(torch.tensor(y_hat), dim=1) == torch.argmax(y_te, dim=1)).item() / y_hat.shape[
            0] / len(trials_te)
        if pp == 0:
            y_hat_total = y_hat
            y_te_total = y_te.detach().numpy()
        else:
            y_hat_total = np.concatenate([y_hat_total, y_hat], axis=0)
            y_te_total = np.concatenate([y_te_total, y_te.detach().numpy()], axis=0)
        pp += 1
    ''' gather acc and loss all trials'''
    cf_matrix_te = confusion_matrix(np.argmax(y_hat_total, axis=1),
                                    np.argmax(y_te_total, axis=1))
    ACC_te.append(acc_total_te)
    LOSS_te.append(loss_total_te)



''' cplot loss ans accuracy'''
f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
axes[0].plot(LOSS_tr, 'r')
axes[0].plot(LOSS_te, 'b')
axes[0].legend(['train', 'test'])
axes[0].set_title('loss')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('value')

axes[1].plot(ACC_tr, 'r')
axes[1].plot(ACC_te, 'b')
axes[1].set_title('accuracy')
axes[1].set_xlabel('epochs')
axes[1].set_ylabel('acc.')
axes[1].legend(['train', 'test'])

''' confusion matrix'''
f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
sns.heatmap((cf_matrix_tr), annot=False, cmap='Blues', ax=ax1)
ax1.set_title(' Confusion Matrix train\n\n')
ax1.set_xlabel('\nPredicted Values')
ax1.set_ylabel('Actual Values ')
ax1.set_title('Train dataset ')
ax1.set_yticks(ticks=np.arange(len(list(phones_code_dic.keys()))), labels=list(phones_code_dic.keys()), rotation=0)
ax1.set_xticks(ticks=np.arange(len(list(phones_code_dic.keys()))), labels=list(phones_code_dic.keys()), rotation=90)

sns.heatmap((cf_matrix_te), annot=False, cmap='Blues', ax=ax2)
ax2.set_title(' Confusion Matrix test\n\n')
ax2.set_xlabel('\nPredicted Values')
ax2.set_ylabel('Actual Values ')
ax2.set_title('Train dataset ')
ax2.set_yticks(ticks=np.arange(len(list(phones_code_dic.keys()))), labels=list(phones_code_dic.keys()), rotation=0)
ax2.set_xticks(ticks=np.arange(len(list(phones_code_dic.keys()))), labels=list(phones_code_dic.keys()), rotation=90)