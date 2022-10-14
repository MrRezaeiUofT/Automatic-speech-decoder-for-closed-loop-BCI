import torch
import torch.nn as nn
from neural_utils import  calDesignMatrix_V2
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def get_trial_data(data_add,trial, h_k, phones_code_dic):
    '''
    get a batch of data and [re[are it for the model
    :param data_add: data address
    :param trial: trial id
    :param h_k: length of history
    :return:
    '''
    scaler = StandardScaler()
    file_name = data_add + 'trials/trial_' + str(trial) + '.pkl'
    with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)

    X_tr = np.swapaxes(data_list_trial[0], 2, 0)
    X_tr = X_tr.reshape([X_tr.shape[0], -1])
    X_tr = scaler.fit_transform(X_tr)
    y_tr = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].to_numpy()
    XDesign = calDesignMatrix_V2(X_tr, h_k + 1).reshape([X_tr.shape[0], -1])

    ''' delete 'sp' and 'NaN' from dataset'''
    sp_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['sp'])[0]
    nan_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['NAN'])[0]
    delet_phonemes_indx=np.concatenate([nan_index,sp_index],axis=0)

    XDesign = np.delete(XDesign, delet_phonemes_indx, 0)
    y_tr = np.delete(y_tr, delet_phonemes_indx, 0)
    return torch.tensor(XDesign, dtype=torch.float32),  torch.tensor(y_tr, dtype=torch.float32)

class SimpleClassifier(nn.Module):
    """
    simple classifier
    """
    def __init__(self, Input_size, output_size, hidden_dim):
        super(SimpleClassifier,  self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.Input_size = Input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        self.fcin = nn.Linear(self.Input_size, self.hidden_dim)
        self.fch = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fout =nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x =  F.relu(self.fcin(x))
        # If the size is a square, you can specify with a single number
        x =  F.relu(self.fch(x))
        x = F.relu(self.fch(x))
        x = F.relu(self.fch(x))
        x = self.fout(x)
        # x = F.softmax(x, dim=1)
        return x


    def accuracy(self, out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()


