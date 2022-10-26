import torch
import torch.nn as nn
from neural_utils import  calDesignMatrix_V2, calDesignMatrix_V3, calDesignMatrix_V4
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from torch.autograd import Variable

def get_trial_data(data_add,trial, h_k,f_k, phones_code_dic, tensor_enable):
    '''
    get a batch of data and [re[are it for the model
    :param data_add: data address
    :param trial: trial id
    :param h_k: length of history
    :return:
    '''
    file_name = data_add + 'trials/trial_' + str(trial) + '.pkl'
    with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)
    data_list_trial[1] =data_list_trial[1].reset_index()
    X_tr = np.swapaxes(data_list_trial[0], 2, 0)
    if tensor_enable:
        X_tr = X_tr.reshape([X_tr.shape[0], -1])
        X_tr-=X_tr.mean(axis=0)
        X_tr /= (X_tr.std(axis=0))
        XDesign = calDesignMatrix_V2(X_tr, h_k + 1)
    else:
        # X_tr -= X_tr.mean(axis=0)
        # X_tr /= ( X_tr.std(axis=0))
        XDesign = calDesignMatrix_V4(X_tr, h_k + 1, f_k)


    y_tr = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].to_numpy()


    ''' delete 'sp' and 'NaN'  and non-onset_phonemes from dataset'''
    sp_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['sp'])[0]
    nan_index = np.where(np.argmax(y_tr, axis=1) == phones_code_dic['NAN'])[0]
    non_phoneme_onset = data_list_trial[1][data_list_trial[1].phoneme_onset == 0].index.to_numpy()
    delet_phonemes_indx = np.unique(np.concatenate([nan_index, sp_index, non_phoneme_onset],axis=0))

    XDesign = np.delete(XDesign, delet_phonemes_indx, 0)
    y_tr = np.delete(y_tr, delet_phonemes_indx, 0)
    if tensor_enable:
        return torch.tensor(XDesign, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    else:
        return XDesign, y_tr

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
        x=torch.nn.functional.softmax(x, dim=1)
        return x


    def accuracy(self, out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()


class RNN_Classifier(nn.Module):
    """
    simple classifier
    """
    def __init__(self, Input_size, history_length, output_size, hidden_dim, n_layers):
        super(RNN_Classifier,  self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.Input_size = Input_size
        self.history_length = history_length
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(
            self.Input_size, self.hidden_dim, self.n_layers, batch_first=True, dropout=.1
        )
        self.flatten = nn.Flatten()
        self.softplus = nn.Softplus()
        self.fin = nn.Linear(Input_size+self.hidden_dim, self.hidden_dim)
        self.fch = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fout = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        h0 = Variable(torch.randn(self.n_layers, x.shape[0], self.hidden_dim))

        out_h, hnh = self.rnn(x[:, :-1, :], h0)
        x_embed = torch.cat((out_h[:,-1,:], torch.squeeze(x[:,-1,:])), -1)
        x_embed = F.relu(self.fin(x_embed))
        x = F.relu(self.fch(x_embed))
        x = self.fout(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


    def accuracy(self, out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()



class CNN_Classifier(nn.Module):
    """
    simple classifier
    """
    def __init__(self):
        super(CNN_Classifier,  self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.Input_size = Input_size
        # self.history_length = history_length
        # self.output_size = output_size
        # self.hidden_dim = hidden_dim
        # self.n_layers = n_layers

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(3, 3)
        # self.conv2 = nn.Conv2d(6, 16, 5)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=10, stride=1, padding=1),
            nn.MaxPool2d(1, 5), nn.ReLU(inplace=True), nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=20, kernel_size=10, stride=1, padding=1),
            nn.MaxPool2d(1, 5), nn.ReLU(inplace=True), nn.BatchNorm2d(20),
        )
        self.fc1 = nn.Linear(320, 42)




    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # x = F.relu(self.fc2(x))
        # x = torch.nn.functional.softmax(x, dim=1)
        return x


