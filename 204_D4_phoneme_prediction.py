from data_utilities import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
from deep_models import SimpleClassifier, get_trial_data
import pickle


patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic = language_data

''' logistic regression model as discriminative model'''
file_name = data_add + 'trials/trial_' + str(1) + '.pkl'
with open(file_name, "rb") as open_file:
        data_list_trial = pickle.load(open_file)

h_k = 0
Input_size = (data_list_trial[0].shape[0]*data_list_trial[0].shape[1])*(h_k+1)
output_size = data_list_trial[1][data_list_trial[1].columns[data_list_trial[1].columns.str.contains("id_onehot")]].shape[-1]
hidden_dim = 100
model = SimpleClassifier(Input_size,output_size, hidden_dim )
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
LOSS=[]
''' training'''
for epochs in range(20):
    for trial in range(1, 10):
        XDesign, y_tr = get_trial_data(data_add, trial, h_k)
        y_hat = model.forward(XDesign)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        loss = criterion(y_hat, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    LOSS.append(loss.detach().numpy())



plt.figure()
plt.plot(LOSS)