from data_utilities import *
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h_k = 120
f_k=25
number_trials = 80
n_components = 2
epsilon = 1e-5

patient_id = 'DM1013'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
save_result_path = datasets_add + patient_id + '/Results/' +'phonems_psd/'
trials_info_df=pd.read_csv(
        datasets_add + patient_id + '/' + 'sub-' + patient_id + '_ses-intraop_task-lombard_annot-produced-sentences.tsv',
        sep='\t')
trials_id =trials_info_df.trial_id.to_numpy()
''' get language model'''
with open(data_add+'language_model_data.pkl', 'rb') as openfile:
    # Reading from json file
    language_data = pickle.load(openfile)
pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic, count_phonemes = language_data


''' gather all features for the phonemes'''
for trial in trials_id:
    if trial == 1:
        XDesign_total, y_tr_total = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id_total = np.argmax(y_tr_total, axis=-1).reshape([-1,1])


    else:
        XDesign, y_tr = get_trial_data(data_add, trial, h_k, f_k, phones_code_dic, tensor_enable=False)
        phonemes_id = np.argmax(y_tr, axis=-1).reshape([-1,1])
        XDesign_total = np.concatenate([XDesign_total,XDesign], axis=0)
        phonemes_id_total = np.concatenate([phonemes_id_total, phonemes_id], axis=0)
        y_tr_total = np.concatenate([y_tr_total, y_tr], axis=0)

X =  np.swapaxes(XDesign_total, -2, -3).squeeze() #
X = X[:,-1,:,:]
y_onehot=  y_tr_total

class phonemes_dataset(Dataset):
    def __init__(self, X,y):
        self.data = np.float32(X)
        self.target = np.float32(y)
        self.data_size=len(X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [self.data[index], self.target[index]]

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
train = phonemes_dataset(X, y_onehot)
tr_per = 0.8
train, valid = random_split(train,[np.floor(len(train)*tr_per).astype('int'),np.floor(len(train)*(1-tr_per)).astype('int')+1])

# Create Dataloader of the above tensor with batch size = 32
trainloader = DataLoader(train, batch_size=32)
validloader = DataLoader(valid, batch_size=32)

#####
class RNN_Network(nn.Module):

    def __init__(self, in_dims, num_classes):
        self.in_dims = in_dims
        self.device = device
        self.num_classes = num_classes
        self.input_size = in_dims[-1]
        self.sequence_length = in_dims[-2]
        # self.n_channels = in_dims[-3]
        self.out_dims = out_dims
        super(RNN_Network, self).__init__()
        self.hidden_size = 100
        self.num_layers = 2
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.num_classes)
        # self.sm= nn.functional.softmax()
        # pass

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc_out(self.fc(out[:, -1, :]))
        return out#torch.nn.functional.softmax(out,dim=-1)


# Building Our Mode
class CNN_Network(nn.Module):
    # Declaring the Architecture
    def __init__(self, in_dims, out_dims):
        self.in_dims = in_dims
        self.out_dims = out_dims

        super(CNN_Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(4, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(4, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(4, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(4),
        )
        self.fc1 = nn.Linear( 58164 , 100)
        # self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, self.out_dims)


    # Forward Pass
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.softmax(x,dim=-1)

in_dims = X.shape[1:]
out_dims = y_onehot.shape[-1]

model = RNN_Network(in_dims,out_dims)
# if torch.cuda.is_available():
model = model.to(device)

# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training with Validation
epochs = 150
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for data, labels in trainloader:
        # Transfer Data to GPU if available

        data, labels = data.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target, labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()

    valid_loss = 0.0
    valid_acc = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for data, labels in validloader:
        # Transfer Data to GPU if available

        data, labels = data.to(device), labels.to(device)
        # Forward Pass
        target = model(data)

        # Find the Loss
        loss = criterion(target, labels)
        # Calculate Loss
        valid_loss += loss.item()


    print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {  valid_loss / len(validloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')

        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), '../saved_model.pth')


predictions_prob_tr = model(torch.tensor(np.float32(X[train.indices])).to(device)).detach().cpu().numpy()
predictions_prob_te = model(torch.tensor(np.float32(X[valid.indices])).to(device)).detach().cpu().numpy()

predictions_tr = np.argmax(predictions_prob_tr, axis=-1)
predictions_te = np.argmax(predictions_prob_te, axis=-1)

labels_tr = np.argmax(y_onehot[train.indices], axis=-1)
labels_te = np.argmax(y_onehot[valid.indices], axis=-1)
''' visualize xgboost result test'''

uniques_te = np.array(np.unique(np.concatenate([predictions_te,labels_te], axis=0)))

conf_matrix_test = confusion_matrix(labels_te, predictions_te)

disp=ConfusionMatrixDisplay(conf_matrix_test, display_labels=np.array(list(phones_code_dic.keys()))[uniques_te])
disp.plot()
plt.title('test_result RNN, acc='+str(100*accuracy_score(labels_te, predictions_te))+'%')
plt.savefig(save_result_path+'test_result_RNN.png')
plt.savefig(save_result_path+'test_result_RNN.svg',  format='svg')
print("Test-Accuracy of RNN Model::",accuracy_score(labels_te, predictions_te))

uniques_tr = np.unique(np.concatenate([labels_tr,predictions_tr], axis=0))

conf_matrix_train = confusion_matrix(labels_tr,predictions_tr )

disp=ConfusionMatrixDisplay(conf_matrix_train, display_labels=np.array(list(phones_code_dic.keys()))[uniques_tr])
disp.plot()
plt.title('train_result RNN, acc='+str(100*accuracy_score(labels_tr , predictions_tr))+'%')
plt.savefig(save_result_path+'train_result_RNN.png')
plt.savefig(save_result_path+'train_result_RNN.svg',  format='svg')
print("Train Accuracy ofRNN Model::",accuracy_score(labels_tr, predictions_tr))