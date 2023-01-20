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
import torch
from torch import nn

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

X =  np.swapaxes(XDesign_total, -2, -3).squeeze() ##
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


# Building Our Mode
class Network(nn.Module):
    # Declaring the Architecture
    def __init__(self, in_dims, out_dims):
        self.in_dims = in_dims
        self.out_dims = out_dims

        super(Network, self).__init__()
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
model = Network(in_dims,out_dims)
if torch.cuda.is_available():
    model = model.cuda()

# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training with Validation
epochs = 50
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    train_acc = 0.0
    for data, labels in trainloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

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
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

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
        torch.save(model.state_dict(), 'saved_model.pth')