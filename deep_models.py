import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np

import random
from sklearn.metrics import confusion_matrix
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from torch.utils.data import Dataset


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional =False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.mixer = nn.Linear(emb_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        # src=src[:,:,None]
        # embedded = self.dropout(self.embedding(src[:,:,1].to(torch.int)))
        embedded = self.dropout(self.embedding(src))
        embedded = self.mixer(embedded)
        # embedded = [src len, batch size, emb dim]

        # embedded=torch.concatenate((embedded,src[:,:,:1]), axis=-1)
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout,bidirectional ):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional =bidirectional )
        if bidirectional:
            self.fc_out = nn.Linear(2*hid_dim, output_dim)
        else:
            self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio, decoder_pretraining):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        if decoder_pretraining:
            src_empty=torch.zeros((src.shape[0],src.shape[1], self.encoder.input_dim))
            hidden, cell = self.encoder(src_empty)
        else:

            hidden, cell = self.encoder(src)

        # first input to the decoder is the first tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs



def train(model, iterator, optimizer, criterion, clip, vocab_size, decoder_pretraining=False):
    model.train()

    epoch_loss = 0
    epoch_acc = 0#np.zeros((vocab_size, vocab_size))
    epoch_conf =  np.zeros((vocab_size, vocab_size))
    for i, batch in enumerate(iterator):
        src ,trg = batch
        src=torch.swapaxes(src,0,1)
        trg = torch.swapaxes(trg, 0, 1)

        optimizer.zero_grad()
        if decoder_pretraining:
            output = model(src, trg,teacher_forcing_ratio=.5,decoder_pretraining=True)
        else:
            output = model(src, trg,teacher_forcing_ratio=.8, decoder_pretraining=False)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape([-1,])

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        epoch_acc += accuracy_score(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(), trg.to(torch.int).detach().cpu().numpy())
        if decoder_pretraining:
            pass
        else:
            # pass
            epoch_conf += confusion_matrix(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                 trg.to(torch.int).detach().cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_conf / len(iterator)


def evaluate(model, iterator, criterion, vocab_size,decoder_pretraining=False):
    model.eval()

    epoch_loss = 0
    epoch_acc=0# np.zeros((vocab_size,vocab_size))
    epoch_conf = np.zeros((vocab_size,vocab_size))
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src ,trg= batch
            src = torch.swapaxes(src, 0, 1)
            trg = torch.swapaxes(trg, 0, 1)

            if decoder_pretraining:
                output = model(src, trg,teacher_forcing_ratio=0, decoder_pretraining=True)
            else:
                output = model(src, trg,teacher_forcing_ratio=0, decoder_pretraining=False)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1,)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            epoch_acc +=accuracy_score(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(), trg.to(torch.int).detach().cpu().numpy())
            if decoder_pretraining:
                pass
            else:
                # pass
                epoch_conf += confusion_matrix(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                        trg.to(torch.int).detach().cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc /len(iterator),epoch_conf /len(iterator),



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class get_dataset(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, data_in,data_out, device):
        self.data_in = torch.tensor(data_in, dtype=torch.float32).to(device)
        self.data_out = torch.tensor(data_out, dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        return [self.data_in[index], self.data_out[index]]

def vec_translate(a, my_dict):
   return np.vectorize(my_dict.__getitem__)(a)

def apply_stress_remove(input_list):
    output_list = []
    if len(input_list) !=0:
        for item in input_list[0]:
            # print(item)
            output_list.append(item[:2])
    else:
        pass

    return [output_list]
