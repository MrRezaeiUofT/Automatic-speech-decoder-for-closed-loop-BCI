import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np

from torch.autograd import Variable
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
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout,encoder_type, bidirectional =False ):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        self.encoder_type=encoder_type
        if self.encoder_type== 'neural':
            self.embedding = nn.Linear(input_dim, emb_dim)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.mixer = nn.Linear(emb_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        if self.encoder_type== 'neural':
            embedded = self.dropout(self.embedding(src))
        else:
            embedded = self.dropout(self.embedding(src.to(torch.int)))


        # embedded = self.mixer(embedded)
        # embedded = [src len, batch size, emb dim]

        # embedded=torch.concatenate((embedded,src[:,:,:1]), axis=-1)
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell, outputs


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
    def __init__(self, encoder_neural, encoder_sentence, decoder,max_hist_len, device):
        super().__init__()

        self.encoder_neural = encoder_neural
        self.encoder_sentence = encoder_sentence
        self.decoder= decoder
        self.device = device
        self.max_hist_len=max_hist_len

        assert encoder_neural.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder neural and decoder must be equal!"
        assert encoder_neural.n_layers == decoder.n_layers, \
            "Encoder and decoder  must have equal number of layers!"
        assert encoder_sentence.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder sentences and decoder must be equal!"
        assert encoder_sentence.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.context_to_mu=nn.Linear(self.encoder_neural.hid_dim, self.encoder_neural.hid_dim)
        self.context_to_logvar = nn.Linear(self.encoder_neural.hid_dim, self.encoder_neural.hid_dim)
    def forward(self, src, trg, teacher_forcing_ratio, decoder_pretraining):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        neural_to_sent_rate=torch.floor(src.shape[0]/trg.shape[0])
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        if decoder_pretraining:
            # src_empty=torch.zeros((src.shape[0],src.shape[1], self.encoder.input_dim))
            hidden_sent, cell_sent,outputs_encoder_sentence = self.encoder_sentence(src)
            # hidden = [batch size,n layers * n directions, hid dim]
            z = Variable(torch.randn(hidden_sent.shape)).to(self.device)

            # z = [n layers * n directions,batch  size, hid dim]
            mu = self.context_to_mu(hidden_sent)
            logvar = self.context_to_logvar(hidden_sent)
            std = torch.exp(0.5 * logvar)
            z = z * std + mu
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
            # first input to the decoder is the first tokens
        else:

            hidden_neural, cell_neural, outputs_encoder_neural = self.encoder_neural(src)
            z = Variable(torch.randn(hidden_neural.shape)).to(self.device)
            mu_neural = self.context_to_mu(hidden_neural)
            logvar_neural = self.context_to_logvar(hidden_neural)
            std_neural = torch.exp(0.5 * logvar_neural)
            z = z * std_neural + mu_neural

            hidden_sent, cell_sent,outputs_encoder_sentence = self.encoder_sentence(trg)
            mu_sent = self.context_to_mu(hidden_sent)
            logvar_sent = self.context_to_logvar(hidden_sent)
            std_sent = torch.exp(0.5 * logvar_sent)
            # kld = (-0.5 * torch.sum(logvar_neural - torch.pow(mu_neural-mu_sent, 2) - torch.exp(logvar_neural) + 1, 1)).mean().squeeze()
            kld = (-0.5 * torch.sum((logvar_neural-logvar_sent)
                                    +self.encoder_neural.hid_dim
                                    -torch.divide(torch.pow(mu_neural - mu_sent, 2), torch.exp(logvar_sent)+1e-5)
                                    - torch.divide(torch.exp(logvar_neural),torch.exp(logvar_sent)+1e-5),
                                    1)).mean().squeeze()


        input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states

            # output_neural, _, _ = self.decoder(trg[0, :], z, cell)
            if decoder_pretraining:
                output, z, cell_sent = self.decoder(input, z, cell_sent)
            else:
                output, z, cell_neural = self.decoder(input, z, cell_neural)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # outputs_neural[t]=output_neural
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs,outputs, kld



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
            output, out_neural, kld = model(src, trg,teacher_forcing_ratio=.8,decoder_pretraining=decoder_pretraining)
        else:
            output, out_neural,kld = model(src, trg,teacher_forcing_ratio=.5, decoder_pretraining=decoder_pretraining)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        # out_neural = out_neural[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape([-1,])

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        if decoder_pretraining:
            loss = criterion(output, trg)+.1*kld
        else:
            loss = criterion(output, trg)+.4*kld

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        epoch_acc += accuracy_score(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(), trg.to(torch.int).detach().cpu().numpy())
        if decoder_pretraining:
            pass
        else:
            pass
            # epoch_conf += confusion_matrix(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
            #                      trg.to(torch.int).detach().cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_conf / len(iterator)


def evaluate(model, iterator, criterion, vocab_size, decoder_pretraining=False,no_LM=False):
    model.eval()

    epoch_loss = 0
    epoch_acc=0# np.zeros((vocab_size,vocab_size))
    epoch_conf = np.zeros((vocab_size,vocab_size))
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src ,trg= batch
            src = torch.swapaxes(src, 0, 1)
            trg = torch.swapaxes(trg, 0, 1)

            if no_LM:
                output, out_neural,_ = model(src, trg,teacher_forcing_ratio=0, decoder_pretraining=decoder_pretraining)
            else:

                output, out_neural,_ = model(src, trg,teacher_forcing_ratio=0, decoder_pretraining=decoder_pretraining)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].reshape(-1, output_dim)
            out_neural=out_neural[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1,)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            if no_LM:
                loss = criterion(out_neural, trg)
                epoch_acc += accuracy_score(out_neural.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                            trg.to(torch.int).detach().cpu().numpy())
                if decoder_pretraining:
                    epoch_conf = confusion_matrix(out_neural.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                                   trg.to(torch.int).detach().cpu().numpy())
                else:
                    # pass
                    epoch_conf += confusion_matrix(out_neural.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                            trg.to(torch.int).detach().cpu().numpy())
            else:

                epoch_acc +=accuracy_score(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(), trg.to(torch.int).detach().cpu().numpy())
                if decoder_pretraining:
                    epoch_conf = confusion_matrix(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                                   trg.to(torch.int).detach().cpu().numpy())
                else:
                    # pass
                    epoch_conf += confusion_matrix(output.argmax(axis=1).to(torch.int).detach().cpu().numpy(),
                                            trg.to(torch.int).detach().cpu().numpy())
            epoch_loss += loss.item()
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




def get_balanced_weight_for_classifier(inputs, vocab_size):
    uniques, counts = np.unique(inputs, return_counts = True)
    counts = (counts+1) / counts.sum()
    weights = np.zeros((vocab_size,))
    weights[uniques.astype('int')] = 1 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights,dtype=torch.float)
