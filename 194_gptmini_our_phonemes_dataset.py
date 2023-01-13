from data_utilities import *
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
set_seed(3407)
import pickle
# create a GPT instance
from mingpt.model import GPT
############# our phoneme dataset from 191xxx #######################
datasets_add = './Datasets/'
phones_df_all = pd.read_csv(datasets_add+'LM/our_phonemes_df.csv') ## for our dataset
phones_df_all = pd.read_csv(datasets_add+'LM/phonemes_df_harvard_dataset.csv') ## for harvard dataset
phonemes_dict_df = pd.read_csv(datasets_add + 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')
phones_code_dic = dict(zip(phonemes_dict_df['phonemes'].to_list(),phonemes_dict_df['ids'].to_list()))


phones_df_all['ph_temp'] = 1
max_sentence_L= phones_df_all.groupby(['trial_id']).sum().ph_temp.max()
dataset = np.zeros((phones_df_all.trial_id.max(), max_sentence_L)).astype('int')
for ii in range(phones_df_all.trial_id.max()):
    current_sent = phones_df_all[phones_df_all.trial_id == ii].phoneme_id.to_numpy().astype('int')
    if max_sentence_L != len(current_sent):
        dataset[ii,:] = np.concatenate([current_sent, (phones_code_dic['NAN']*np.ones((max_sentence_L-len(current_sent),)) )], axis=0).astype('int')
    else:
        dataset[ii, :] = current_sent.astype('int')

data_in = dataset
data_out = np.concatenate([dataset,(phones_code_dic['NAN']*np.ones((dataset.shape[0],1)))],axis=1)[:,1:].astype('int')
vocab_size = len(phones_code_dic)
#########################################

class prepare_phoneme_dataset(Dataset):
    def __init__(self ,data_in,data_out, vocab_size):
        self.data_in = torch.tensor(data_in, dtype=torch.int64)
        self.data_out = torch.tensor(data_out, dtype=torch.int64)
        self.sentence_length = data_out.shape[1]
        self.data_length = data_out.shape[0]
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.sentence_length
    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        return [self.data_in[index], self.data_out[index]]



train_dataset = prepare_phoneme_dataset(data_in, data_out, vocab_size=vocab_size)
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 2000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
# now let's perform some evaluation
model.eval()
num_samples = 1
steps=10
do_sample = True
input_length= 10
for ii in np.random.randint(0,train_dataset.data_length,10):
    x = torch.tensor(train_dataset[ii][0][:input_length], dtype=torch.long).to(trainer.device)
    x = x.expand(num_samples, -1)
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    print('-' * 80)
    print('predicted:')
    print(y)
    print('True:')
    print(train_dataset[ii][0][:input_length+steps])

