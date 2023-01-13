from data_utilities import *
import torch
from mingpt.utils import set_seed
set_seed(3407)
from mingpt.model import GPT
############# our phoneme dataset from 191xxx #######################
datasets_add = './Datasets/'

data_in, data_out, vocab_size = get_phonems_data(datasets_add,
                     phonemes_add= 'LM/our_phonemes_df.csv',
                     dict_add = 'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')

#########################################
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

