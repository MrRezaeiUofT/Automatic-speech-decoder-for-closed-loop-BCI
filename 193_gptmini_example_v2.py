import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
model_type = 'gpt2-xl'
device = 'cuda'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval()


num_samples=3
steps=20
prompt = 'Andrej Karpathy, the'
do_sample = True
if use_mingpt:
        tokenizer = BPETokenizer()

        x = tokenizer(prompt).to(device)

x = x.expand(num_samples, -1)

# forward the model `steps` times to get samples, in a batch
y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-' * 80)
        print(out)
