import sys
import random
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append('../')

from ttc import *
from gpt2 import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

PROMPT = 'History\n'
MAX_NEW_TOKENS = 1022
N_SAMPLES = 10
BINS = [8,16,32,64,128,256,512]

def create_ttc_token_map(tokenizer):
    ttc_tokens = create_ttc_tokens()
    ttc_token_ids = tokenizer(''.join(ttc_tokens))['input_ids']
    ttc_values_to_ids = dict(zip(ttc_tokens, ttc_token_ids))
    return ttc_values_to_ids

def generate_ttc(model, 
                 tokenizer, 
                 prompt, 
                 max_new_tokens, 
                 target_length):
    ttc_values_to_ids = create_ttc_token_map(tokenizer)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    prompt_length_offset = inputs['input_ids'].shape[1]
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=max_new_tokens,
                                   target_length=target_length,
                                   ttc_values_to_ids=ttc_values_to_ids,
                                   prompt_length_offset=prompt_length_offset,
                                   pad_token_id=tokenizer.eos_token_id,
                                   use_cache=False,
                                   temperature=0.75,
                                   top_p=0.95,
                                   do_sample=True)
    return generated_ids.shape[1]
                                   
def generate(model, 
             tokenizer, 
             prompt, 
             max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.eos_token_id,
                                   use_cache=True,
                                   temperature=0.75,
                                   top_p=0.95,
                                   do_sample=True)
    return generated_ids.shape[1]
                                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='rank of local task')
    args = parser.parse_args()

    SEED = 2024 + args.rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    gpt2 = AutoModelForCausalLM.from_pretrained('gpt2').cuda()
    gpt2_tok = AutoTokenizer.from_pretrained('gpt2')

    gpt2_lengths = []
    for i in tqdm(range(N_SAMPLES*len(BINS))):
        gpt2_lengths.append(generate(gpt2,
                                     gpt2_tok,
                                     PROMPT,
                                     MAX_NEW_TOKENS))
    gpt2_results = pd.DataFrame({'gpt2':gpt2_lengths})
    gpt2_results.to_pickle(f'./hist_results/gpt2_rank_{args.rank}.pkl')

    del gpt2, gpt2_tok

    gpt2_ttc = GPT2TTC.from_pretrained('../results/glamorous_bee').cuda()
    gpt2_ttc_tok = AutoTokenizer.from_pretrained('../results/glamorous_bee')

    ttc_lengths = {i:[] for i in range(len(BINS))}
    for i in tqdm(range(len(BINS))):
        target_length = BINS[i]
        for _ in range(N_SAMPLES):
            ttc_lengths[i].append(generate_ttc(gpt2_ttc,
                                               gpt2_ttc_tok,
                                               PROMPT,
                                               MAX_NEW_TOKENS,
                                               target_length))

    ttc_results = pd.DataFrame(ttc_lengths)
    ttc_results.to_pickle(f'./hist_results/ttc_rank_{args.rank}.pkl')
