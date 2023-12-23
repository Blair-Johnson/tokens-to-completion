import random
from ttc import *
from gpt2 import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


#model = AutoModelForCausalLM.from_pretrained("gpt2")
#config = AutoConfig.from_pretrained("gpt2")
#model = GPT2TTC._from_config(config)
model = GPT2TTC.from_pretrained("gpt2")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

label_stack = []
input_id_stack = []
ttc_id_stack = []
attention_mask_stack = []
for i in range(3):
    sample = tokenizer(dataset['train'][i]['text'])
    input_ids = sample['input_ids']

    ttc_tokens = create_ttc_tokens()
    add_ttc_tokens(model, tokenizer, ttc_tokens)

    ttc = map_seq_to_ttc(input_ids, ttc_tokens)
    ttc = tokenizer(''.join(ttc))['input_ids']
    input_ids.append(tokenizer.eos_token_id)

    label_stack.append(input_ids.copy()[-128:])
    input_id_stack.append(input_ids.copy()[-128:])
    ttc_id_stack.append(ttc[-128:])
    attention_mask_stack.append(sample['attention_mask'][-128:])

sample['labels'] = torch.tensor(label_stack)
sample['input_ids'] = torch.tensor(input_id_stack)
sample['ttc_ids'] = torch.tensor(ttc_id_stack)
sample['attention_mask'] = torch.tensor(attention_mask_stack)

res = model(**sample)
print('done!')
