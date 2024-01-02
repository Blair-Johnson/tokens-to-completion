import random
from ttc import *
from gpt2 import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


#model = AutoModelForCausalLM.from_pretrained("gpt2")
#config = AutoConfig.from_pretrained("gpt2")
#model = GPT2TTC._from_config(config)
model = GPT2TTC.from_pretrained("results/glamorous_bee").cuda()
print("Initialized model.")
tokenizer = AutoTokenizer.from_pretrained("results/glamorous_bee")
print("Initialized tokenizer.")

ttc_tokens = create_ttc_tokens()
ttc_token_ids = tokenizer(''.join(ttc_tokens))['input_ids']
ttc_values_to_ids = dict(zip(ttc_tokens, ttc_token_ids))
print(ttc_token_ids)
print(ttc_values_to_ids)

#prompt = "-- PepsiCo --\nPepsiCo, Inc. is an American multinational food, snack, and"
prompt = "History\n"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
prompt_length_offset = inputs['input_ids'].shape[1]
print("Tokenized inputs.")
generated_ids = model.generate(**inputs, max_new_tokens=1024, 
                                         target_length=512,
                                         ttc_values_to_ids=ttc_values_to_ids,
                                         prompt_length_offset=prompt_length_offset,
                                         use_cache=False,
                                         temperature=0.75,
                                         top_p=0.95,
                                         do_sample=True)
print("Generated outputs.")
res = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print("Decoded outputs.")

print(res)
print('done!')
