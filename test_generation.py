import random
from ttc import *
from gpt2 import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


#model = AutoModelForCausalLM.from_pretrained("gpt2")
#config = AutoConfig.from_pretrained("gpt2")
#model = GPT2TTC._from_config(config)
#model = GPT2TTC.from_pretrained("results/glamorous_bee").cuda()
model = GPT2TTC.from_pretrained("gpt2").cuda()
print("Initialized model.")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Initialized tokenizer.")

#prompt = "-- PepsiCo --\nPepsiCo, Inc. is an American multinational food, snack, and"
prompt = "The War of 1812\n"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
prompt_length_offset = inputs['input_ids'].shape[1]
print("Tokenized inputs.")
generated_ids = model.generate(**inputs, max_new_tokens=128, 
                                         target_length=64,
                                         bos_offset=6,
                                         temperature=0.75,
                                         num_beams=5,
                                         top_p=0.95,
                                         do_sample=True)
print("Generated outputs.")
res = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print("Decoded outputs.")

print(res)
print('done!')
