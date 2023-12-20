import random
from ttc import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
tok_text = tokenizer(dataset['train'][0]['text'])

ttc_tokens = create_ttc_tokens()
add_ttc_tokens(model, tokenizer, ttc_tokens)

ttc_numbers = index_tokens_log2(tok_text)
samp = random.sample(ttc_numbers, 30)

mapped = map_values_to_closest_labels(samp, ttc_tokens)

