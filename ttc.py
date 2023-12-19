
def create_ttc_tokens(min_power=3, max_power=11):
    ttc_tokens = [f'<SUB-{2**exp}>' for exp in range(min_power, max_power+1)]
    return ttc_tokens

def add_ttc_tokens(model, tokenizer, ttc_tokens):
    tokenizer.add_tokens(ttc_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
