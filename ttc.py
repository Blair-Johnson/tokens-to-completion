import numpy as np

def create_ttc_tokens(min_power=3, max_power=11):
    ttc_tokens = [f'<SUB-{2**exp}>' for exp in range(min_power, max_power+1)]
    ttc_tokens.append(f'<SUPER-{2**max_power}>')
    return ttc_tokens

def add_ttc_tokens(model, tokenizer, ttc_tokens):
    tokenizer.add_tokens(ttc_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64) # for A100

def find_closest_upper_bound_label(value, thresholds, max_token):
    """
    Finds the closest upper bound label for a given value.
    :param value: The log2 transformed value.
    :param thresholds: List of tuples with (log2(threshold), label).
    :return: The corresponding label.
    """
    for threshold, label in thresholds:
        if value < threshold:
            return label
    return max_token  # In case no label is found.

def map_values_to_closest_labels(values, labels):
    """
    Maps a list of log2 transformed values to their closest upper bound labels.
    :param values: List of integers (log2 transformed).
    :param labels: List of strings representing the thresholds.
    :return: List of strings with the mapped labels.
    """
    threshold_labels = [label for label in labels if 'SUB' in label]
    max_token = [label for label in labels if 'SUPER' in label][0]
    # Prepare the thresholds list with their log2 values
    thresholds = sorted([(int(np.log2(int(label.strip('<>').split('-')[1]))), label) for label in threshold_labels])
    # Map each value to its label
    return [find_closest_upper_bound_label(value, thresholds, max_token) for value in values]

def index_tokens_log2(input_ids):
    ttc_numbers = list(reversed(range(len(input_ids))))
    ttc_numbers[-1] = 1 # avoid log(0)
    ttc_numbers = list(np.log2(ttc_numbers).astype(int))
    return ttc_numbers

def map_seq_to_ttc(example, ttc_tokens):
    ttc_numbers = index_tokens_log2(example)
    mapped = map_values_to_closest_labels(ttc_numbers, ttc_tokens)
    return mapped
