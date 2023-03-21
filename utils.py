import re
import torch


def sluggify(string):
    """Converts a string to a slug."""
    string = re.sub(r'[^-\w\s./]', '',
                    string).strip().lower()
    string = re.sub(r'[.]+', '-', string)
    return re.sub(r'[-\s]+', '-', string)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
