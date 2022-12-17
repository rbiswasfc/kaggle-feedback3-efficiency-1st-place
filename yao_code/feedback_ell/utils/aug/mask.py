"""
@created by: heyao
@created at: 2022-08-28 16:30:33
"""
from typing import Optional, Tuple

import numpy as np
import torch
import transformers


def regular_mask_aug_without_pad(input_ids, tokenizer, mask_ratio=.25):
    all_inds = np.arange(1, len(input_ids) - 1)  # make sure CLS and SEP not masked
    n_mask = max(int(len(all_inds) * mask_ratio), 1)
    np.random.shuffle(all_inds)
    mask_inds = all_inds[:n_mask]
    input_ids[mask_inds] = tokenizer.mask_token_id
    return input_ids


def mask_tokens(tokenizer: transformers.PreTrainedTokenizer, inputs: torch.Tensor,
                special_tokens_mask: Optional[torch.Tensor] = None,
                mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix = probability_matrix.float()
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


if __name__ == '__main__':
    from transformers import AutoTokenizer

    model_path = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.mask_token_id)
    input_ids = np.array([1, 232, 231, 342, 3623, 3423, 5314, 2])
    print(regular_mask_aug_without_pad(input_ids, tokenizer=tokenizer))
    input_ids = torch.LongTensor([input_ids.tolist()])
    print(mask_tokens(tokenizer, input_ids, mlm_probability=0.15))
