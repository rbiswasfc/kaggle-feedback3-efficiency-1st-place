"""
@created by: heyao
@created at: 2022-06-23 13:09:01
"""
import torch

try:
    from cocolm.modeling_cocolm import COCOLMModel
    from cocolm.tokenization_cocolm import COCOLMTokenizer
except ImportError:
    COCOLMTokenizer = None
    COCOLMModel = None
    print(f"can not import COCOLM, please download github gist manually. [{__name__}]")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def padding_tokens(input_ids, max_length, mask_padding_with_zero=True, pad_token_id=0, pad_token_segment_id=0):
    padding_length = max_length - len(input_ids)
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    token_type_ids = []
    if len(token_type_ids) == 0:
        padding_length = max_length
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}


def encode_plus(self, text, text_pair=None, add_special_tokens=True, max_length=512, padding="max_length",
                truncation=True, stride=None,
                is_split_into_words=None, pad_to_multiple_of=None, return_tensors=None, return_token_type_ids=None,
                return_attention_mask=None, return_overflowing_tokens=None, return_special_tokens_mask=None,
                return_offsets_mapping=None, return_length=None, verbose=None):
    tokens_a = self.tokenize(text)
    if text_pair is not None:
        text_pair = self.tokenize(text_pair)
        if truncation:
            _truncate_seq_pair(tokens_a, text_pair, max_length - 4)
    elif truncation and padding != "do_not_pad":
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[:max_length - 2]

    if add_special_tokens:
        tokens = [self.dictionary.bos_word] + tokens_a + [self.dictionary.eos_word]
        if text_pair is not None:
            tokens += [self.dictionary.eos_word] + text_pair + [self.dictionary.eos_word]
    else:
        tokens = tokens_a + text_pair

    ids = self.convert_tokens_to_ids(tokens)
    if padding == "do_not_pad":
        return {"input_ids": ids}
    return padding_tokens(ids, max_length)


if COCOLMTokenizer is not None:
    COCOLMTokenizer.encode_plus = encode_plus
