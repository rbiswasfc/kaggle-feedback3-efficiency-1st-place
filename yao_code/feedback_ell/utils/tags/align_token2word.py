"""
@created by: heyao
@created at: 2022-10-14 16:36:02
"""
import numpy as np
from transformers import PreTrainedTokenizer


def tokenize_and_align(words, other_flags, tokenizer: PreTrainedTokenizer, **tokenizer_parameters):
    assert len(words) == len(other_flags), "word and other flag should have same length."
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, **tokenizer_parameters)
    word_ids = encoded.word_ids()
    other_flags = np.array(other_flags)
    aligned_deps = other_flags[word_ids]
    return encoded, word_ids, aligned_deps


if __name__ == '__main__':
    import pickle

    from transformers import AutoTokenizer

    with open("/home/heyao/kaggle/feedback-ells/input/deps.pkl", "rb") as f:
        dependency = pickle.load(f)
    words = [i for i, _ in dependency[3]]
    deps = [i for _, i in dependency[3]]

    model_path = "/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/roberta-large"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
    print(dependency[0][12:16])
    print(f"{'token':<15} || {'word':<15} || {'deps':<15}")
    encoded, word_ids, aligned_deps = tokenize_and_align(words, deps, tokenizer)
    for token, word_id, other_flag in zip(encoded["input_ids"], word_ids, aligned_deps):
        print(f"{tokenizer.convert_ids_to_tokens(token):<15} || {words[word_id]:<15} || {other_flag:<15}")
