"""
@created by: heyao
@created at: 2022-08-25 01:24:33
"""
import torch
import transformers


class SequenceBucketPadCollator(object):
    def __init__(self, max_length, tokenizer: transformers.PreTrainedTokenizer, is_train=True,
                 target_is_float=False, pad_val=-100):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.target_is_float = target_is_float
        self.pad_val = pad_val

    def _ensure_max_length(self, x):
        max_len = max([sum(i["attention_mask"]) for i in x])
        max_len = min(self.max_length, max_len)
        return max_len

    def __call__(self, batch):
        """

        :param batch:
        :return:
        """
        has_weight = len(batch[0]) == 3
        if not isinstance(batch[0], tuple):
            x = batch
            y = tuple()
            weight = None
        else:
            x = [i[0] for i in batch]
            y = [i[1] for i in batch]
            weight = None
            if has_weight:
                weight = [i[-1] for i in batch]
        xs = {}
        max_len = self._ensure_max_length(x)
        for key in x[0].keys():
            if "labels" not in key and "tag" not in key:
                pad_val = self.tokenizer.pad_token_id
            else:
                pad_val = self.pad_val
            # if self.target_is_float and "labels" in key:
            #     continue
            if "stat" not in key:
                xs[key] = torch.vstack([torch.LongTensor(i[key] + [pad_val] * (max_len - len(i[key]))) for i in x])
            else:
                xs[key] = torch.vstack([torch.FloatTensor(i[key]) for i in x])
        y = torch.FloatTensor(y) if self.target_is_float else torch.LongTensor(y)
        if not has_weight:
            return xs, y
        return xs, y, torch.FloatTensor(weight)


class ReinaSequenceBucketPadCollator(object):
    def __init__(self, max_length, tokenizer: transformers.PreTrainedTokenizer, is_train=True,
                 target_is_float=False, pad_val=-100):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.target_is_float = target_is_float
        self.pad_val = pad_val

    def _ensure_max_length(self, x):
        max_len = max([sum(i["attention_mask"]) for i in x])
        max_len = min(self.max_length, max_len)
        return max_len

    def __call__(self, batch):
        """

        :param batch:
        :return:
        """
        if not isinstance(batch[0], tuple):
            x = batch
            y_target = tuple()
            y = tuple()
        else:
            x = [i[0] for i in batch]
            y_target = [i[1] for i in batch]
            y = [i[2] for i in batch]
        xs = {}
        max_len = self._ensure_max_length(x)
        for key in x[0].keys():
            if "labels" not in key and "tag" not in key:
                pad_val = self.tokenizer.pad_token_id
            else:
                pad_val = self.pad_val
            # if self.target_is_float and "labels" in key:
            #     continue
            if "stat" not in key:
                xs[key] = torch.vstack([torch.LongTensor(i[key] + [pad_val] * (max_len - len(i[key]))) for i in x])
            else:
                xs[key] = torch.vstack([torch.FloatTensor(i[key]) for i in x])
        y = torch.FloatTensor(y) if self.target_is_float else torch.LongTensor(y)
        y_target = torch.FloatTensor(y_target) if self.target_is_float else torch.LongTensor(y_target)
        return xs, y, y_target


class MaxSeqLenPadCollator(SequenceBucketPadCollator):
    def __init__(self, max_length, tokenizer: transformers.PreTrainedTokenizer, is_train=True, target_is_float=True,
                 pad_val=-100):
        super().__init__(max_length, tokenizer, is_train=is_train, target_is_float=target_is_float, pad_val=pad_val)

    def _ensure_max_length(self, x):
        return self.max_length


if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer
    from feedback_ell.utils.dataset.simple import CompetitionDataset

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    collate_fn = SequenceBucketPadCollator(max_length=1024, tokenizer=tokenizer)
    data = [
        ({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}, [1, 2, 3, 4]),
        ({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}, [1, 2, 3, 4]),
        ({"input_ids": [1, 2, 3, 5], "attention_mask": [1, 1, 1, 1]}, [1, 2, 3, 5]),
        ({"input_ids": [1, 2, 2, 5], "attention_mask": [1, 1, 1, 1]}, [1, 2, 2, 5])
    ]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    dataset = CompetitionDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    it = iter(dataloader)
    while 1:
        try:
            print(next(it))
        except StopIteration:
            print(next(dataloader.__iter__()))
            break
