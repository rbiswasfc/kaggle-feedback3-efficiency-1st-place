"""
@created by: heyao
@created at: 2022-10-14 22:44:08
"""
from itertools import chain
from typing import List, Union

from sklearn.preprocessing import LabelEncoder


class TagEncoder(object):
    def __init__(self):
        self.label_encoder: Union[None, LabelEncoder] = None
        self.special_tokens = ["[SPECIAL]"]

    def fit_from_list(self, x: List[List]):
        label_encoder = LabelEncoder()
        label_encoder.fit(list(chain.from_iterable(x)))
        classes = label_encoder.classes_.tolist()
        if "[SPECIAL]" in classes:
            classes.remove('[SPECIAL]')
        label_encoder.classes_ = self.special_tokens + classes
        self.label_encoder = label_encoder
        return self

    def convert_ids_to_texts(self, x: List[int]):
        raise NotImplementedError()

    def convert_texts_to_ids(self, x: List[str]):
        return self.label_encoder.transform(x)


if __name__ == '__main__':
    tag_encoder = TagEncoder()
    tag_encoder.fit_from_list([["a", "b", "c"], ["a", "d", "e"]])
    print(tag_encoder.label_encoder.classes_)
    print(tag_encoder.convert_texts_to_ids(["[SPECIAL]", "b", "c", "a"]))
