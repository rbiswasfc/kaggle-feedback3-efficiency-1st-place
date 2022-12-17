"""
@created by: heyao
@created at: 2022-11-02 12:27:33
"""
import numpy as np
import nltk


def _shuffle(seq, text, sep="\n\n"):
    np.random.shuffle(seq)
    new_text = sep.join(seq)
    if new_text == text and len(seq) > 1:
        return _shuffle(seq, text, sep=sep)
    return new_text


def shuffle_paragraph(text):
    paragraph = text.split("\n\n")
    return _shuffle(paragraph, text)


def shuffle_sentence(text):
    paragraphs = text.split("\n\n")
    new_paragraphs = []
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        new_paragraphs.append(_shuffle(sentences, paragraph, sep=""))
    return _shuffle(new_paragraphs, text)
