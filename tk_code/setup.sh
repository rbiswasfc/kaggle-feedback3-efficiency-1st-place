#!/usr/bin/env bash

mkdir datasets
kaggle datasets download -d trushk/fb3-ell-train-folds
kaggle competitions download -c feedback-prize-english-language-learning
kaggle datasets download -d trushk/fb3-en-pl
kaggle datasets download -d conjuring92/kw-dataset
kaggle datasets download -d trushk/fb3-en-pl-8f
kaggle datasets download -d trushk/mlm-deb-l-dapt-tapt

unzip fb3-ell-train-folds.zip -d processed
unzip feedback-prize-english-language-learning.zip -d feedback-prize-english-language-learning
unzip fb3-en-pl.zip -d processed/pl
unzip kw-dataset.zip -d kw-dataset
unzip fb3-en-pl-8f.zip -d processed/pl
unzip mlm-deb-l-dapt-tapt.zip -d mlm-deb-l-dapt-tapt
