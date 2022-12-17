#!/usr/bin/env bash

mkdir datasets
kaggle datasets download -d conjuring92/feedback-ells-dataset-migration
unzip feedback-ells-dataset-migration.zip -d ./datasets
kaggle datasets download -d trushk/fb3-en-pl
unzip fb3-en-pl.zip -d ./datasets/fb3-en-pl
kaggle datasets download -d trushk/fb3-en-pl-8f
unzip fb3-en-pl.zip -d ./datasets/fb3-en-pl-8f