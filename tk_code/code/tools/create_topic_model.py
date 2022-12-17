import glob, pandas as pd, numpy as np, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import sys
# %% [code]
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sklearn.feature_extraction.text import CountVectorizer

# %% [code]
sws = stopwords.words("english") + ["n't",  "'s", "'ve"]
topic_model_dir = '../output/topic_model/'
os.makedirs(topic_model_dir, exist_ok=True)

df = pd.read_csv('../datasets/feedback-prize-english-language-learning/train.csv')
DEBUG = False
if DEBUG:
    df = df.iloc[0:10].reset_index(drop=True)

# %% [code]
docs = []
for i,row in df.iterrows():
    txt = row['full_text']
    word_tokens = word_tokenize(txt)
    txt = " ".join([w for w in word_tokens if not w.lower() in sws])
    docs.append(txt)

# %% [code]
#embedding_model = pipeline("feature-extraction", model="microsoft/deberta-v3-base")
#topic_model = BERTopic(embedding_model=embedding_model)
vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words=sws)
topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=50, verbose=True)
#topic_model = BERTopic(n_gram_range=(1, 3), top_n_words=5, verbose=True)

topics, probs = topic_model.fit_transform(docs)


# %% [code]
tm_meta = topic_model.get_topic_info()
tm_meta.to_csv(f"{topic_model_dir}/topic_model_metadata.csv", index=False)

pred_topics = pd.DataFrame()
pred_topics["text_id"] = df['text_id']
pred_topics["topic"] = topics
pred_topics['prob'] = probs
pred_topics.to_csv(f"{topic_model_dir}/topic_model_feedback.csv", index=False)



# %% [code]
topic_model.save(f"{topic_model_dir}/feedback_2021_topic_model")

# %% [code]
import pandas as pd

topics_meta = pd.read_csv(f'{topic_model_dir}/topic_model_metadata.csv')
topics_df = pd.read_csv(f'{topic_model_dir}/topic_model_feedback.csv')

def get_topic(txt):
    txt = txt.split('_')[1:]
    txt = " ".join(txt)
    return txt

topics_meta['topic'] = topics_meta['Name'].apply(get_topic)

topics_df = topics_df.merge(topics_meta[['Topic', 'topic']], left_on='topic', right_on='Topic', how='left')
topics_df = topics_df[['text_id', 'topic_y']]
topics_df = topics_df.rename(columns={'topic_y': 'topic'})
print(topics_df.head())
topics_df.to_csv('./topics.csv', index=False)