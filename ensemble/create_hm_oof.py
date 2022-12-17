import pandas as pd

labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

df = pd.read_csv('./oof_df_21_grouped_v3b.csv')
df.drop(columns=labels, inplace=True)
df.rename(columns={
    'pred_cohesion': 'cohesion',
    'pred_syntax': 'syntax',
    "pred_vocabulary": 'vocabulary',
    "pred_phraseology": 'phraseology',
     "pred_grammar": 'grammar',
    "pred_conventions": 'conventions'
}, inplace=True)

df.to_csv('oof_hm.csv', index=False)