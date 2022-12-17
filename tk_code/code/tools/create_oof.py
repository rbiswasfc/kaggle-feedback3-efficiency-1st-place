import pandas as pd
import os
import glob

oof_path = '/Users/trushant/kaggle/exp209'

oof_files = glob.glob(oof_path + "/oof_df_fold*.csv")
print(oof_files)
oof_df = pd.DataFrame()
for oof_path in oof_files:
    oof_df_ = pd.read_csv(oof_path)
    print(oof_df_.shape)
    oof_df = pd.concat([oof_df, oof_df_])

#TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#oof_df.drop(columns=TARGET_COLUMNS, inplace=True)

oof_df = oof_df.sort_values(by='text_id').reset_index(drop=True)
print(oof_df.shape)
oof_df.to_csv('exp209-dbv3b-mindist.csv', index=False)

