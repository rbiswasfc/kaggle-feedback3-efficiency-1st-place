import pandas as pd

op_folder = '../output/lgb_ens08_nofts'

TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


for i, col in enumerate(TARGET_COLUMNS):
    oof_file = op_folder + f'_{col}/oof_df.csv'
    oof_df = pd.read_csv(oof_file)
    if i==0:
        df = oof_df
    else:
        df = df.merge(oof_df, how='left')
    print(df.shape)

print(df.head())
df.to_csv(f'{op_folder.split("/")[-1]}.csv', index=False)