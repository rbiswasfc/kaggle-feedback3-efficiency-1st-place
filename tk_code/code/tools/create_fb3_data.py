from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import os

fp_dir = '../datasets/feedback-prize-effectiveness'
ID_COL = 'essay_id'
TEXT_COL = 'essay_text'

# Load test essays
def _load_essay(essay_id):
    filename = os.path.join(f"{fp_dir}/train", f"{essay_id}.txt")
    with open(filename, "r") as f:
        text = f.read()
    return [essay_id, text]


def read_essays(essay_ids, num_jobs=12):
    train_essays = []
    results = Parallel(n_jobs=num_jobs, verbose=1)(delayed(_load_essay)(essay_id) for essay_id in essay_ids)
    for result in results:
        train_essays.append(result)

    result_dict = dict()
    for e in train_essays:
        result_dict[e[0]] = e[1]

    essay_df = pd.Series(result_dict).reset_index()
    essay_df.columns = [ID_COL, TEXT_COL]
    return essay_df

test_df = pd.read_csv(f'{fp_dir}/train.csv')
#test_df = test_df.sample(n=10)
essay_ids = test_df[ID_COL].unique().tolist()
print(f'Number of essays: {len(essay_ids)}')
essay_df = read_essays(essay_ids)

final_df = essay_df[[ID_COL, TEXT_COL]]
final_df.rename(columns={ID_COL: 'text_id', TEXT_COL: 'full_text'}, inplace=True)
final_df.to_csv('../datasets/processed/fp_2022_train.csv', index=False)
print(f'Final shape: {final_df.shape}')