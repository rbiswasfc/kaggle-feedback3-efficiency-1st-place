"""
@created by: heyao
@created at: 2022-09-08 17:18:06
"""
import pandas as pd


def _load_text(path, id_):
    with open(f"{path}/{id_}.txt", "r") as f:
        return f.read().strip()


def load_unlabeled_data(fb1_path=None, fb3_path=None):
    fb1_path = fb1_path or "/home/heyao/kaggle/feedback-effective/input/feedback-prize-2021"
    # fb2_path = fb2_path or "/home/heyao/kaggle/feedback-effective/input/feedback-prize-effectiveness"
    fb3_path = fb3_path or "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning"
    assert not fb1_path.endswith("/")
    # assert not fb2_path.endswith("/")
    assert not fb3_path.endswith("/")
    task_data = pd.read_csv(fb3_path + "/train.csv")
    df1 = pd.read_csv(fb1_path + "/train.csv")
    # df2 = pd.read_csv(fb2_path + "/train.csv")
    first_ids = df1.id.unique()
    # second_ids = df2.essay_id.unique()
    df1 = pd.DataFrame({"text_id": first_ids})
    # df2 = pd.DataFrame({"text_id": second_ids})
    path1 = fb1_path + "/train"
    # path2 = fb2_path + "/train"
    df1["full_text"] = df1.text_id.apply(lambda x: _load_text(path1, x))
    # df2["full_text"] = df2.text_id.apply(lambda x: _load_text(path2, x))
    # df_all = df1.drop_duplicates(["text_id"])
    df = df1[~df1.text_id.isin(task_data.text_id)].reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = load_unlabeled_data()
    print(df.shape)
    print(df.head(2))
