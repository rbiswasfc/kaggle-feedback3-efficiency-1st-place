import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, required=True)
ap.add_argument("-output", type=str, required=True)
args = ap.parse_args()

df = pd.read_csv(args.input)
TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


def conv_hl(s):
    if s > 1 and s < 1.25:
        return 1
    elif s > 1.25 and s < 1.75:
        return 1.5
    elif s > 1.75 and s < 2:
        return 2
    elif s > 2 and s < 2.25:
        return 2
    elif s > 2.25 and s < 2.75:
        return 2.5
    elif s > 2.75 and s < 3:
        return 3
    elif s > 3 and s < 3.25:
        return 3
    elif s > 3.25 and s < 3.75:
        return 3.5
    elif s > 3.75 and s < 4:
        return 4
    elif s > 4 and s < 4.25:
        return 4
    elif s > 4.25 and s < 4.75:
        return 4.5
    elif s > 4.75 and s < 5:
        return 5
    else:
        return s


for col in TARGET_COLUMNS:
    df.loc[:, col] = df[col].apply(lambda x: conv_hl(x))

df.to_csv(args.output, index=False)
