import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/bengaliai-cv19/train.csv")
    print(df.head())
    df.loc[:, "kfold"] = -1

    # shuffle the dataset and make new index col
    # to maintain the index order
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_id.values
    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values

    mskf = MultilabelStratifiedKFold(n_splits=5)

    # label the fold# in the dataset for train/test validation
    for fold, (train_idx, valid_idx) in enumerate(mskf.split(X, y)):
        print("TRAIN: ", train_idx, "VALID: ", valid_idx)
        df.loc[valid_idx, "kfold"] = fold

    print(df.shape)
    print(df.kfold.value_counts())
    df.to_csv("../input/bengaliai-cv19/train_folds.csv", index=False)
