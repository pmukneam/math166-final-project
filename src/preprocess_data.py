import re

import pandas as pd
from datasets import load_dataset


# Preprocessing functions
def drop_na(dataframe):
    return dataframe.dropna()


def drop_duplicates(dataframe):
    return dataframe.drop_duplicates()


def make_lower_case(dataframe):
    return dataframe.applymap(lambda x: x.lower() if isinstance(x, str) else x)


def remove_special_characters(dataframe, keep_numbers=False):
    def remove_special_chars(text):
        if isinstance(text, str):
            if keep_numbers:
                # Remove non-alphabetic characters (keeping numbers)
                return re.sub(r'[^a-zA-Z0-9\s]', '', text)
            else:
                # Remove non-alphabetic characters (excluding numbers)
                return re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    # Apply the function to all string columns
    return dataframe.applymap(remove_special_chars)


def split_train_test(df, ratio=0.25):
    # creates two dataframes, one to be used as training data, and the other to be used for classification.
    training_df = df.sample(frac=ratio, random_state=17)
    leftover_df = df.drop(training_df.index)
    return training_df, leftover_df
