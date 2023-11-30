from datasets import load_dataset
import pandas as pd
import re

# Preprocessing functions
def drop_na(dataframe):
    return dataframe.dropna()

def drop_duplicates(dataframe):
    return dataframe.drop_duplicates()

def make_lower_case(dataframe):
    return dataframe.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def remove_special_characters(dataframe):
    def remove_special_chars(text):
        x = re.sub(r'[^a-zA-Z\s]', '', text) if isinstance(text, str) else text
        x = str(x)
        return x.strip()
    return dataframe.applymap(remove_special_chars)

# Load Twitter Sentiment Analysis dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")

# Assuming the dataset has a 'train' split
train_data = dataset['train']

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(train_data)

# Display the original DataFrame
print("Original DataFrame:")
print(df.head())

# Apply preprocessing functions
df = drop_na(df)
df = drop_duplicates(df)
df = make_lower_case(df)
df = remove_special_characters(df)

# Display the preprocessed DataFrame
print("\nPreprocessed DataFrame:")
print(df.head())

# Testing Section
def test_preprocessing_functions():
    # Test drop_na function
    df_na = drop_na(df.copy())
    print("\nDataFrame after drop_na:")
    print(df_na.head())

    # Test drop_duplicates function
    df_duplicates = drop_duplicates(df.copy())
    print("\nDataFrame after drop_duplicates:")
    print(df_duplicates.head())

    # Test make_lower_case function
    df_lower = make_lower_case(df.copy())
    print("\nDataFrame after make_lower_case:")
    print(df_lower.head())

    # Test remove_special_characters function
    df_no_special_chars = remove_special_characters(df.copy())
    print("\nDataFrame after remove_special_characters:")
    print(df_no_special_chars.head())

# Run the testing section
test_preprocessing_functions()
