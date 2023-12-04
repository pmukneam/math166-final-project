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

# Load Twitter Sentiment Analysis dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")

# Combine the 'train' and 'test' splits
combined_data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])], ignore_index=True)

# Convert the combined dataset to a Pandas DataFrame
df = pd.DataFrame(combined_data)

# Split the data into training and test sets
training_df, test_df = split_train_test(df, ratio=0.8)

# Display the shapes of the training and test sets
print("Training set shape:", training_df.shape)
print("Test set shape:", test_df.shape)

# Apply preprocessing functions to training and test sets
training_df = drop_na(training_df)
training_df = drop_duplicates(training_df)
training_df = make_lower_case(training_df)
training_df = remove_special_characters(training_df, keep_numbers=True)

test_df = drop_na(test_df)
test_df = drop_duplicates(test_df)
test_df = make_lower_case(test_df)
test_df = remove_special_characters(test_df, keep_numbers=True)

# Display the preprocessed DataFrames
print("\nPreprocessed Training DataFrame:")
print(training_df.head())

print("\nPreprocessed Test DataFrame:")
print(test_df.head())
