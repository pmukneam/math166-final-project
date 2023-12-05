# Import libraries
import re

import pandas as pd


def prepare_dataset(file_path, num_rows):
    # Read the CSV file, selecting only the 'review_text' and 'review_score' columns, and the specified number of rows
    df = pd.read_csv(file_path,
                     usecols=[
                         "Sentiment", "SentimentText"], nrows=num_rows)
    # Apply the preprocess_review_text function to the 'review_text' column and store the results in a new column 'filtered_review_text'
    df['text'] = df['SentimentText'].replace('\n', ' ', regex=True)

    # Remove duplicate rows
    df.drop_duplicates(subset=["text"], inplace=True)

    # Remove NAs Row
    df.dropna(subset=["text"], inplace=True)

    # Remove rows where 'filtered_review_text' is empty
    df = df[df["text"].str.len() > 0]

    # Shuffle the DataFrame and reset the index
    df = df.sample(frac=1).reset_index(drop=True)

    # sentiment = {'Positive': 1, 'Negative': -1}
    #
    # df['Sentiment'] = df['Sentiment'].map(sentiment)
    #
    return df


def filter_dataframe(df, n, pos):
    # Filter positive and negative reviews
    df_positive = df[df['Sentiment'] == 1]
    df_negative = df[df['Sentiment'] == 0]

    # Calculate the number of positive and negative reviews
    n_positive = round(n * pos)
    n_negative = n - n_positive

    # Select n_positive from df_positive and n_negative from df_negative
    df_positive_sample = df_positive.sample(n_positive)
    df_negative_sample = df_negative.sample(n_negative)

    # Concatenate the two dataframes
    result = pd.concat([df_positive_sample, df_negative_sample])

    return result


def update_sentiment_values(df):
    # Define the mapping
    sentiment_mapping = {1: 1, 0: -1}

    # Apply the mapping
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

    return df

# ------------------------------------------------------------ Main ------------------------------------------------------------


raw_file_path = 'dataset/raw/twitter-sentiment-analysis.csv'
processed_df = prepare_dataset(raw_file_path, 1000000)
filtered_df = filter_dataframe(processed_df, 100000, 0.6)
filtered_df = update_sentiment_values(filtered_df)
filtered_df.to_csv('dataset/raw/filtered_dataset.csv',
                   index=False)
