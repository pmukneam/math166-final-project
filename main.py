import re

import pandas as pd
from datasets import load_dataset

from src.model import predict_linear_svm, train_linear_svm
from src.preprocess_data import (drop_duplicates, drop_na, make_lower_case,
                                 remove_special_characters, split_train_test)
from src.validate import evaluate_score
from src.vectorization import (vectorize_data, vectorize_leftover_df,
                               vectorize_training_df)

# Load Twitter Sentiment Analysis dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset['train'])


df = drop_na(df)
df = drop_duplicates(df)
df = make_lower_case(df)
df = remove_special_characters(df, keep_numbers=True)

# # Split the data into training and test sets
# training_df, test_df = split_train_test(df, ratio=0.8)
#
# # Display the shapes of the training and test sets
# print("Training set shape:", training_df.shape)
# print("Test set shape:", test_df.shape)
#
# # Apply preprocessing functions to training and test sets
# training_df = drop_na(training_df)
# training_df = drop_duplicates(training_df)
# training_df = make_lower_case(training_df)
# training_df = remove_special_characters(training_df, keep_numbers=True)
#
# test_df = drop_na(test_df)
# test_df = drop_duplicates(test_df)
# test_df = make_lower_case(test_df)
# test_df = remove_special_characters(test_df, keep_numbers=True)
#
# Display the preprocessed DataFrames
# print("\nPreprocessed Training DataFrame:")
# print(training_df.head())
#
# print("\nPreprocessed Test DataFrame:")
# print(df.head())

# splite the dataset into training and test sets
training_df, testing_df = split_train_test(df, ratio=0.8)
y_train = training_df['feeling']
y_test = testing_df['feeling']
# TF-IDF vectorization
x_train = vectorize_training_df(training_df)
x_test = vectorize_leftover_df(testing_df)
# vectorized_training_df = vectorize_data(training_df)
# vectorized_leftover_df = vectorize_data(leftover_df)

# # print dimension/size
# print("x_train.shape:", x_train.shape)
# print("x_test.shape:", x_test.shape)
# print("y_train.shape:", y_train.shape)
# print("y_test.shape:", y_test.shape)
#
# # print head
# print("x_train.head():", x_train[0:5])
# print("x_test.head():", x_test[0:5])
# print("y_train.head():", y_train[0:5])
# print("y_test.head():", y_test[0:5])
#

# train the model
svm_model = train_linear_svm(x_train, y_train, reg_para=0.5)
# predict the labels
y_pred = predict_linear_svm(x_test, svm_model)
# evaluate the model
scores = evaluate_score(y_test, y_pred)
print(scores)
