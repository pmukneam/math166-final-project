import re
import time

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.model import predict_linear_svm, train_linear_svm
from src.preprocess_data import (drop_duplicates, drop_na, make_lower_case,
                                 remove_special_characters, split_train_test)
from src.validate import evaluate_score
from src.vectorization import (vectorize_data, vectorize_leftover_df,
                               vectorize_training_df)

# Load Twitter Sentiment Analysis dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")
# df = pd.read_csv("dataset/raw/twitter-sentiment-analysis.csv",
#                  usecols=['Sentiment', 'SentimentText'])

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Load the dataset
# df = pd.read_csv('dataset/raw/twitter-sentiment-analysis.csv')
# df = df.iloc[:50000]
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
y_train = np.asarray(training_df['feeling'])
y_test = np.asarray(testing_df['feeling'])
# TF-IDF vectorization
# x_train = vectorize_training_df(training_df)
# x_test = vectorize_leftover_df(testing_df)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
x_train, x_test = vectorize_data(training_df, testing_df, tfidf_vectorizer)
# vectorized_training_df = vectorize_data(training_df)
# vectorized_leftover_df = vectorize_data(leftover_df)

# NMF
nmf_model = NMF(n_components=50, init='random', random_state=0, max_iter=1000)
x_train_nmf = nmf_model.fit_transform(x_train)
x_test_nmf = nmf_model.transform(x_test)


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

reg_arr = [0.01, 0.1, 1, 10, 100]

for reg_para in reg_arr:
    # train the model
    tic = time.perf_counter()
    # svm_model = train_linear_svm(x_train, y_train, reg_para=reg_para)
    svm_model = train_linear_svm(x_train_nmf, y_train, reg_para=reg_para)
    # predict the labels
    toc = time.perf_counter()
    # save the model
    # joblib.dump(svm_model, 'models/svm_model_reg_' + str(reg_para) + '.pkl')

    # y_pred = predict_linear_svm(x_test, svm_model)
    y_pred = predict_linear_svm(x_test_nmf, svm_model)
    # evaluate the model
    scores = evaluate_score(y_test, y_pred)
    print("reg_para:", reg_para)
    print("Training time:", toc - tic)
    print(scores)
