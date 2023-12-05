import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_training_df(training_df):
    # vectorize the training data, using the tf-idf method.
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    # tfidf_matrix = pd.DataFrame(tfidf_vectorizer.fit_transform(
    # training_df['text']).toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    # df_train = pd.concat([training_df, tfidf_matrix], axis=1)
    x_train = tfidf_vectorizer.fit_transform(
        training_df['SentimentText']).toarray()
    # save TFIDF mode
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    return x_train


def vectorize_data(training_df, testing_df, tfidf_vectorizer):
    x_train = tfidf_vectorizer.fit_transform(
        training_df['text']).toarray()
    x_test = tfidf_vectorizer.transform(
        testing_df['text']).toarray()
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    return x_train, x_test


def vectorize_leftover_df(leftover_df):
    # Not sure if this function is needed!
    # vectorize the leftover data, using the tf-idf method.
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    # tfidf_matrix = pd.DataFrame(tfidf_vectorizer.fit_transform(
    #     leftover_df['text']).toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    # df_leftover = pd.concat([leftover_df, tfidf_matrix], axis=1)
    x_test = tfidf_vectorizer.fit_transform(
        leftover_df['SentimentText']).toarray()
    return x_test


# print(pd.head(df_train))
