import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def split_train_test(df, ratio=0.25):
    # creates two dataframes, one to be used as training data, and the other to be used for classification.
    training_df = pd.sample(df, frac=ratio, random_state=17)
    leftover_df = pd.drop(df, training_df.index)

def vectorize_training_df(training_df):
    # vectorize the training data, using the tf-idf method.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = pd.DataFrame(tfidf_vectorizer.fit_transform(training_df['text']).toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df = pd.concat([df, tfidf_matrix], axis=1)

def vectorize_leftover_df(leftover_df):
    # Not sure if this function is needed!
    # vectorize the leftover data, using the tf-idf method.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = pd.DataFrame(tfidf_vectorizer.fit_transform(leftover_df['text']).toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df = pd.concat([df, tfidf_matrix], axis=1)

print(pd.head(df))
