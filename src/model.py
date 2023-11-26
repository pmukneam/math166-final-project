# This file contains the functions for training and testing the linear SVM model.
from sklearn.svm import LinearSVC


def train_linear_svm(x_train, y_train, reg_para, dual="auto", max_iter=1000):
    """
    Train a linear SVM model with the given training data and parameters.
    :param x_train: training data
    :param y_train: training labels
    :param reg_para: regularization parameter
    :param dual: dual or primal formulation
    :param max_iter: maximum number of iterations
    :return: trained linear SVM model
    """
    svm_model = LinearSVC(C=reg_para, dual=dual)
    svm_model.fit(x_train, y_train, max_iter=max_iter)
    return svm_model


def predict_linear_svm(x_test, svm_model):
    """
    Predict the labels of the given test data using the given SVM model.
    :param x_test: test data
    :param svm_model: SVM model
    :return: predicted labels
    """
    return svm_model.predict(x_test)
