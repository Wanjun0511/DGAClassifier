import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from data_process import DataProcess
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model


class LSTMClassifier(object):

    def __init__(self, X):

        # build a character dictionary
        self.valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}

        self.max_features = len(self.valid_chars) + 1
        self.maxlen = np.max([len(x) for x in X])

        self.model = None

    def convert(self, data):
        data = [[self.valid_chars[y] for y in x] for x in data]
        data = sequence.pad_sequences(data, maxlen=self.maxlen)
        return data

    def _build_LSTM_model(self):

        model = Sequential()
        model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        return model

    def fit(self, X_train, Y_train, batch_size, epochs):

        # convert charater to numeric according to dictionary
        X_train = self.convert(X_train)

        # train a LSTM model
        model = self._build_LSTM_model()

        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

        self.model = model

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def predict_prob(self, X_test):
        if self.model is not None:
            return self.model.predict_proba(X_test)

    def predict(self, X_test):
        if self.model is not None:
            return self.model.predict_classes(X_test)

    def accuracy(self, Y_test, Y_predict):
        return accuracy_score(Y_test, Y_predict)

    def auc(self, Y_test, Y_predict_prob):
        return roc_auc_score(Y_test, Y_predict_prob)

    def f1(self, Y_test, Y_predict):
        return f1_score(Y_test, Y_predict)

    def confusion_matrix(self, Y_test, Y_predict):
        return confusion_matrix(Y_test, Y_predict)


def run1():

    # get data
    data_process = DataProcess()
    X, Y = data_process.get_all_data()
    X_train, Y_train = data_process.get_train_data()
    X_test, Y_test = data_process.get_test_data()

    # build model
    lstm_classifier = LSTMClassifier(X)
    lstm_classifier.fit(X_train, Y_train, batch_size=128, epochs=1)
    lstm_classifier.save_model("../model/lstm_model_epoch1.h5")

    # convert character to numeric
    X_test = lstm_classifier.convert(X_test)

    # predict and evaluate
    prob = lstm_classifier.predict_prob(X_test)
    auc = lstm_classifier.auc(Y_test, prob)

    pred = lstm_classifier.predict(X_test)
    acc = lstm_classifier.accuracy(Y_test, pred)
    f1 = lstm_classifier.f1(Y_test, pred)
    cm = lstm_classifier.confusion_matrix(Y_test, pred)

    print "the accuracy is : " + str(acc)
    print "the auc is : " + str(auc)
    print "the f1 score is : " + str(f1)
    print "the confusion matrix is : \n"
    print cm


def run2():

    # get data
    data_process = DataProcess()
    X, Y = data_process.get_all_data()
    X_test, Y_test = data_process.get_test_data()

    # load model
    lstm_classifier = LSTMClassifier(X)
    lstm_classifier.load_model("../model/lstm_model_epoch1.h5")

    # convert character to numeric
    X_test = lstm_classifier.convert(X_test)

    # predict and evaluate
    prob = lstm_classifier.predict_prob(X_test)
    auc = lstm_classifier.auc(Y_test, prob)

    pred = lstm_classifier.predict(X_test)
    acc = lstm_classifier.accuracy(Y_test, pred)
    f1 = lstm_classifier.f1(Y_test, pred)
    cm = lstm_classifier.confusion_matrix(Y_test, pred)

    print "the accuracy is : " + str(acc)
    print "the auc is : " + str(auc)
    print "the f1 score is : " + str(f1)
    print "the confusion matrix is : \n"
    print cm


def test():

    # get data
    data_process = DataProcess()
    X, Y = data_process.get_all_data()

    # load model
    lstm_classifier = LSTMClassifier(X)
    lstm_classifier.load_model("../model/lstm_model_epoch1.h5")

    X_test = ["lsowcnfrq.com", "mnjkfgerpw.com", "zxsfgasdq.com", "wanjun0511.github.io"]

    # convert character to numeric
    X_test = lstm_classifier.convert(X_test)

    pred = lstm_classifier.predict(X_test)

    print pred


if __name__ == '__main__':

    # if you have no model, and need to train one, then run1()
    # if you already have a local model, then run2()
    # run test() to predict some handwriting domains
    test()


# 1 epoch:
# the accuracy is : 0.988475932954
# the auc is : 0.999062430802
# the f1 score is : 0.987831558192
# the confusion matrix is :
#
# [[198543   1292]
#  [  3102 178352]]

