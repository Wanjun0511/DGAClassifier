from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from data_process import DataProcess
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import load_model


class TFIDFNNClassifier(object):

    def __init__(self, X):

        self.tfidf_vec = TfidfVectorizer(input='content', analyzer='char', ngram_range=(2, 2))
        term_doc_matrix = self.tfidf_vec.fit_transform(X)

        self.max_features = term_doc_matrix.shape[1]

        self.model = None

    def convert(self, data):
        return self.tfidf_vec.transform(data)

    def _build_nn_model(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.max_features, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model

    def fit(self, X_train, Y_train, batch_size, epochs):

        X_train = self.convert(X_train)

        model = self._build_nn_model()

        model.fit(X_train.todense(), Y_train, batch_size=batch_size, epochs=epochs)

        self.model = model

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def predict_prob(self, X_test):
        if self.model is not None:
            return self.model.predict_proba(X_test.todense())

    def predict(self, X_test):
        if self.model is not None:
            return self.model.predict_classes(X_test.todense())

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
    tfidf_nn = TFIDFNNClassifier(X)
    tfidf_nn.fit(X_train, Y_train, batch_size=128, epochs=10)
    tfidf_nn.save_model("../model/tfidf_nn_model_epoch10.h5")

    # convert character to numeric
    X_test = tfidf_nn.convert(X_test)

    # preict and evaluate
    prob = tfidf_nn.predict_prob(X_test)
    auc = tfidf_nn.auc(Y_test, prob)

    pred = tfidf_nn.predict(X_test)
    acc = tfidf_nn.accuracy(Y_test, pred)
    f1 = tfidf_nn.f1(Y_test, pred)
    cm = tfidf_nn.confusion_matrix(Y_test, pred)

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
    tfidf_nn = TFIDFNNClassifier(X)
    tfidf_nn.load_model("../model/tfidf_nn_model_epoch10.h5")

    # convert character to numeric
    X_test = tfidf_nn.convert(X_test)

    # predict and evaluate
    prob = tfidf_nn.predict_prob(X_test)
    auc = tfidf_nn.auc(Y_test, prob)

    pred = tfidf_nn.predict(X_test)
    acc = tfidf_nn.accuracy(Y_test, pred)
    f1 = tfidf_nn.f1(Y_test, pred)
    cm = tfidf_nn.confusion_matrix(Y_test, pred)

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
    tfidf_nn = TFIDFNNClassifier(X)
    tfidf_nn.load_model("../model/tfidf_nn_model_epoch10.h5")

    X_test = ["lsowcnfrq.com", "mnjkfgerpw.com", "zxsfgasdq.com", "wanjun0511.github.io"]
    X_test = tfidf_nn.convert(X_test)

    pred = tfidf_nn.predict(X_test)

    print pred


if __name__ == '__main__':

    # if you have no model, and need to train one, then run1()
    # if you already have a local model, then run2()
    # run test() to predict some handwriting domains
    test()


# 1 epoch:
# loss: 0.0810
# the accuracy is : 0.977670480921
# the auc is : 0.996857931269
# the f1 score is : 0.97642426357
# the confusion matrix is :
#
# [[196465   3530]
#  [  4984 176310]]

# 10 epochs:
# loss:             0.0271
# the accuracy is : 0.988313326637
# the auc is :      0.998910876319
# the f1 score is : 0.987692987544
# the confusion matrix is :
#
# [[198026   1804]
#  [  2652 178807]]


