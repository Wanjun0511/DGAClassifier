import csv
from sklearn.cross_validation import train_test_split

def get_positive_samples():

    with open("../data/top-1m.csv", 'r') as f:
        positive_domain = []
        len = 0
        reader = csv.reader(f)
        for row in reader:
            row = list(row)
            domain = row[1]
            positive_domain.append(domain)
            len += 1
        label = [0 for i in range(len)]     # 1000,000
        return positive_domain, label


def get_negative_samples():

    with open("../data/dga-feed.txt", 'r') as f:
        negative_domain = []
        len = 0
        while True:
            row = f.readline()
            if not row:
                break
            if not row.startswith('#'):
                domain = row.split(',')[0]
                negative_domain.append(domain)
                len += 1
        label = [1 for i in range(len)]     # 889,400
        return negative_domain, label


class DataProcess(object):
    def __init__(self):

        positive_x, positive_y = get_positive_samples()
        negative_x, negative_y = get_negative_samples()
        positive_x.extend(negative_x)
        X = positive_x
        positive_y.extend(negative_y)
        Y = positive_y

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def get_all_data(self):
        return self.X, self.Y

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test

