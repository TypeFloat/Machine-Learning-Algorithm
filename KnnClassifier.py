import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class KnnClassifier:

    def __init__(self):
        self.dists = 0
    
    def predict(self, X_train, y_train, X_predict, k):
        num_predict = X_predict.shape[0]
        num_train = X_train.shape[0]
        self.dists = np.zeros((num_predict, num_train))
        m = np.dot(X_predict, X_train.T)
        te = np.square(X_predict).sum(axis=1).reshape(-1, 1)
        tr = np.square(X_train).sum(axis=1).reshape(-1, 1)
        self.dists = np.sqrt(-2 * m + tr.T + te)
        y_predict = np.zeros(num_predict)
        for i in range(num_predict):
            closest_y = []
            labels = y_train[np.argsort(self.dists[i, :])].flatten()
            closest_y = labels[0:k]
            c = Counter(closest_y)
            y_predict[i] = c.most_common(1)[0][0]
        y_predict.reshape((-1, 1))
        return y_predict

    def show_distance(self):
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.imshow(self.dists, interpolation='none')
        plt.show()