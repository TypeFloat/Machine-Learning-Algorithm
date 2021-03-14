import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        self.w = 0
        self.b = 0
        self.loss_list = []

    def _sigmoid(self, x):
        z = 1 / (1 + np.exp(-x))
        return z

    def fit(self, X, y, step, epochs):
        m = X.shape[0]
        self.W, self.b = np.zeros((X.shape[1], 1)), 0
        for _ in range(epochs):
            y_hat = self._sigmoid(np.dot(X, self.W) + self.b)
            loss = -1 / m * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
            self.loss_list.append(loss)
            dW = np.dot(X.T, (y_hat-y)) / m
            db = np.sum(y_hat-y) / m
            self.W = self.W - step * dW
            self.b = self.b - step * db

    def predict(self, X):
        y_predict = self._sigmoid(np.dot(X, self.W) + self.b).reshape((-1,))
        y_predict = np.array([1 if float(elem) > 0.5 else 0 for elem in y_predict]).reshape((-1, 1))
        return y_predict

    def show_loss(self):
        plt.plot(self.loss_list)
        plt.show()