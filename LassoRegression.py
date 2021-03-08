import numpy as np

class LassoRegression:

    def __init__(self):
        self.w = 0
        self.b = 0
        self.loss_list = []

    def _sign_(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def fit(self, X, y, step, epoches, alpha):
        self.w, self.b = np.zeros((X.shape[1],)), 0
        m = X.shape[0]
        for _ in range(epoches):
            y_hat = np.dot(X, self.w) + self.b
            loss = np.sum((y_hat - y)**2) / 2 / m + np.sum(alpha * np.abs(self.w))
            self.loss_list.append(loss)
            dw = np.dot(X.T, (y_hat - y)) + alpha * np.vectorize(self._sign_)(self.w)
            db = np.sum(y_hat - y)
            self.w -= step * dw
            self.b -= step * db

    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
