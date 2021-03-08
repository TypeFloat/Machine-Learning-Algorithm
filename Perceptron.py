import numpy as np


class Perceptron:

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X, y, step):
        self.w, self.b = np.zeros((X.shape[1],)), 0
        end_condition = False
        while not end_condition:
            update_criteria = False
            for i in range(len(X)):
                X_train = X[i]
                y_train = y[i]
                if y_train * (np.dot(X_train, self.w) + self.b) <= 0:
                    update_criteria = True
                    dw = -1 * np.dot(y_train.T, X_train)  
                    db = -1 * np.sum(y_train)
                    self.w -= step * dw
                    self.b -= step * db
            if update_criteria == False:
                end_condition = True
        
    def predict(self, X):
        y = np.array([1 if np.dot(x, self.w) + self.b >= 0 else -1 for x in X])
        return y
