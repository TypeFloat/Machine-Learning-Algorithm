import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def train(X, y, step, epoches):
    # 对w,b进行初始化,w是与自变量维度相同的向量，b是标量
    w, b = np.zeros((X.shape[1],)), 0
    # 记录每一次参数更新后的损失值，可用于绘制图像
    loss_list =[]

    m = X.shape[0]
    for _ in range(epoches):

        # 求损失值
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y)**2) / 2 / m
        loss_list.append(loss)

        # 求偏导
        dw = np.dot(X.T, (y_hat - y))
        db = np.sum(y_hat - y)

        # 参数更新
        w -= step * dw
        b -= step * db

    # 保存参数
    params = {'w':w, 'b':b}

    return params, loss_list


def predict(X, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(X, w) + b
    return y_pred


if __name__ == '__main__':
    # 导入波士顿数据集 这里使用了sklearn内置的数据集
    # 因为代码中没有进行内存优化使用该数据集会导致数据溢出
    # 因此对数据集整体缩小处理
    boston = datasets.load_boston()
    data_X = boston.data / 500
    data_y = boston.target / 500
    
    # 简单划分数据集
    scale = int(0.8 * data_X.shape[0])
    train_X = data_X[:scale]
    test_X = data_X[scale:]
    train_y = data_y[:scale]
    test_y = data_y[scale:]

    # 训练
    params, loss_list = train(train_X, train_y, 0.001, 100)

    # 展示损失值变化
    plt.plot(loss_list)
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.savefig('loss')

    # 预测
    y_pred = predict(test_X, params)
    loss = np.sum((y_pred - test_y)**2) / 2 / test_y.shape[0]
    
    print("训练集loss：", loss_list[-1])
    print("测试集loss：", loss)
    