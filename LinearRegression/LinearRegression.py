import numpy as np
from sklearn import linear_model as lm

# 准备数据
table = np.loadtxt('data/data_train.csv', delimiter=',')
train_x = table[:, 0:3]
train_y = table[:, 3]

# 数据归一化
train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
train_y = (train_y - train_y.mean()) / train_y.std()

# 训练模型
lm_model = lm.LinearRegression()
lm_model.fit(train_x, train_y)
print('coef:', lm_model.coef_)
print('intercept:', lm_model.intercept_)

# 预测
table = np.loadtxt('data/data_test.csv', delimiter=',')
test_x = table[:, 0:3]
test_y = table[:, 3]

# 数据归一化
test_x = (test_x - test_x.mean(axis=0)) / test_x.std(axis=0)
test_y = (test_y - test_y.mean()) / test_y.std()

# 模型的score值
print(lm_model.score(test_x, test_y))
# 预测结果
print(lm_model.predict([[5, 24, 10]]))
