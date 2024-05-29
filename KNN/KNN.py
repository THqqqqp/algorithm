import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

# 前四个是特征，后两个是测试数据
ar_x = [
    [4032, 1680, 1450, 5.3, 5.6],
    [4330, 1535, 1885, 7.8, 14.5],
    [4053, 1740, 1449, 6.2, 10.8],
    [5087, 1868, 1500, 8.5, 25.6],
    [4560, 1822, 1645, 7.8, 15.8],
    [3797, 1510, 1820, 5.5, 9.6]
]
ar_y = [0, 1, 0, 1, 1, 0]

# 数据归一化
ar_min = np.min(ar_x, 0)
ar_max = np.max(ar_x, 0)
ar_range = ar_max - ar_min
nor_ar = np.around((ar_x - ar_range) / ar_min, 4)
print("归一化后数据:\n", nor_ar)

# 划分数据集
train_x = nor_ar[:4]
train_y = ar_y[:4]

# 测试数据集
test_x = nor_ar[4:]
test_y = ar_y[4:]

# 建立模型
knnModel = KNN(n_neighbors=3)
# 训练模型
knnModel.fit(train_x, train_y)
# 测试模型
pre_y = knnModel.predict(test_x)
print("预测结果:", pre_y)
