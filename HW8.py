import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('/Users/wangliwei/downloads/non_linear.csv')

# 特征和标签
X = data.drop('label', axis=1)  # 使用除 'label' 以外的所有列作为特征
y = data['label']               # 'label' 列作为标签

# 将数据分为 75% 训练集和 25% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 初始化并训练线性核 SVM 模型
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"线性核 SVM 模型的准确率: {accuracy_linear:.2f}")

# 初始化并训练 RBF 核 SVM 模型
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF 核 SVM 模型的准确率: {accuracy_rbf:.2f}")

# 找到最佳的模型
best_accuracy = max(accuracy_linear, accuracy_rbf)
print(f"最佳模型的准确率: {best_accuracy:.2f}")
