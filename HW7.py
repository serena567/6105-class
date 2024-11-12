import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('/Users/wangliwei/downloads/amr_ds.csv')

# 步骤 1: 使用 Naïve Bayes 模型预测 `not_MDR`
# 请替换 'Column1' 和 'Column2' 为实际用于预测的列名
X = data[['Ampicillin', 'Penicillin']]  # 替换为实际的列名
y = data['Not_MDR']

# 将数据分为 75% 训练集和 25% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 初始化 Naïve Bayes 模型
nb = GaussianNB()

# 训练模型
nb.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb.predict(X_test)

# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes 模型的准确率: {accuracy:.2f}")

# 步骤 2: 计算 `amp_pen`, `amp_nmdr`, 和 `pen_nmdr`
amp_pen = data[(data['Ampicillin'] == 1) & (data['Penicillin'] == 1)].shape[0]
amp_nmdr = data[(data['Ampicillin'] == 1) & (data['Not_MDR'] == 1)].shape[0]
pen_nmdr = data[(data['Penicillin'] == 1) & (data['Not_MDR'] == 1)].shape[0]

print(f"amp_pen: {amp_pen}, amp_nmdr: {amp_nmdr}, pen_nmdr: {pen_nmdr}")

# 步骤 3: 创建马尔科夫链并计算长期稳定状态
# 定义转移矩阵
transition_matrix = np.array([
    [0, amp_pen / (amp_nmdr + amp_pen), amp_nmdr / (amp_nmdr + amp_pen)],
    [amp_pen / (pen_nmdr + amp_pen), 0, pen_nmdr / (pen_nmdr + amp_pen)],
    [amp_nmdr / (amp_nmdr + pen_nmdr), pen_nmdr / (amp_nmdr + pen_nmdr), 0]
])

# 计算稳定状态
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]
stationary = stationary / stationary.sum()  # 归一化
stationary = stationary.real.flatten()  # 提取实部

print("长期稳定状态的概率:", stationary)

# 步骤 4: Viterbi 算法预测最可能的状态序列
# 定义状态和观测矩阵
states = ['Ampicillin', 'Penicillin', 'Not_MDR']
observations = ['Infection after surgery', 'No infection after surgery', 'Infection after surgery']
obs_matrix = np.array([[0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])

# 假设初始概率均等
initial_probs = np.array([1/3, 1/3, 1/3])

# Viterbi 算法实现
def viterbi(obs, states, start_p, trans_p, obs_p):
    n_states = len(states)
    T = len(obs)
    dp = np.zeros((n_states, T))
    path = np.zeros((n_states, T), dtype=int)

    # 初始化起始状态概率
    dp[:, 0] = start_p * obs_p[:, obs[0]]
    for t in range(1, T):
        for s in range(n_states):
            dp[s, t] = np.max(dp[:, t-1] * trans_p[:, s]) * obs_p[s, obs[t]]
            path[s, t] = np.argmax(dp[:, t-1] * trans_p[:, s])

    # 回溯以找到最优路径
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])
    for t in range(T-2, -1, -1):
        best_path[t] = path[best_path[t+1], t+1]

    return [states[i] for i in best_path]

# 将观测值编码：1 表示感染，0 表示无感染
obs_encoded = [1, 0, 1]
sequence = viterbi(obs_encoded, states, initial_probs, transition_matrix, obs_matrix)
print("最可能的状态序列:", sequence)
