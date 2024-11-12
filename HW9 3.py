import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
df = pd.read_csv('/Users/wangliwei/Downloads/world_ds.csv')

# 检查数据的前几行
print(df.head())

# 假设 'development_status' 是标签列，所有其他列都是特征
# 移除非数值型的 'Country' 列
X = df.drop(['development_status', 'Country'], axis=1)
y = df['development_status']

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 转换
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 输出主成分
print(X_pca)
