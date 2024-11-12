import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据集
df = pd.read_csv('/Users/wangliwei/Downloads/world_ds.csv')

# 假设 'country' 列是字符串类型的特征，进行标签编码
label_encoder = LabelEncoder()
df['Country'] = label_encoder.fit_transform(df['Country'])

# 选择特征和目标变量
X = df.drop('development_status', axis=1)  # 假设 'development_status' 是标签列
y = df['development_status']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 使用前向选择方法选择三个最佳特征
selector = SequentialFeatureSelector(knn, n_features_to_select=3)
selector.fit(X_train, y_train)

# 打印选择的特征
selected_features = X.columns[selector.get_support()]
print(f'Selected features: {selected_features}')
