import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("/Users/wangliwei/Downloads/world_ds.csv")

# 2. Drop the non-numeric columns like 'development_status' and 'Country'
X = df.drop(['development_status', 'Country'], axis=1)  # Assuming 'development_status' is the label and 'Country' is non-numeric
y = df['development_status']  # This is the label

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA to create 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 5. Calculate the correlation of each principal component with the original features
components = pd.DataFrame(pca.components_, columns=X.columns)
print("Principal Components and their correlations with original features:")
print(components)

# Now X_pca contains the transformed features, and components shows the correlations
