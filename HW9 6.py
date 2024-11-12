import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Step 1: Load the dataset
df = pd.read_csv('/Users/wangliwei/Downloads/world_ds.csv')

# Step 2: Define the features (X) and the target (y)
X = df.drop(['development_status', 'Country'], axis=1)  # Features (excluding 'development_status' and 'Country')
y = df['development_status']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the features (important for PCA and LDA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and evaluate KNN classifier on original features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
accuracy_original = knn.score(X_test_scaled, y_test)
print('Accuracy on original features:', accuracy_original)

# Step 6: Apply PCA (3 components) and evaluate KNN
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
knn.fit(X_train_pca, y_train)
accuracy_pca = knn.score(X_test_pca, y_test)
print('Accuracy on PCA features:', accuracy_pca)

# Step 7: Apply LDA (2 components) and evaluate KNN
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)
knn.fit(X_train_lda, y_train)
accuracy_lda = knn.score(X_test_lda, y_test)
print('Accuracy on LDA features:', accuracy_lda)
