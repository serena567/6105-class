import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset (adjust file path as needed)
df = pd.read_csv('/Users/wangliwei/Downloads/world_ds.csv')

# Step 2: Define the features (X) and the target (y)
X = df.drop(['development_status', 'Country'], axis=1)  # Remove the target column and non-numeric columns
y = df['development_status']  # Target variable

# Step 3: Standardize the features (important for LDA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Create the LDA model and transform the data
lda = LDA(n_components=2)  # n_components is the number of linear discriminants you want to create
X_lda = lda.fit_transform(X_scaled, y)

# Step 5: Print the new features (LDA components)
print(X_lda)
