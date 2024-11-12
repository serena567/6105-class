import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Load your dataset
data_path = '/Users/wangliwei/Desktop/diabetes.csv'
data = pd.read_csv(data_path)

# Prepare Features and Labels
X = data.drop(columns='Outcome')  # Use 'Outcome' instead of 'outcome'
y = data['Outcome']                # Use 'Outcome' instead of 'outcome'

# Initialize Models
rf_3 = RandomForestClassifier(n_estimators=3, random_state=42)
ada_3 = AdaBoostClassifier(n_estimators=3, random_state=42)
rf_50 = RandomForestClassifier(n_estimators=50, random_state=42)
ada_50 = AdaBoostClassifier(n_estimators=50, random_state=42)

# Perform Cross-Validation
scores_rf_3 = cross_val_score(rf_3, X, y, cv=5)
scores_ada_3 = cross_val_score(ada_3, X, y, cv=5)
scores_rf_50 = cross_val_score(rf_50, X, y, cv=5)
scores_ada_50 = cross_val_score(ada_50, X, y, cv=5)

# Calculate Mean Scores
mean_rf_3 = scores_rf_3.mean()
mean_ada_3 = scores_ada_3.mean()
mean_rf_50 = scores_rf_50.mean()
mean_ada_50 = scores_ada_50.mean()

# Compare Results
print(f'Mean Score RF (3 estimators): {mean_rf_3}')
print(f'Mean Score AdaBoost (3 estimators): {mean_ada_3}')
print(f'Mean Score RF (50 estimators): {mean_rf_50}')
print(f'Mean Score AdaBoost (50 estimators): {mean_ada_50}')

# Summary of Comparisons
if mean_rf_3 > mean_ada_3:
    print("RF (3) outperforms AdaBoost (3)")
else:
    print("AdaBoost (3) outperforms RF (3)")

if mean_rf_50 > mean_ada_50:
    print("RF (50) outperforms AdaBoost (50)")
else:
    print("AdaBoost (50) outperforms RF (50)")
