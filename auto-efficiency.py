import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


# --- Data Cleaning ---
# The 'horsepower' column contains '?' for missing values.
# We replace '?' with NaN and then drop rows with NaN values.
data['horsepower'] = data['horsepower'].replace('?', np.nan)
data.dropna(inplace=True)
data['horsepower'] = data['horsepower'].astype(float)

# Drop the 'car name' column as it's a unique identifier and not a useful feature.
data = data.drop('car name', axis=1)

# Separate features (X) and target (y)
X = data.drop('mpg', axis=1)
y = data['mpg']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- a) Usage of your decision tree ---
print("--- Training and Evaluating Your Custom Decision Tree ---")
# For regression, the criterion isn't used for splitting in our implementation
# (it's based on variance reduction), but we still pass it.
my_tree = DecisionTree(criterion='information_gain', max_depth=5)
my_tree.fit(X_train, y_train)
y_hat_my_tree = my_tree.predict(X_test)

# Calculate and print the RMSE for your model
rmse_my_tree = rmse(y_hat_my_tree, y_test)
print(f"  RMSE of your custom Decision Tree: {rmse_my_tree:.4f}")

# --- b) Compare with scikit-learn's decision tree ---
print("\n--- Training and Evaluating Scikit-learn's Decision Tree ---")
sklearn_tree = SklearnDecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)
y_hat_sklearn = sklearn_tree.predict(X_test)

# Calculate and print the RMSE for the scikit-learn model
# We can use our own rmse function for a fair comparison
rmse_sklearn = rmse(pd.Series(y_hat_sklearn), y_test.reset_index(drop=True))
print(f"  RMSE of scikit-learn's Decision Tree: {rmse_sklearn:.4f}")

print("\n--- Comparison ---")
if rmse_my_tree < rmse_sklearn + 0.5: # Allow for small implementation differences
    print("Your model's performance is comparable to scikit-learn's!")
else:
    print("Scikit-learn's model performed significantly better.")