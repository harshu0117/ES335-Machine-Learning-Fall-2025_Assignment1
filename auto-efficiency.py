import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Assuming your DecisionTree implementation and metrics are in these files
from tree.base import DecisionTree
from metrics import rmse

# Set random seed for reproducibility
np.random.seed(42)

# --- Part 3a: Usage of Your Decision Tree for Automotive Efficiency ---

# 1. Load the dataset
# The dataset is space-separated. We'll also handle the missing values ('?')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]
df = pd.read_csv(
    url, names=column_names,
    na_values="?", comment="\t",
    sep=" ", skipinitialspace=True
)

# 2. Preprocess the data
# Drop the car_name column as it's not a useful feature
df = df.drop("car_name", axis=1)
# Drop rows with missing values for simplicity
df = df.dropna()
# One-hot encode the 'origin' feature
df = pd.get_dummies(df, columns=['origin'], prefix='', prefix_sep='')

# 3. Split the data into features (X) and target (y)
X = df.drop("mpg", axis=1)
y = df["mpg"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train your custom decision tree
# Using 'information_gain' (which defaults to MSE for regression) and max_depth=5
my_tree = DecisionTree(criterion="information_gain", max_depth=5)
my_tree.fit(X_train, y_train)

# 5. Make predictions and evaluate your model
my_tree_predictions = my_tree.predict(X_test)
my_tree_rmse = rmse(my_tree_predictions, y_test)

print("--- Part 3a: Your Decision Tree Performance ---")
print(f"RMSE of your Decision Tree: {my_tree_rmse:.4f}")
print("-" * 45)


# --- Part 3b: Comparison with Scikit-learn's Decision Tree ---

# 1. Train a scikit-learn Decision Tree Regressor
# We use the same max_depth for a fair comparison
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)

# 2. Make predictions and evaluate the scikit-learn model
sklearn_predictions = sklearn_tree.predict(X_test)
sklearn_rmse = np.sqrt(mean_squared_error(y_test, sklearn_predictions))

print("\n--- Part 3b: Scikit-learn vs. Your Model ---")
print(f"RMSE of Scikit-learn's Decision Tree: {sklearn_rmse:.4f}")
print(f"RMSE of Your Decision Tree:             {my_tree_rmse:.4f}")
print("-" * 45)