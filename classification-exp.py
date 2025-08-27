import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Convert to pandas DataFrame/Series for our Decision Tree
X = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
y = pd.Series(y)

# --- Question 2 a) ---
print("--- Question 2 a) ---")
# Split data into 70% training and 30% testing
split_index = int(0.7 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the decision tree
tree = DecisionTree(criterion='information_gain', max_depth=5)
tree.fit(X_train, y_train)

# Make predictions on the test set
y_hat = tree.predict(X_test)

# Show the results
print("Learned Decision Tree:")
tree.plot()
print("\nPerformance on Test Set:")
print(f"  Accuracy: {accuracy(y_hat, y_test):.4f}")
for cls in y.unique():
    # Sorting unique classes to ensure consistent order
    cls = sorted(y.unique())[cls]
    print(f"  Class {cls}:")
    print(f"    Precision: {precision(y_hat, y_test, cls):.4f}")
    print(f"    Recall:    {recall(y_hat, y_test, cls):.4f}")


# --- Question 2 b) ---
print("\n--- Question 2 b) ---")
print("Finding optimal depth using 5-fold cross-validation...")

depths = range(2, 11) # Test depths from 2 to 10
mean_accuracies = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    fold_accuracies = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model
        dt_fold = DecisionTree(criterion='information_gain', max_depth=depth)
        dt_fold.fit(X_train_fold, y_train_fold)
        
        # Evaluate on the validation set
        y_val_hat = dt_fold.predict(X_val_fold)
        fold_accuracies.append(accuracy(y_val_hat, y_val_fold))
        
    mean_accuracies.append(np.mean(fold_accuracies))
    print(f"  Depth: {depth}, Mean CV Accuracy: {np.mean(fold_accuracies):.4f}")

optimal_depth = depths[np.argmax(mean_accuracies)]
print(f"\nOptimal tree depth found: {optimal_depth}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(depths, mean_accuracies, marker='o')
plt.title('Cross-Validation Accuracy vs. Tree Depth')
plt.xlabel('Max Depth')
plt.ylabel('Mean 5-Fold CV Accuracy')
plt.xticks(depths)
plt.grid(True)
plt.show()