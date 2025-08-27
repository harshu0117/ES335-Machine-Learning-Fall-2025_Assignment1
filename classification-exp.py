import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# Assuming your DecisionTree implementation and metrics are in these files
from tree.base import DecisionTree
from metrics import accuracy, precision, recall

# Set random seed for reproducibility
np.random.seed(42)

# --- Part 2a: Usage of Decision Tree on the Dataset ---

# 1. Generate the dataset
X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2,
    class_sep=0.5,
)

# Convert to pandas DataFrame for our DecisionTree implementation
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_s = pd.Series(y, name="target")

# For plotting the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Classification Data")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.savefig("classification_dataset.png") # Save the plot
plt.close()


# 2. Split the data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_s, test_size=0.3, random_state=42
)

# 3. Train the decision tree
# Using 'information_gain' as criterion and a max_depth of 5 as an example
tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)

# 4. Make predictions on the test set
y_hat = tree.predict(X_test)

# 5. Show the performance metrics
acc = accuracy(y_hat, y_test)
# Calculate per-class precision and recall
unique_classes = sorted(y_s.unique())
precisions = [precision(y_hat, y_test, cls) for cls in unique_classes]
recalls = [recall(y_hat, y_test, cls) for cls in unique_classes]

print("--- Part 2a: Performance Metrics ---")
print(f"Accuracy: {acc:.4f}")
for i, cls in enumerate(unique_classes):
    print(f"Class {cls}:")
    print(f"  Precision: {precisions[i]:.4f}")
    print(f"  Recall: {recalls[i]:.4f}")
print("-" * 35)


# --- Part 2b: 5-Fold Nested Cross-Validation for Optimal Depth ---

print("\n--- Part 2b: Finding Optimal Depth with Nested CV ---")

# Define the range of depths to test
depths_to_try = range(2, 11)

# Outer loop for evaluation (5 folds)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop for hyperparameter tuning (also 5 folds)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

outer_cv_scores = []

fold_num = 1
for train_idx, test_idx in outer_cv.split(X_df, y_s):
    X_train_outer, X_test_outer = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train_outer, y_test_outer = y_s.iloc[train_idx], y_s.iloc[test_idx]

    best_depth = -1
    best_avg_accuracy = -1

    # Inner loop to find the best depth for this outer fold
    for depth in depths_to_try:
        inner_fold_accuracies = []
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner, X_val_inner = X_train_outer.iloc[train_idx_inner], X_train_outer.iloc[val_idx_inner]
            y_train_inner, y_val_inner = y_train_outer.iloc[train_idx_inner], y_train_outer.iloc[val_idx_inner]

            # Train a tree with the current depth
            dt = DecisionTree(criterion="information_gain", max_depth=depth)
            dt.fit(X_train_inner, y_train_inner)
            
            # Evaluate on the inner validation set
            y_pred_inner = dt.predict(X_val_inner)
            inner_fold_accuracies.append(accuracy(y_pred_inner, y_val_inner))
        
        # Average accuracy for the current depth
        avg_accuracy = np.mean(inner_fold_accuracies)
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_depth = depth
            
    # Train the best model on the full outer training data
    best_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    best_tree.fit(X_train_outer, y_train_outer)
    
    # Evaluate on the outer test set
    y_pred_outer = best_tree.predict(X_test_outer)
    outer_fold_accuracy = accuracy(y_pred_outer, y_test_outer)
    outer_cv_scores.append(outer_fold_accuracy)
    
    print(f"Outer Fold {fold_num}: Best Depth={best_depth}, Test Accuracy={outer_fold_accuracy:.4f}")
    fold_num += 1

print(f"\nAverage accuracy from nested cross-validation: {np.mean(outer_cv_scores):.4f}")

# To find the overall optimal depth, we can check which depth was selected most often
# or retrain on the full dataset with the average best depth. For simplicity here,
# let's just note the best depths found.
print(f"Optimal depth is likely around the most frequently chosen values in the folds.")