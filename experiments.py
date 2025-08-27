"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def _build_tree(self, X, y, depth):
        # Base cases: If max depth is reached, all labels are the same, or not enough samples to split.
        if depth == self.max_depth or len(y.unique()) == 1 or len(X) < 2:
            if check_if_real(y):
                return y.mean()
            else:
                return y.mode().iloc[0]

        features = X.columns
        best_feature, best_split_value = opt_split_attribute(X, y, self.criterion, features)

        # If no split provides any information gain, create a leaf.
        if best_feature is None:
            if check_if_real(y):
                return y.mean()
            else:
                return y.mode().iloc[0]

        # --- BUG FIX STARTS HERE ---
        # Check if the best split found would result in an empty child node.
        if X[best_feature].dtype.kind in 'ifc':
            left_indices = X[best_feature] <= best_split_value
            right_indices = X[best_feature] > best_split_value
        else:
            left_indices = X[best_feature] == best_split_value
            right_indices = X[best_feature] != best_split_value
        
        # If either branch is empty, stop and make a leaf node with the current data's mode/mean.
        if not np.any(left_indices) or not np.any(right_indices):
            if check_if_real(y):
                return y.mean()
            else:
                return y.mode().iloc[0]
        # --- BUG FIX ENDS HERE ---

        # Create a node for the tree
        tree_node = {(best_feature, best_split_value): {}}
        
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        tree_node[(best_feature, best_split_value)]['Y'] = self._build_tree(X_left, y_left, depth + 1)
        tree_node[(best_feature, best_split_value)]['N'] = self._build_tree(X_right, y_right, depth + 1)
        
        return tree_node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.tree = self._build_tree(X, y, 0)

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree # Leaf node

        feature, value = list(tree.keys())[0]
        
        # Use .loc for safe access by feature name
        feature_val = x.loc[feature]

        if pd.api.types.is_numeric_dtype(feature_val):
            go_yes = feature_val <= value
        else:
            go_yes = feature_val == value
            
        if go_yes:
            return self._predict_single(x, tree[(feature, value)]['Y'])
        else:
            return self._predict_single(x, tree[(feature, value)]['N'])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.apply(self._predict_single, axis=1, args=(self.tree,))

    def _plot_recursive(self, tree, indent=""):
        if not isinstance(tree, dict):
            print(indent + "Leaf:", tree)
            return

        feature, value = list(tree.keys())[0]
        
        if isinstance(value, (int, float)):
             condition = f"{feature} <= {value:.2f}"
        else:
             condition = f"{feature} == '{value}'"

        print(indent + f"?({condition})")
        
        print(indent + "--> Y:")
        self._plot_recursive(tree[(feature, value)]['Y'], indent + "\t")
        print(indent + "--> N:")
        self._plot_recursive(tree[(feature, value)]['N'], indent + "\t")

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        self._plot_recursive(self.tree)