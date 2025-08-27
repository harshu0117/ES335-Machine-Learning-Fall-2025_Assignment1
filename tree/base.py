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
        # Base cases for recursion
        if depth == self.max_depth or len(y.unique()) == 1 or len(y) < 2:
            if check_if_real(y):
                return y.mean()
            else:
                return y.mode()[0]

        features = X.columns
        best_feature, best_split_value = opt_split_attribute(X, y, self.criterion, features)

        if best_feature is None:
             if check_if_real(y):
                return y.mean()
             else:
                return y.mode()[0]

        # Create a node for the tree
        tree_node = {(best_feature, best_split_value): {}}

        # Split data and build subtrees
        if X[best_feature].dtype.kind in 'ifc': # Real feature split
            left_indices = X[best_feature] <= best_split_value
            right_indices = X[best_feature] > best_split_value
        else: # Discrete feature split
            left_indices = X[best_feature] == best_split_value
            right_indices = X[best_feature] != best_split_value

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Handle empty splits
        if len(y_left) == 0 or len(y_right) == 0:
            if check_if_real(y):
                return y.mean()
            else:
                return y.mode()[0]

        tree_node[(best_feature, best_split_value)]['Y'] = self._build_tree(X_left, y_left, depth + 1)
        tree_node[(best_feature, best_split_value)]['N'] = self._build_tree(X_right, y_right, depth + 1)

        return tree_node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.tree = self._build_tree(X, y, 0)

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree # Leaf node

        feature, value = list(tree.keys())[0]

        # Handle unseen categories
        if feature not in x:
            # If a feature is not in the test data, return the majority class of the current node
            all_leaf_values = []
            def collect_leaf_values(node):
                if not isinstance(node, dict):
                    all_leaf_values.append(node)
                    return
                # In Python 3, node.values() returns a view object, not a list. We need to handle sub-dictionaries correctly.
                for key in node:
                    for sub_key in node[key]:
                        collect_leaf_values(node[key][sub_key])

            collect_leaf_values(tree)
            return pd.Series(all_leaf_values).mode()[0]

        # --- THIS IS THE FIX ---
        # Determine which branch to follow by checking the type of the split value `value`
        if isinstance(value, (int, float)):
            go_yes = x[feature] <= value
        else:
            go_yes = x[feature] == value
        # --- END OF FIX ---

        if go_yes:
            return self._predict_single(x, tree[(feature, value)]['Y'])
        else:
            return self._predict_single(x, tree[(feature, value)]['N'])


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        return X.apply(self._predict_single, axis=1, args=(self.tree,))

    def _plot_recursive(self, tree, indent=""):
        if not isinstance(tree, dict):
            print(indent + "Leaf:", tree)
            return

        feature, value = list(tree.keys())[0]

        # Check if feature is real or discrete for printing condition
        if isinstance(value, (int, float)):
             condition = f"{feature} <= {value:.2f}"
        else:
             condition = f"{feature} == '{value}'"

        print(indent + f"-> If {condition}:")

        print(indent + "  |--> Y:")
        self._plot_recursive(tree[(feature, value)]['Y'], indent + "  |\t")
        print(indent + "  |--> N:")
        self._plot_recursive(tree[(feature, value)]['N'], indent + "  |\t")

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        self._plot_recursive(self.tree)