"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np
import pandas as pd

def check_if_real(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return y.dtype.kind in 'ifc'

def entropy(y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def gini_index(y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return 1 - np.sum(probabilities**2)

def information_gain(y: pd.Series, splits: list, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    parent_impurity = 0
    if check_if_real(y): # Regression
        parent_impurity = np.var(y)
    else: # Classification
        if criterion == 'information_gain':
            parent_impurity = entropy(y)
        else:
            parent_impurity = gini_index(y)
            
    total_len = len(y)
    weighted_impurity = 0
    for split in splits:
        split_len = len(split)
        if split_len == 0:
            continue
        if check_if_real(y): # Regression
            weighted_impurity += (split_len / total_len) * np.var(split)
        else: # Classification
            if criterion == 'information_gain':
                weighted_impurity += (split_len / total_len) * entropy(split)
            else:
                 weighted_impurity += (split_len / total_len) * gini_index(split)

    return parent_impurity - weighted_impurity

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -1
    best_feature = None
    best_split_value = None

    for feature in features:
        if X[feature].dtype.kind in 'ifc': # Real-valued feature
            unique_values = sorted(X[feature].unique())
            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i+1]) / 2
                left_y = y[X[feature] <= split_value]
                right_y = y[X[feature] > split_value]
                gain = information_gain(y, [left_y, right_y], criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split_value = split_value
        else: # Discrete/Categorical feature
            unique_values = X[feature].unique()
            for value in unique_values:
                # For discrete features, we split on equality
                eq_y = y[X[feature] == value]
                neq_y = y[X[feature] != value]
                gain = information_gain(y, [eq_y, neq_y], criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split_value = value
                    
    return best_feature, best_split_value