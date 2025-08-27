import numpy as np
import pandas as pd

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    return (y_hat == y).sum() / len(y)

def precision(y_hat: pd.Series, y: pd.Series, cls: int) -> float:
    """
    Function to calculate the precision
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    predicted_positive = (y_hat == cls).sum()
    return true_positive / predicted_positive if predicted_positive > 0 else 0

def recall(y_hat: pd.Series, y: pd.Series, cls: int) -> float:
    """
    Function to calculate the recall
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    actual_positive = (y == cls).sum()
    return true_positive / actual_positive if actual_positive > 0 else 0

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(RMSE)
    """
    return np.sqrt(np.mean((y_hat - y)**2))

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(MAE)
    """
    return np.mean(np.abs(y_hat - y))