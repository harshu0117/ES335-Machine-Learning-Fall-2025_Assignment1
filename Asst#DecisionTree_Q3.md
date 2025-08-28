
# Assignment 1: Question 3 - Automotive Efficiency

This report details the application of our custom Decision Tree Regressor on the automotive efficiency (auto-mpg) dataset and compares its performance with the `scikit-learn` implementation.

---

## 3a) Custom Decision Tree on Auto-MPG Dataset

We trained our implemented decision tree on the preprocessed auto-mpg dataset to predict the miles per gallon (mpg).

-   **Criterion Used**: `information_gain` (which correctly defaults to MSE for regression).
-   **Max Depth**: 5

### Performance:

-   **Root Mean Squared Error (RMSE)**: 3.3060

---

## 3b) Performance Comparison with Scikit-learn

For a direct comparison, we trained a `DecisionTreeRegressor` from the `scikit-learn` library using the same `max_depth` of 5.

### Comparison Results:

| Model                      | RMSE   |
| :------------------------- | :----- |
| **Your Decision Tree** | 3.3060 |
| **Scikit-learn Decision Tree** | 3.3072 |

### Conclusion:

The performance of our custom decision tree is outstanding, achieving an RMSE of **3.3060**, which is virtually identical to the scikit-learn model's RMSE of **3.3072**. This demonstrates that our implementation is robust and correctly captures the patterns in the data, performing on par with a highly optimized, industry-standard library for this specific task.

***
