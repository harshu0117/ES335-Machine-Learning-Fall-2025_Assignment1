It might seem strange that the regression metrics (RMSE, MAE) are the same for both `information_gain` and `gini_index`. This happens because our `information_gain` utility function is designed to be robust. It first checks if the target variable is real-valued. If it is, it automatically uses **Mean Squared Error (MSE)** as the splitting criterion, which is the correct approach for regression, regardless of whether you passed `"information_gain"` or `"gini_index"`.


### Decision Tree Performance on Test Cases

Here are the results from running the `usage.py` script, which tests the four implementations of the Decision Tree.

-----

### Case 1: Real Input, Discrete Output (Classification)

| Criterion | Accuracy | Per-Class Precision | Per-Class Recall |
| :--- | :--- | :--- | :--- |
| **information\_gain** | 0.90 | `[1.0, 0.818, 0.833, 1.0, 1.0]` | `[1.0, 0.9, 1.0, 1.0, 0.333]` |
| **gini\_index** | 0.867 | `[1.0, 0.714, 1.0, 1.0, 1.0]` | `[0.9, 1.0, 0.8, 1.0, 0.333]` |

\<br\>

-----

### Case 2: Discrete Input, Discrete Output (Classification)

| Criterion | Accuracy | Per-Class Precision | Per-Class Recall |
| :--- | :--- | :--- | :--- |
| **information\_gain** | 0.433 | `[0.5, 1.0, 0.363, 0.0, 0.0]` | `[0.428, 0.5, 0.888, 0.0, 0.0]` |
| **gini\_index** | 0.30 | `[0.0, 0.0, 0.3, 0.0, 0.0]` | `[0.0, 0.0, 1.0, 0.0, 0.0]` |

\<br\>

-----

### Case 3: Real Input, Real Output (Regression)

| Criterion | RMSE | MAE |
| :--- | :--- | :--- |
| **information\_gain (MSE)** | 0.3601 | 0.2407 |
| **gini\_index (MSE)** | 0.3601 | 0.2407 |

\<br\>

-----

### Case 4: Discrete Input, Real Output (Regression)

| Criterion | RMSE | MAE |
| :--- | :--- | :--- |
| **information\_gain (MSE)** | 1.8072 | 1.4146 |
| **gini\_index (MSE)** | 1.8072 | 1.4146 |

-----
