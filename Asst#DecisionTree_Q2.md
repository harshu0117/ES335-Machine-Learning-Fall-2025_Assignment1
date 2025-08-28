# Assignment 1: Question 2 - Classification Experiments

This file documents the experiments performed using our custom Decision Tree implementation on a generated dataset.

-----

## 2a) Performance on a 70/30 Split

The decision tree was trained on 70% of the generated data and evaluated on the remaining 30%.

  - **Criterion Used**: `information_gain`
  - **Max Depth**: 5

### Performance Metrics:

  - **Accuracy**: 0.8333
  - **Per-Class Metrics**:
      - **Class 0**:
          - **Precision**: 0.8125
          - **Recall**: 0.8667
      - **Class 1**:
          - **Precision**: 0.8571
          - **Recall**: 0.8000

### Generated Data Plot

Here is a visualization of the dataset we used for this experiment:

-----

## 2b) Finding Optimal Depth with Nested Cross-Validation

We used a 5-fold nested cross-validation approach to determine the optimal tree depth. The inner loop selected the best depth from a range of 2 to 10, and the outer loop evaluated the model's performance.

### Results of Nested Cross-Validation:

  - **Outer Fold 1**: Best Depth=8, Test Accuracy=0.8500
  - **Outer Fold 2**: Best Depth=4, Test Accuracy=0.8000
  - **Outer Fold 3**: Best Depth=6, Test Accuracy=0.7500
  - **Outer Fold 4**: Best Depth=7, Test Accuracy=0.8000
  - **Outer Fold 5**: Best Depth=8, Test Accuracy=0.9000

### Conclusion:

The **average accuracy** from the nested cross-validation was **0.8200**.

Based on the results, the optimal depth appears to be around **8**, as it was selected most frequently in the outer folds and contributed to the highest accuracy score (0.9000). This indicates that a moderately deep tree is effective for this particular dataset.

