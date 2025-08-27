import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Assuming your DecisionTree implementation is in this file
from tree.base import DecisionTree

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(N, M, input_type='discrete', output_type='discrete'):
    """
    Generates a synthetic dataset.
    N: Number of samples
    M: Number of features
    input_type: 'discrete' (binary) or 'real'
    output_type: 'discrete' (binary) or 'real'
    """
    if input_type == 'discrete':
        X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)), columns=[f'attr_{i}' for i in range(M)])
    else: # real
        X = pd.DataFrame(np.random.rand(N, M), columns=[f'attr_{i}' for i in range(M)])

    if output_type == 'discrete':
        # A simple rule for generating the target variable
        y = pd.Series((X.sum(axis=1) % 2).astype(int))
    else: # real
        # A simple linear combination for the target variable
        y = pd.Series(X.dot(np.random.rand(M)) + np.random.normal(0, 0.1, N))
        
    return X, y

def run_experiments(cases):
    """
    Runs timing experiments for different dataset sizes and feature counts.
    """
    # Define the range of N (samples) and M (features) to test
    N_range = [100, 250, 500, 1000, 1500]
    M_range = [5, 10, 15, 20, 25]
    
    # Fixed values for the other dimension
    fixed_M = 10
    fixed_N = 500

    for title, params in cases.items():
        print(f"--- Running Experiments for: {title} ---")
        
        # --- 1. Vary N, Fixed M ---
        train_times_n = []
        predict_times_n = []
        for N in N_range:
            X, y = generate_data(N, fixed_M, **params)
            X_train, X_test, y_train, y_test = X[:int(N*0.7)], X[int(N*0.7):], y[:int(N*0.7)], y[int(N*0.7):]

            tree = DecisionTree(criterion='information_gain', max_depth=5)
            
            # Time the training
            start_time = time.time()
            tree.fit(X_train, y_train)
            train_times_n.append(time.time() - start_time)
            
            # Time the prediction
            start_time = time.time()
            tree.predict(X_test)
            predict_times_n.append(time.time() - start_time)

        # --- 2. Vary M, Fixed N ---
        train_times_m = []
        predict_times_m = []
        for M in M_range:
            X, y = generate_data(fixed_N, M, **params)
            X_train, X_test, y_train, y_test = X[:int(fixed_N*0.7)], X[int(fixed_N*0.7):], y[:int(fixed_N*0.7)], y[int(fixed_N*0.7):]

            tree = DecisionTree(criterion='information_gain', max_depth=5)
            
            # Time the training
            start_time = time.time()
            tree.fit(X_train, y_train)
            train_times_m.append(time.time() - start_time)
            
            # Time the prediction
            start_time = time.time()
            tree.predict(X_test)
            predict_times_m.append(time.time() - start_time)

        # --- 3. Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Runtime Complexity Analysis: {title}', fontsize=16)

        # Plot 1: Training time vs. N
        axes[0, 0].plot(N_range, train_times_n, marker='o')
        axes[0, 0].set_title('Training Time vs. Number of Samples (N)')
        axes[0, 0].set_xlabel('Number of Samples (N)')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True)

        # Plot 2: Prediction time vs. N
        axes[0, 1].plot(N_range, predict_times_n, marker='o', color='green')
        axes[0, 1].set_title('Prediction Time vs. Number of Samples (N)')
        axes[0, 1].set_xlabel('Number of Samples (N)')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True)
        
        # Plot 3: Training time vs. M
        axes[1, 0].plot(M_range, train_times_m, marker='o', color='red')
        axes[1, 0].set_title('Training Time vs. Number of Features (M)')
        axes[1, 0].set_xlabel('Number of Features (M)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)

        # Plot 4: Prediction time vs. M
        axes[1, 1].plot(M_range, predict_times_m, marker='o', color='purple')
        axes[1, 1].set_title('Prediction Time vs. Number of Features (M)')
        axes[1, 1].set_xlabel('Number of Features (M)')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'runtime_analysis_{title.replace(" ", "_")}.png')
        plt.close()
        print(f"Plots saved to runtime_analysis_{title.replace(' ', '_')}.png")


if __name__ == '__main__':
    # Define the four cases to test
    test_cases = {
        "Discrete Input Discrete Output": {'input_type': 'discrete', 'output_type': 'discrete'},
        "Real Input Discrete Output": {'input_type': 'real', 'output_type': 'discrete'},
        "Discrete Input Real Output": {'input_type': 'discrete', 'output_type': 'real'},
        "Real Input Real Output": {'input_type': 'real', 'output_type': 'real'},
    }
    run_experiments(test_cases)
    print("\nAll experiments are complete.")