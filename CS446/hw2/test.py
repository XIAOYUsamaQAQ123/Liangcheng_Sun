import matplotlib.pyplot as plt
import hw2
import hw2_utils
import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Example data
w = np.array([0.5, -1.2, 0.8, -0.3, 1.5])
X_train = pd.DataFrame({
    'bias': [1, 1, 1],
    'feature1': [0.1, 0.2, 0.3],
    'feature2': [0.4, 0.5, 0.6],
    'feature3': [0.7, 0.8, 0.9],
    'feature4': [1.0, 1.1, 1.2]
})
X_test = pd.DataFrame({
    'bias': [1, 1],
    'feature1': [0.15, 0.25],
    'feature2': [0.45, 0.55],
    'feature3': [0.75, 0.85],
    'feature4': [1.05, 1.15]
})

# Select top 2 features
top_k_indices, top_k_features, X_train_selected, X_test_selected = hw2.select_top_k_weights(w, X_train, X_test, k=2)

print("Top-k indices:", top_k_indices)
print("Top-k features:", top_k_features)
print("X_train_selected:\n", X_train_selected)
print("X_test_selected:\n", X_test_selected)