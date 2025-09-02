import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


## Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        D (2 x d Float Tensor): MAP estimation of P(X_j=1|Y=i)

    '''
    # N, d = X.shape

    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]
    
    D0 = (X_class_0.sum(dim=0)) / (X_class_0.shape[0])
    D1 = (X_class_1.sum(dim=0)) / (X_class_1.shape[0])
    D = torch.stack([D0, D1]).to(dtype=torch.float32)
    return D

def bayes_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    p = torch.mean((y == 0).float()).to(dtype=torch.float32)
    return p

def bayes_classify(D, p, X):
    '''
    Arguments:
        D (2 x d Float Tensor): returned value of `bayes_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    p_1 = 1 - p
    p_0 = p
    X = X.to(dtype=torch.float32) # do we really need it?
    
    
    alpha = 0.1
    D_smooth = (D + alpha) / (1 + 2 * alpha) # hope it works

    log_prob_0 = X @ torch.log(D_smooth[0]) + (1 - X) @ torch.log(1 - D_smooth[0]) + torch.log(p_0)
    log_prob_1 = X @ torch.log(D_smooth[1]) + (1 - X) @ torch.log(1 - D_smooth[1]) + torch.log(p_1)


    y_pred = (log_prob_1 > log_prob_0).long()

    return y_pred



## Problem Logistic Regression
def logistic(X, Y, lrate=0.025, num_iter=5000):
    '''
    Arguments:
        X (N x d float): the feature matrix
        Y (N float): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 float: the parameters w'
    
    '''
    N, d = X.shape
    X = np.concatenate([np.ones([N, 1]), X], axis=1)
    w = np.zeros((d + 1, 1))
    Y = Y.reshape(-1, 1)

    prev_loss = float('inf')
    
    for _ in range(num_iter):
        logits = X @ w
        logits = np.clip(logits, -100, 100)
        predictions = 1 / (1 + np.exp(-logits))
        
        loss = -np.mean(Y * np.log(predictions + 1e-11) + (1 - Y) * np.log(1 - predictions + 1e-11))
        
        if abs(prev_loss - loss) < 0.000001:
            break
        prev_loss = loss

        gradient = -(1/N) * (X.T @ (Y - predictions))
        w -= lrate * gradient
    
    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    w_logistic = logistic(X, Y)
    w_ols = linear_gd(X, Y)
    
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', alpha=0.7)
    x_vals = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    ax.plot(x_vals, -(w_logistic[0] + w_logistic[1] * x_vals) / w_logistic[2], label='Logistic Regression', color='blue')
    ax.plot(x_vals, -(w_ols[0] + w_ols[1] * x_vals) / w_ols[2], label='Least Squares', color='green')
    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    return plt.gcf()



## Problem Linear Regression
#
# This function implements gradient descent for linear regression, and is provided as reference
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    n = X.shape[0]
    X = np.concatenate([np.ones([n,1]),X], axis=1)
    w = np.zeros([X.shape[1],1])
    Y = Y.reshape([-1,1])
    for t in range(num_iter):
        w = w - (1/n)*lrate * (X.T @ (X @ w - Y))
    return w[:,0]

def add_bias(X):
    """
    Add a bias term (a column of ones) as the first column to the input data.

    Parameters:
        X : pandas DataFrame of shape (num_samples, num_features)
    
    Returns:
        Xp : data matrix (pandas DataFrame) with a bias column (all ones) added as the first column, with column name 'bias'
    """
    return np.c_[np.ones(X.shape[0]), X]


def least_squares_regression(X_train, y_train):
    """
    Compute the closed-form least squares regression solution w
                 min_w ||X_train w - y_train ||_2^2

    Parameters:
       X_train: n x d training data (possibly contain bias column)
       y_train: n-dimensional training target

    Returns: 
       np.array, d-dimensional weight vector w that solves the least squares problem
    """
    return np.linalg.pinv(X_train) @ y_train


def select_top_k_weights(w, X_train, X_test, k):
    """
    Select top-k features based on the absolute value of weights, including the bias term.
    
    Parameters:
        w: d-dimensional weight vector (first element corresponds to bias)
        X_train: DataFrame (with bias column as the first column)
        X_test: DataFrame (columns matching those of X_train)
        k: number of top features to select
    Returns: (top_k_indices, top_k_features, X_train_selected, X_test_selected)
            - top_k_indices: np.array, top-k feature indices. Please arrange the indices in descending order by weight (placing the index with the highest weight first, followed by the others).
            - top_k_features: names of the selected features
            - X_train_selected, X_test_selected: DataFrames with only the selected features.
    """
    weights_without_bias = w[1:]
    abs_weights = np.abs(weights_without_bias)
    sorted_indices = np.argsort(abs_weights)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_features = X_train.columns[1:][top_k_indices].tolist()
    
    selected_features = np.concatenate(([X_train.columns[0]], top_k_features))
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return top_k_indices, top_k_features, X_train_selected, X_test_selected

def normalize_features(X_train, X_test):
    """
    Standardize features: fit StandardScaler on X_train and transform both X_train and X_test.
    See lecture slides: you should fit the transformation on X_train only, 
       and then apply the transformation both to X_train and X_test.
    This produces normalized data  X_train_scaled and X_test_scaled respectived
    
    Parameters:
       X_train, X_test: pandas DataFrames with the original features before scaling (without the bias term).
    Returns:
        X_train_scaled, X_test_scaled: pandas DataFrames after scaling.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

def compute_condition_number(X):
    """
    Compute the condition number of matrix X, defined as the ratio of the largest 
    singular value to the smallest singular value.

    Parameters:
        X: n x d data matrix (Data Frame)

    Returns:
        float: The condition number (largest singular value / smallest singular value).
    """
    U, S, Vt = np.linalg.svd(X)
    return S.max() / S.min()
