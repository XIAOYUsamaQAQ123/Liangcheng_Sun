import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from scipy.special import expit as sigmoid
from sklearn.model_selection import KFold
import pickle



# Problem 5: Neural Networks on XOR

class XORNet(nn.Module):
    def __init__(self):
        """
        Initialize the layers of your neural network

        You should use nn.Linear
        """
        super(XORNet, self).__init__()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4, 1)
    
    def set_l1(self, w, b):
        """
        Set the weights and bias of your first layer
        @param w: (2,2) torch tensor
        @param b: (2,) torch tensor
        """
        self.l1.weight.data = w
        self.l1.bias.data = b
    
    def set_l2(self, w, b):
        """
        Set the weights and bias of your second layer
        @param w: (1,2) torch tensor
        @param b: (1,) torch tensor
        """
        self.l2.weight.data = w
        self.l2.bias.data = b
    
    def forward(self, xb):
        """
        Compute a forward pass in your network.  Note that the nonlinearity should be F.relu.
        @param xb: The (n, 2) torch tensor input to your model
        @return: an (n, 1) torch tensor
        """
        output = F.relu(self.l1(xb))
        output = self.l2(output)
        return output


def fit(net, optimizer,  X, Y, n_epochs):
    """ Fit a net with BCEWithLogitsLoss.  Use the full batch size.
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param X: an (N, D) torch tensor
    @param Y: an (N, 1) torch tensor
    @param n_epochs: int, the number of epochs of training
    @return epoch_loss: Array of losses at the beginning and after each epoch. Ensure len(epoch_loss) == n_epochs+1
    """
    loss_fn = nn.BCEWithLogitsLoss() #note: input to loss function needs to be of shape (N, 1) and (N, 1)
    with torch.no_grad():
        epoch_loss = [loss_fn(net(X), Y)]
    for _ in range(n_epochs):
        #TODO: compute the loss for X, Y, it's gradient, and optimize
        #TODO: append the current loss to epoch_loss
        optimizer.zero_grad()
        y_pred = net(X)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return epoch_loss

# Problem 4: Gradient Boosting

# Reference for RegressionTree implementation
class RegressionTree:
    """
    A wrapper for the DecisionTreeRegressor from scikit-learn.
    """
    def __init__(self):
        self.tree = DecisionTreeRegressor(max_depth=3)
    
    def fit(self, X, y):
        """
        Fit the regression tree on the given data.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target values.
        """
        self.tree.fit(X, y)
    
    def predict(self, X):
        """
        Predict using the fitted regression tree.
        
        Parameters:
        X (array-like): Feature matrix.
        
        Returns:
        array-like: Predicted values.
        """
        return self.tree.predict(X)
    

# TODO: complete compute_functional_gradient and fit_next_tree functions in this class
class gradient_boosted_trees:
    """
    Gradient Boosted Trees for regression or binary classification using a RegressionTree learner.
    """
    def __init__(self, step_size, number_of_trees, loss_function):
        """
        Initialize the gradient boosted trees model.
        
        Parameters:
        step_size (float): The constant step size (learning rate) for updating predictions.
        number_of_trees (int): The total number of trees to use in boosting.
        loss_function (str): The loss function to use; either "least_squares" or "logistic_regression".
        """
        self.step_size = step_size
        self.number_of_trees = number_of_trees
        self.loss_function = loss_function  
        self.trees = []

    def compute_functional_gradient(self, y, f_old):
        """
        Compute the negative functional gradient of the loss function.
        
        Parameters:
        y (array-like): True target values.
        f_old (array-like): Current predictions  (using the tree ensemble already constructed).
        
        Returns:
        array-like: The computed negative functional gradient.
        """
        f = self.loss_function
        if f == "least_squares":
            return y - f_old
        elif f == "logistic_regression":
            return y - sigmoid(f_old)
        else:
            return -1

    def fit_next_tree(self, X, y, f_old):
        """
        Fit the next tree on the negative functional gradient and update predictions.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): True target values.
        f_old (array-like): Current predictions (using the tree ensemble already constructed).
        
        Returns:
        array-like: Updated predictions after adding the contribution of the fitted tree.
        """
        gradient = self.compute_functional_gradient(y, f_old)
        tree = RegressionTree()
        tree.fit(X, gradient)
        self.trees.append(tree)
        update = tree.predict(X)
        return f_old + self.step_size * update

    def predict(self, X, num_trees=None):
        """
        Make predictions using the gradient boosted trees model.
        
        Parameters:
        X (array-like): Feature matrix.
        num_trees (int, optional): Number of trees to use for prediction. If None, use all trees.
        
        Returns:
        array-like: The tree ensemble prediction.
        """
        if num_trees is None:
            num_trees = len(self.trees)
        f = np.full((X.shape[0],), 0.0)
        for i in range(num_trees):
            f += self.step_size * self.trees[i].predict(X)
        return f

    def fit(self, X, y):
        """
        Fit the gradient boosted trees model on the given data.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target values.
        """
        f = np.full(y.shape, 0.0)
        for i in range(self.number_of_trees):
            f = self.fit_next_tree(X, y, f)

def cross_validation(X, y, loss_function, step_sizes, number_of_trees, kf):
    """
    Perform cross-validation to compute the average accuracy for each candidate step size.
    
    This function accepts a pre-generated cross-validation splitter (such as a KFold instance)
    to avoid additional randomness in the splitting process.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target values.
    loss_function (str): Loss function to use ("least_squares" or "logistic_regression").
    step_sizes (list): List of candidate step sizes.
    number_of_trees (int): Number of trees to use in boosting.
    kf: A cross-validation splitter instance (e.g., KFold) created outside the function.
    
    Returns:
    cv_accuracies (list): A list of average cross-validation accuracies corresponding to each step size.
    """

    cv_accuracies = []
    for step in step_sizes:
        accuracies = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = gradient_boosted_trees(step, number_of_trees, loss_function)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            if loss_function == "logistic_regression":
                y_pred = (sigmoid(preds) > 0.5).astype(int)
            else:
                y_pred = np.round(preds).astype(int)

            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)
        cv_accuracies.append(np.mean(accuracies))
    return cv_accuracies

def main():
    """
    Main procedure that loads training data, selects the best step_size via cross-validation,
    trains the final model, and evaluates it using the ModelTester.
    """
    # Load training data from CSV.
    # Assumes that 'train.csv' has a header row and the last column is the target.
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    
    # Candidate step sizes.
    step_sizes = [0.01, 0.1, 1, 10]

    for loss_choice in ["logistic_regression", "least_squares"]:
        print("Loss function:", loss_choice)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_accuracies = cross_validation(X_train, y_train, loss_function=loss_choice,
                                             step_sizes=step_sizes, number_of_trees=10,
                                             kf=kf)
        latex_rows = []
        for step, acc in zip(step_sizes, cv_accuracies):
            print(f"  Step size: {step:5.2f} CV Accuracy: {acc:.2f}")
            latex_rows.append(f"{step:5.2f} & {acc:.2f} \\\\")
            best_index = np.argmax(cv_accuracies)
            best_step_size = step_sizes[best_index]
            print("  Best step size: {}".format(best_step_size))
    
        # Train the final model on the full training data.
        final_model = gradient_boosted_trees(step_size=best_step_size,
                                                     number_of_trees=10,
                                                     loss_function=loss_choice)
        final_model.fit(X_train, y_train)
    
        # Save the final model to a file named after the loss function.
        model_filename = f"final_model_{loss_choice}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(final_model, f)


    
        # Print LaTeX table with step sizes, CV accuracies, and the best step size.
        print("\n\\begin{table}[h]")
        print("\\centering")
        escaped_loss = loss_choice.replace("_", "\\_")
        print(f"\\caption{{Results for loss function: {escaped_loss}}}")
        print("\\begin{tabular}{cc}")
        print("\\hline")
        print("Step Size & CV Accuracy \\\\")
        print("\\hline")
        for row in latex_rows:
            print(row)
            print("\\hline")
        print(f"\\multicolumn{{2}}{{c}}{{Best step size: {best_step_size:5.2f}}} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")    
        print("\n")

# Run this locally to generate pickle files to upload to autograder, comment out when submitting to autograder
if __name__ == "__main__":
    main()