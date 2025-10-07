"""
Gradient descent for linear regression exercises.
"""
import numpy as np


class LinearRegression:
    """Linear regression model using gradient descent."""

    def __init__(self, w, lr=0.01):
        """Initialize the linear regression model.

        Parameters:
            w - Initial weight vector
            lr - The learning rate for gradient descent (eta)
        """
        self.lr = lr
        self.w = w.copy()

    def predict(self, x):
        """Returns h(x) = w^T x

        Parameters:
            x - d-dimensional numpy array (with bias value 1 at position 0)

        Returns:
            Prediction for input x
        """
        return self.w.T @ x

    def calc_loss(self, X, y):
        """Returns 0.5 * SSE for the data in X

        Parameters:
            X - n x d numpy array, n points dimensionality d (with 1's in col 0)
            y - length-n numpy array of target values

        Returns:
            Loss value (0.5 * sum of squared errors)
        """
        pass

    def batch_gd(self, X, y):
        """Perform one epoch of batch gradient descent and update weights.

        Parameters:
            X - n x d numpy array, n points dimensionality d (with 1's in col 0)
            y - length-n numpy array of target values
        """
        pass

    def stochastic_gd(self, X, y):
        """Perform one epoch of stochastic gradient descent and update weights.

        Parameters:
            X - n x d numpy array, n points dimensionality d (with 1's in col 0)
            y - length-n numpy array of target values
        """
        pass

    def fit(self, X, y, epochs, method="batch"):
        """Fit the model using the specified gradient descent method.

        Parameters:
            X - n x d numpy array, n points dimensionality d (with 1's in col 0)
            y - length-n numpy array of target values
            epochs - number of epochs to train
            method - "batch" for batch gradient descent or "sgd" for stochastic gradient descent

        Returns:
            numpy array of losses after each epoch
        """
        losses = np.zeros(epochs)

        for epoch in range(epochs):
            if method == "batch":
                self.batch_gd(X, y)
            elif method == "sgd":
                # Shuffle the data for each epoch
                indices = np.random.permutation(y.size)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                self.stochastic_gd(X_shuffled, y_shuffled)
            else:
                raise ValueError("Method must be 'batch' or 'sgd'")

            # Calculate and store loss after this epoch
            losses[epoch] = self.calc_loss(X, y)

        return losses


def train_mpg():
    import matplotlib.pyplot as plt

    X = np.load("mpg/X_mpg.npy")
    y = np.load("mpg/y_mpg.npy")

    # Normalize the columns so that they are mean 0 and unit variance
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # prepend a column of 1's
    ones_col = np.ones((X.shape[0], 1))  # column of zeros
    X = np.hstack([ones_col, X])

    # intialize random weights
    w = np.random.random(X.shape[1])

    model = LinearRegression(w, lr=0.001)
    losses = model.fit(X, y, 50, method="sgd")
    print(losses)
    plt.plot(losses)
    plt.show()


def worksheet_answers():
    """Print answers to the gradient descent activity."""

    # It is convenient to pre-prepend a column of 1's to facilitate
    # the bias weights.
    X = np.array([[1, 1, 2.0], 
                  [1, -2, 5], 
                  [1, 0, 1]])
    y = np.array([1, 6, 1])
    w = np.array([0, 1, 0.5])

    print("1a")
    model = LinearRegression(w, lr=0.01)
    print(model.predict(np.array([1, 2.0, 3.0])))

    print("\n1b")
    print(model.calc_loss(X, y))

    print("\n2a")
    model.batch_gd(X, y)
    print(model.calc_loss(X, y))

    print("\n2b")
    model2 = LinearRegression(w, lr=0.01)
    model2.stochastic_gd(X, y)
    print(model2.calc_loss(X, y))


if __name__ == "__main__":
    # main()
    # train_mpg()
    x = np.array([[1, 1, 2],
                [1, -2, 5],
                [1, 0, 1]])
    w = np.array([0, 1, 0.5])
    model = LinearRegression(w)
    print(model.predict(x))
