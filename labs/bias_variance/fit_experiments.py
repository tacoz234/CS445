""" Polynomial Line Data Fitting and Bias Variance Experiments

author: Nathan Sprague
version: 1/15/19

"""

import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:

    def fit(self, x, y, degree):
        self.x = x
        self.y = y
        self.degree = degree
        coefficients = np.polyfit(x, y, degree)
        self.poly = np.poly1d(coefficients)

    def plot(self, x=None, y=None, show_error=False):
        if self.x is None:
            print("Please fit some data first!")
        else:
            granularity = 100
            xs = np.linspace(np.min(self.x), np.max(self.x), granularity)
            ys = self.poly(xs)
            plt.plot(xs, ys, 'b')
            if show_error:
                for (x_val, y_val) in zip(x, y):
                    pred = self.poly(x_val)
                    plt.plot([x_val, x_val], [pred, y_val], 'r')

    def predict(self, x):
        if self.x is None:
            print("Please fit some data first!")
        else:
            return self.poly(x)

    def evaluate(self, x, y):
        return np.mean(np.sum((self.poly(x) - y) ** 2))


def bias_variance_experiment(num_trials, train_size, degree, source, display=True):
    granularity = 100
    predictions = np.empty((num_trials, granularity))
    model = PolynomialRegression()
    for i in range(num_trials):
        x, y = source.gen_data(train_size)
        model.fit(x, y, degree)
        xs = np.linspace(.1, .9, granularity)
        ys = model.predict(xs)
        predictions[i, :] = ys
        if display:
            plt.plot(xs, ys, 'b', alpha=.1)

    xs = np.linspace(.1, .9, granularity)
    ys = source.true_fun(xs)

    mean_pred = np.mean(predictions, axis=0)
    avg_sqrd_bias = np.mean((np.mean(predictions, axis=0) - ys) ** 2)
    avg_variance = np.mean(np.var(predictions, axis=0))

    if display:
        plt.plot(xs, ys, 'r', lw=2, label='$f(x)$')
        plt.plot(xs, mean_pred, color='springgreen', lw=2,
                 label='$E[(\\hat{f}(x)]$')
        plt.legend()
        plt.show()
        print("mean squared bias: {:.4f}".format(avg_sqrd_bias))
        print("mean variance: {:.4f}".format(avg_variance))
    return avg_sqrd_bias, avg_variance


def tune_experiment(num_trials, train_size, min_degree, max_degree, source):
    biases = []
    variances = []
    degrees = range(min_degree, max_degree)
    for degree in degrees:
        avg_sqrd_bias, avg_variance = bias_variance_experiment(num_trials, train_size, degree, source, display=False)
        biases.append(avg_sqrd_bias)
        variances.append(avg_variance)
    biases = np.array(biases)
    variances = np.array(variances)
    plt.plot(degrees, biases)
    plt.plot(degrees, variances)
    plt.plot(degrees, biases + variances)
    plt.legend(["mean sqrd bias", "variance", "sum"])
    plt.xlabel("polynomial degree")
    plt.show()


if __name__ == "__main__":
    ds = datasource.DataSource()
    bias_variance_experiment(100, 30, 3, ds, True)
