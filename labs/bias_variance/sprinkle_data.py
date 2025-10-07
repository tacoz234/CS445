""" Artificial Data Source for Bias/Variance Experiments

author: Nathan Sprague
version: 1/15/20

"""

import numpy as np
import matplotlib.pyplot as plt


class SprinkleDataSource:

    def __init__(self, variance=.07):
        self.variance = variance

    def gen_data(self, num, seed=None):
        if seed:
            np.random.seed(seed)
        x = np.random.random(num)
        y_no_noise = self.true_fun(x)
        y = y_no_noise + np.random.normal(0, np.sqrt(self.variance), num)
        return x, y

    def true_fun(self, x: np.ndarray):
        return np.sin(x * 7) + 1.2 * np.exp(-((x - .6) / .1) ** 2) + 4 * x ** 2 + 3.0


def save_data(source, num, filename):
    x, y = source.gen_data(num)
    np.savetxt(filename, np.transpose((x, y)), fmt='%.5f', delimiter=',')
    return x, y


if __name__ == "__main__":
    dg = SprinkleDataSource()
    x, y = save_data(dg, 30, 'sprinkles_tmp.csv')
    plt.plot(x, y, '*')
    xs = np.arange(0, 1, .01)
    ys = dg.true_fun(xs)
    # plt.plot(xs, ys)
    plt.show()
