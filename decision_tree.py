"""Pure Python Decision Tree Classifier.

Simple binary decision tree classifier. Splits are based on gini impurity.
API is a subset of the scikit-learn API.

Author: CS445 Instructor and ???
Version:

"""
from collections import namedtuple, Counter
import numpy as np

# -----------------------------------------------------------------------
# Your code should not depend on any external libraries other than numpy.
# You are free to add any private methods you like, but the
# public API must match the provided docstrings below.
# -----------------------------------------------------------------------

# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple(
    "Split",
    [
        "dim",
        "pos",
        "X_left",
        "y_left",
        "counts_left",
        "X_right",
        "y_right",
        "counts_right",
    ],
)


class Split(Split_):
    """
    Represents a possible split point during the decision tree
    creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        counts_left (Counter): label counts
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
        counts_right (Counter): label counts
    """

    def __repr__(self):
        result = "Split(dim={}, pos={},\nX_left=\n".format(self.dim, self.pos)
        result += repr(self.X_left) + ",\ny_left="
        result += repr(self.y_left) + ",\ncounts_left="
        result += repr(self.counts_left) + ",\nX_right=\n"
        result += repr(self.X_right) + ",\ny_right="
        result += repr(self.y_right) + ",\ncounts_right="
        result += repr(self.counts_right) + ")"

        return result


def split_generator(X, y, keep_counts=True):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy array with length num_samples
    :param keep_counts: Maintain counters (only useful for classification.)
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        counts_left = Counter()
        counts_right = Counter(y)

        # Get the indices in sorted order so we can sort both  data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        last_split = 0
        # Loop through the midpoints between each point in the
        # current dimension
        for index in range(1, X_sort.shape[0]):
            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                flipped_counts = Counter(y_sort[last_split:index])
                counts_left = counts_left + flipped_counts
                counts_right = counts_right - flipped_counts

                last_split = index
                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(
                    dim,
                    pos,
                    X_sort[0:index, :],
                    y_sort[0:index],
                    counts_left,
                    X_sort[index::, :],
                    y_sort[index::],
                    counts_right,
                )


class DecisionTreeClassifier:
    """
    A binary decision tree classifier for use with real-valued attributes.

    """

    def __init__(self, max_depth=None):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth (minimum is 1), None for no limit.
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array of samples with shape (num_samples, num_features)
        :param y: Numpy array of targets with length num_samples
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")

    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using either the the majority label or the mean value
        as the prediction.

        :param X:  Numpy array of samples with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predictions.
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")

    def score(self, X, y):
        """
        Calculate the accuracy of the decision tree on the provided data.

        :param X: Numpy array of samples with shape (num_samples, num_features)
        :param y: Numpy array of targets with length num_samples
        :return: A float representing the fraction of correct predictions.
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")

    # Trailing underscore indicates properties that are only available after fitting.
    @property
    def feature_importances_(self):
        """Return the feature importances based on the splits made in the tree.

        Feature importances are the weighted and normalized gini gains for each feature.

        gini gain is the reduction in gini impurity from a split:
            gini_gain = gini_node - (gini_left * n_node/n_left + gini_right * n_node/n_right)

        weighted: The contribution of each split is weighted by the number of samples split at that node.
             importanances[dim] += (n_node / n) * gini_gain

        normalized: The sum of all feature importances are scaled to sum to 1.

        :return: A numpy array with length num_features containing the
                 feature importances for each feature.
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        raise NotImplementedError("DecisionTreeClassifier is not implemented yet.")


class Node:
    """
    It will probably be useful to have a Node class.  In order to use the
    visualization code in draw_trees, the node class must have three
    attributes:

    Attributes:
        left:  A Node object or Null for leaves.
        right - A Node object or Null for leaves.
        split - A Split object representing the split at this node,
                or Null for leaves
    """

    def __init__(self, X, y, split=None):
        self.left = None
        self.right = None
        self.split = split
        # Feel free to add any other attributes you like.


def tree_demo():
    import draw_tree as draw_tree

    X = np.array([[0.88, 0.39], [0.49, 0.52], [0.68, 0.26], [0.57, 0.51], [0.61, 0.73]])
    y = np.array([1, 0, 0, 0, 1])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    draw_tree.draw_tree(X, y, clf)


if __name__ == "__main__":
    tree_demo()
