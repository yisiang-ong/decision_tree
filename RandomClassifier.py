import numpy as np
from numpy.random import default_rng


class RandomClassifier:
    def __init__(self, random_generator=default_rng()):
        self.random_generator = random_generator
        self.unique_y = []

    def fit(self, x, y):
        """ Fit the training data to the classifier.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)
            y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.unique_y = list(set(y))

    def predict(self, x):
        """ Perform prediction given some examples.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
            y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """
        random_indices = self.random_generator.integers(0, len(self.unique_y), len(x))
        y = np.array(self.unique_y)
        return y[random_indices]
