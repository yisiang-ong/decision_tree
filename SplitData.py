import numpy as np
from numpy.random import default_rng


class SplitData:
    def __init__(self, seed):
        self.seed = seed                        # make the random generator traceable
        self.rg = default_rng(seed)             # random_generator (np.random.Generator): A random generator

    def split_dataset(self, x, y, test_proportion):
        """ Split dataset into training and test sets, according to the given
            test set proportion.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)
            y (np.ndarray): Class labels, numpy array with shape (N,)
            test_proportion (float): the desired proportion of test examples (0.0-1.0)

        Returns:
            tuple: returns a tuple of (x_train, x_test, y_train, y_test)
                   - x_train (np.ndarray): Training instances shape (N_train, K)
                   - x_test (np.ndarray): Test instances shape (N_test, K)
                   - y_train (np.ndarray): Training labels, shape (N_train, )
                   - y_test (np.ndarray): Test labels, shape (N_test, )
        """

        shuffled_indices = self.rg.permutation(len(x))
        n_test = round(len(x) * test_proportion)
        n_train = len(x) - n_test
        x_train = x[shuffled_indices[:n_train]]
        y_train = y[shuffled_indices[:n_train]]
        x_test = x[shuffled_indices[n_train:]]
        y_test = y[shuffled_indices[n_train:]]
        return x_train, x_test, y_train, y_test

    def k_fold_split(self, n_splits, n_instances):
        """ Split n_instances into n mutually exclusive splits at random.

        Args:
            n_splits (int): Number of splits
            n_instances (int): Number of instances to split

        Returns:
            list: a list (length n_splits). Each element in the list should contain a
                numpy array giving the indices of the instances in that split.
        """

        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = self.rg.permutation(n_instances)

        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, n_splits)

        return split_indices

    def train_test_k_fold(self, n_folds, n_instances):
        """ Generate train and test indices at each fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple)
                with two elements: a numpy array containing the train indices, and another
                numpy array containing the test indices.
        """

        # split the dataset into k splits
        split_indices = self.k_fold_split(n_folds, n_instances)

        folds = []
        for k in range(n_folds):
            # pick k as test
            test_indices = split_indices[k]

            # combine remaining splits as train
            # this solution is fancy and worked for me
            # feel free to use a more verbose solution that's more readable
            train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

            folds.append([train_indices, test_indices])

        return folds

    def train_val_test_k_fold(self, n_folds, n_instances):
        """ Generate train and test indices at each fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple)
                with three elements:
                - a numpy array containing the train indices
                - a numpy array containing the val indices
                - a numpy array containing the test indices
        """

        # split the dataset into k splits
        split_indices = self.k_fold_split(n_folds, n_instances)

        folds = []
        for k in range(n_folds):
            # pick k as test, and k+1 as validation (or 0 if k is the final split)
            test_indices = split_indices[k]
            val_indices = split_indices[(k + 1) % n_folds]

            # concatenate remaining splits for training
            train_indices = np.zeros((0,), dtype=np.int)
            for i in range(n_folds):
                # concatenate to training set if not validation or test
                if i not in [k, (k + 1) % n_folds]:
                    train_indices = np.hstack([train_indices, split_indices[i]])

            folds.append([train_indices, val_indices, test_indices])

        return folds
