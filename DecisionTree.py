"""
Date: 23rd Oct 2021
Author:
Project: Decision Tree, coursework1, Intro2ML
"""
import numpy as np
from Evaluator import Evaluator


class Node:
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=None, label=None, major_label=None,
                 depth=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label
        self.major_label = major_label
        self.depth = depth


class DecisionTree:
    def __init__(self):
        self.root = None
        self.depth = 0

    # calculate the entropy of a given set
    @staticmethod
    def _calc_shannon_ent(y):
        """
        Args:
            y: np.ndarray ([y1, y2, ..., yn]), where yi means the label of i-th sample

        Returns:
            Shannon entropy of y
        """
        label_set, label_counts = np.unique(y, return_counts=True)
        label_prob = (1.0 / label_counts.sum()) * label_counts
        shannon_ent = np.sum(-1. * label_prob * np.log2(label_prob))
        return shannon_ent

    @staticmethod
    def _vote_label(y):
        """
        get most common label given dataset
        Args:
            y: np.array, label set

        Returns:
            int, the major label
        """
        label, counts = np.unique(y, return_counts=True)
        return label[np.argmax(counts)]

    @staticmethod
    def _split_data_by_node(node, x, y):
        """
        split the dataset according to the node criterion
        Args:
            node: (Node) current criterion node
            x: (np.ndarray) feature dataset before splitting
            y: (np.array) label set before splitting

        Returns:
            tuple(tuple(left_feature_set, left_label_set), tuple(right_feature_set, right_label_set))
        """
        return (x[x[:, node.attribute] <= node.value], y[x[:, node.attribute] <= node.value]), \
               (x[x[:, node.attribute] > node.value], y[x[:, node.attribute] > node.value])

    def _get_tree_depth(self, root):
        if root is None:
            return 0
        elif root.is_leaf:
            return 1
        else:
            return 1 + max(self._get_tree_depth(root.left), self._get_tree_depth(root.right))

    def _find_split_on_single_cts_attr(self, x, y, attr_index):  # x=dataset[:, :-1], y=dataset[:, -1]
        """
        Args:
            x: np.ndarray, feature matrix
            y: np.array, labels
            attr_index: index of the specific attribute to be considered (int)
        Returns:
            (float, float): (the optimal split point on that attribute, Information gain with this split)
        """
        assert len(x) == len(y) > 0
        assert len(x[1]) > attr_index

        total_samples_number = len(x)
        target_x = x[:, attr_index]
        base_entropy = self._calc_shannon_ent(y)  # initial entropy H(S_{all})
        sorted_attr = np.sort(np.unique(target_x))

        remain_entropy = base_entropy
        res_split_point = None

        for i in range(1, len(sorted_attr)):
            # float == float, not suitable, to be corrected
            if np.unique(y[np.logical_or(target_x == sorted_attr[i - 1], target_x == sorted_attr[i])]).size == 1:
                continue

            cur_split_point = 0.5 * (sorted_attr[i - 1] + sorted_attr[i])
            left_index = target_x <= cur_split_point
            right_index = target_x > cur_split_point
            left_entropy = self._calc_shannon_ent(y[left_index])
            right_entropy = self._calc_shannon_ent(y[right_index])

            cur_entropy = (1.0 / total_samples_number) * (
                    sum(left_index) * left_entropy + sum(right_index) * right_entropy)

            if cur_entropy < remain_entropy:
                remain_entropy, res_split_point = cur_entropy, cur_split_point

        return res_split_point, base_entropy - remain_entropy

    def _find_split(self, x, y):  # x=dataset[:, :-1], y=dataset[:, -1]
        """

        Args:
            x: np.ndarray, feature matrix
            y: np.array, labels

        Returns:
            (int, float, float): (the index of optimal attribute to use,
            the optimal split point on that attribute, Information gain with this split)
        """
        assert len(x) == len(y)

        split_res = []
        for attr_index in range(x.shape[1]):
            split_res.append(self._find_split_on_single_cts_attr(x, y, attr_index))

        optimal_attr = np.array(split_res)[:, 1].argmax()
        return optimal_attr, split_res[optimal_attr][0], split_res[optimal_attr][1]

    def fit(self, x, y, depth=0):  # x=dataset[:, :-1], y=dataset[:, -1]
        """
        create decision tree
        """
        assert len(x) == len(y) > 0

        # if all samples have the same label
        if np.unique(y).size == 1:
            leaf_node = Node(is_leaf=True, label=int(y[0]), depth=depth, major_label=int(y[0]))
            self.root, self.depth = leaf_node, depth
            return leaf_node, depth
        else:
            # find split point
            split_res = self._find_split(x, y)
            node = Node(attribute=split_res[0], value=split_res[1], is_leaf=False, depth=depth,
                        major_label=int(self._vote_label(y)))
            ((left_x, left_y), (right_x, right_y)) = self._split_data_by_node(node, x, y)
            l_branch, l_depth = self.fit(left_x, left_y, depth + 1)
            r_branch, r_depth = self.fit(right_x, right_y, depth + 1)
            node.left = l_branch
            node.right = r_branch
            self.root, self.depth = node, max(l_depth, r_depth)
            return node, max(l_depth, r_depth)

    def _predict_one_sample(self, root, sample):
        """
        predict label of a specific sample using given tree
        Args:
            root: (Node) root node of the decision tree
            sample: (np.array) sample features
        Return:
            y: predicted label of this sample
        """
        assert root is not None

        if root.is_leaf:
            return root.major_label

        if sample[root.attribute] < root.value:
            y = self._predict_one_sample(root.left, sample)
        else:
            y = self._predict_one_sample(root.right, sample)
        return y

    def predict(self, x):
        """
        predict label from whole test data
        Args:
            x: (np.ndarray) the feature-dataset that needs to be predicted.
        Return:
            y: (np.array) labels predicted
        """
        y = np.array([None] * len(x))
        n, _ = x.shape
        for i in range(n):
            temp_sample = x[i, :]
            y[i] = self._predict_one_sample(self.root, temp_sample)
        return y

    def _prune_tree(self, root, x_valid, y_valid):
        """ attempt to Bottom-up prune tree using validation data

        Args:
            root:
            x_valid:
            y_valid:

        Returns:

        """
        if not root or root.is_leaf:
            return

        if not root.left.is_leaf:
            self._prune_tree(root.left, x_valid, y_valid)

        if not root.right.is_leaf:
            self._prune_tree(root.right, x_valid, y_valid)

        if root.right.is_leaf and root.left.is_leaf:
            y_predict_before_prune = self.predict(x_valid)
            accuracy_before = Evaluator(y_predict_before_prune, y_valid,
                                        np.unique(np.concatenate((y_valid, y_predict_before_prune)))).accuracy()
            root.is_leaf = True
            y_predict_after_prune = self.predict(x_valid)
            accuracy_after = Evaluator(y_predict_after_prune, y_valid,
                                       np.unique(np.concatenate((y_valid, y_predict_after_prune)))).accuracy()

            if accuracy_after < accuracy_before:
                root.is_leaf = False

    def prune_and_update_tree(self, root, x_valid, y_valid):
        self._prune_tree(root, x_valid, y_valid)
        self.depth = self._get_tree_depth(self.root)
