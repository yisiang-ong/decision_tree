import numpy as np


class Evaluator:
    def __init__(self, y_prediction, y_gold, class_labels):
        """
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_labels (np.ndarray): a list of unique class labels. Defaults to the union of y_gold and y_prediction.
        """
        assert len(y_gold) == len(y_prediction)

        self.y_prediction = y_prediction
        self.y_gold = y_gold
        self.classes = class_labels

        # Compute the confusion matrix.
        # self.confusion_matrix: np.array, shape (C, C), where C is the number of classes.
        # Rows are ground truth per class, columns are predictions
        self.confusion = np.zeros((len(self.classes), len(self.classes)))
        for (i, label) in enumerate(self.classes):
            indices = (self.y_gold == label)
            predictions = self.y_prediction[indices]

            (unique_labels, counts) = np.unique(predictions, return_counts=True)

            frequency_dict = dict(zip(unique_labels, counts))

            for (j, class_label) in enumerate(self.classes):
                self.confusion[i, j] = frequency_dict.get(class_label, 0)

    def accuracy(self):
        """
        Compute the accuracy given the ground truth and predictions

        Returns:
            float : the accuracy
        """
        try:
            return np.sum(self.y_gold == self.y_prediction) / len(self.y_gold)
        except ZeroDivisionError:
            return 0.

    @staticmethod
    def accuracy_from_confusion(confusion):
        """
        Compute the accuracy given the confusion matrix
        Returns:
            float : the accuracy
        """

        if np.sum(confusion) > 0:
            return np.sum(np.diag(confusion)) / np.sum(confusion)
        else:
            return 0.

    @staticmethod
    def precision(confusion):
        """
        Compute the precision score per class given the ground truth and predictions
        Also return the macro-averaged precision across classes.
        Returns:
            tuple: returns a tuple (precisions, macro_precision) where
                - precisions is a np.ndarray of shape (C,), where each element is the
                  precision for class c
                - macro-precision is macro-averaged precision (a float)
        """

        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])

        # Compute the macro-averaged precision
        macro_p = 0.
        if len(p) > 0:
            macro_p = np.mean(p)

        return p, macro_p

    @staticmethod
    def recall(confusion):
        """
        Compute the recall score per class given the ground truth and predictions
        Also return the macro-averaged recall across classes.
        Returns:
            tuple: returns a tuple (recalls, macro_recall) where
                - recalls is a np.ndarray of shape (C,), where each element is the
                    recall for class c
                - macro-recall is macro-averaged recall (a float)
        """

        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])

        # Compute the macro-averaged recall
        macro_r = 0.
        if len(r) > 0:
            macro_r = np.mean(r)

        return r, macro_r

    @staticmethod
    def f1_score(confusion):
        """
        Compute the F1-score per class given the ground truth and predictions
        Also return the macro-averaged F1-score across classes.
        Returns:
            tuple: returns a tuple (f1s, macro_f1) where
                - f1s is a np.ndarray of shape (C,), where each element is the
                  f1-score for class c
                - macro-f1 is macro-averaged f1-score (a float)
        """

        (precisions, macro_p) = Evaluator.precision(confusion)
        (recalls, macro_r) = Evaluator.recall(confusion)

        # just to make sure they are of the same length
        assert len(precisions) == len(recalls)

        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1
        macro_f = 0.
        if len(f) > 0:
            macro_f = np.mean(f)

        return f, macro_f

    def get_confusion_by_label(self, label):
        """

        Args:
            label: the specific label we want to compute confusion matrix for

        Returns:
            confusion matrix by class
        """
        confusion = np.zeros((2, 2))

        confusion[0, 0] = len([i for (i, v) in enumerate(self.y_prediction) if self.y_gold[i] == self.y_gold[i]
                               and v == label])
        confusion[0, 1] = len(self.y_gold == label) - confusion[0, 0]
        confusion[1, 0] = len(self.y_prediction == label) - confusion[0, 0]
        confusion[1, 1] = len(self.y_gold != label) - confusion[1, 0]

        return confusion

    @staticmethod
    def get_confusion_by_label_from_confusion(label, confusion):
        """

        Args:
            label: the index of the specific label in classes, we want to compute confusion matrix for this specific label
            confusion: original multi-class confusion matrix
        Returns:
            confusion matrix by class
        """
        assert label < len(confusion)

        label_confusion = np.zeros((2, 2))
        label_confusion[0, 0] = confusion[label, label]
        label_confusion[0, 1] = np.sum(confusion[label, :]) - label_confusion[0, 0]
        label_confusion[1, 0] = np.sum(confusion[:, label]) - label_confusion[0, 0]
        label_confusion[1, 1] = np.sum(confusion) - label_confusion[0, 0] - label_confusion[0, 1] - label_confusion[1, 0]

        return label_confusion
