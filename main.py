import numpy as np
import os
from DecisionTree import DecisionTree as Tree
from RandomClassifier import RandomClassifier
from SplitData import SplitData
from Evaluator import Evaluator
import PlotTree


def load_data(data_path, data_type='clean'):
    """
    Load dataset from files
    Args:
        data_path: path of dataset; data_type: clean or noisy
        data_type:
    Return: numpy arrays data
    """
    if data_type == 'clean':
        data = np.loadtxt(os.path.join(data_path, 'clean_dataset.txt'))
    elif data_type == 'noisy':
        data = np.loadtxt(os.path.join(data_path, 'noisy_dataset.txt'))
    else:
        raise NameError("data_type should be either clean or noisy.")

    x = data[:, :-1]
    y = np.array(data[:, -1], dtype=int)
    classes = np.unique(y)

    return x, y, classes


def main():
    print("Loading clean dataset.")
    # loads data as matrix from file
    clean_x, clean_y, clean_classes = load_data('wifi_db/', 'clean')

    print("Loading noisy dataset.")
    # loads data as matrix from file
    noisy_x, noisy_y, noisy_classes = load_data('wifi_db/', 'noisy')

    # choose one dataset from the two options above
    x, y, classes = noisy_x, noisy_y, noisy_classes

    # --------------------------- Step 1 --------------------------- #
    #  generate an object helps to split data  #
    data_spliter = SplitData(seed=60012)

    '''
    # 1. divide dataset so train:test = 9:1
    x_train, x_test, y_train, y_test = data_spliter.split_dataset(x, y, test_proportion=0.1)
    # 2. train the decision tree
    tree1 = Tree.DecisionTree()
    tree1.fit(x_train, y_train)
    tree1_predictions = tree1.predict(x_test)
    print(tree1_predictions)
    # 2'. train the random classifier tree
    random_classifier = RandomClassifier(random_generator=default_rng(seed=60000))
    random_classifier.fit(x_train, y_train)
    random_predictions = random_classifier.predict(x_test)
    print(random_predictions)
    # 3. evaluate the un-pruned decision tree, compare it to the RandomClassifier
    evaluator_random = Evaluator(random_predictions, y_test, classes)
    evaluator_tree1 = Evaluator(tree1_predictions, y_test, classes)
    print("Ground truth:", y_test)
    print("Random:", random_predictions)
    print(evaluator_random.confusion_matrix)
    print("Tree1:", tree1_predictions)
    print(evaluator_tree1.confusion_matrix)
    '''

    # --------------------------- Step 2 --------------------------- #
    #  10-fold evaluation pre-prune tree  #

    evaluators2 = np.ndarray((10,), dtype=Evaluator)
    trees2 = np.ndarray((10,), dtype=Tree)

    for i, (train_indices, test_indices) in enumerate(data_spliter.train_test_k_fold(10, len(x))):
        print("Fold", i)
        # get the dataset from the correct splits
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        # train the tree and make predictions
        trees2[i] = Tree()
        trees2[i].fit(x_train, y_train)
        trees2_predictions = trees2[i].predict(x_test)

        # evaluate
        evaluators2[i] = Evaluator(trees2_predictions, y_test, classes)

    # 10 trees' averaged confusion matrix
    avg_confusion_matrix2 = np.mean([x.confusion for x in evaluators2], axis=0)
    avg_confusion_by_label2 = [Evaluator.get_confusion_by_label_from_confusion(label, avg_confusion_matrix2)
                               for label in range(len(classes))]
    print(f"the average of 10 confusion matrix: \n {avg_confusion_matrix2}")

    # 10 trees' averaged accuracy
    avg_accuracy_by_label2 = [Evaluator.accuracy_from_confusion(confusion) for confusion in avg_confusion_by_label2]
    accuracy_by_tree2 = np.array([x.accuracy() for x in evaluators2])
    print('\n'.join([f"Room{classes[x]}'s average accuracy: {avg_accuracy_by_label2[x]} " for x in range(len(classes))]),
          f"\n respective accuracies of 10 trees: \n {accuracy_by_tree2} \n "
          f"the mean accuracy of 10 trees: {accuracy_by_tree2.mean()},    std: {accuracy_by_tree2.std()}")

    # 10 trees' recall, precision and f1
    recall_by_label_by_tree2 = np.array([x.recall(x.confusion)[0] for x in evaluators2])
    avg_recall_by_tree2 = np.array([x.recall(x.confusion)[1] for x in evaluators2])
    precision_by_label_by_tree2 = np.array([x.precision(x.confusion)[0] for x in evaluators2])
    avg_precision_by_tree2 = np.array([x.precision(x.confusion)[1] for x in evaluators2])
    f1_by_label_by_tree2 = np.array([x.f1_score(x.confusion)[0] for x in evaluators2])
    avg_f1_by_tree2 = np.array([x.f1_score(x.confusion)[1] for x in evaluators2])
    print('\n'.join([f"Room{x}'s average recall rate: {y} "
                     for x, y in dict(zip(classes, np.mean(recall_by_label_by_tree2, axis=0))).items()]),
          f"\n average macro_recall rate: {avg_recall_by_tree2.mean()}")
    print('\n'.join([f"Room{x}'s average precision rate: {y} "
                     for x, y in dict(zip(classes, np.mean(precision_by_label_by_tree2, axis=0))).items()]),
          f"\n average macro_precision rate: {avg_precision_by_tree2.mean()}")
    print('\n'.join([f"Room{x}'s average f1 score: {y} "
                     for x, y in dict(zip(classes, np.mean(f1_by_label_by_tree2, axis=0))).items()]),
          f"\n average macro_f1_score: {avg_f1_by_tree2.mean()}")

    # 10 trees' depths
    print("depth of the 10 trees: " + str([x.depth for x in trees2]),
          f"\n average depth: {np.mean([x.depth for x in trees2])}")

    # plot confusion matrix
    ticks = ['Room ' + str(i+1) for i in range(len(classes))]
    PlotTree.plot_confus_mat(avg_confusion_matrix2, ticks, 'Confusion Matrix')

    # plotting depth against performance of pre-prune tree for analysis
    PlotTree.plot_tree_performance(np.array([x.depth for x in trees2]), accuracy_by_tree2, avg_f1_by_tree2)

    # --------------------------- Step 3 --------------------------- #
    #  10-9-cross-validation  #
    n_outer_folds = 10
    n_inner_folds = 9

    evaluators3 = np.ndarray((n_outer_folds, n_inner_folds), dtype=Evaluator)
    total_depths3 = np.zeros((n_outer_folds, n_inner_folds), dtype=int)
    accuracies3 = np.zeros((n_outer_folds, n_inner_folds), dtype=float)
    m_f1score3 = np.zeros((n_outer_folds, n_inner_folds), dtype=float)

    # Outer CV (10-fold)
    for i, (trainval_indices, test_indices) in enumerate(data_spliter.train_test_k_fold(n_outer_folds, len(x))):
        print("Fold", i)
        x_trainval = x[trainval_indices, :]
        y_trainval = y[trainval_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        # Pre-split data for inner cross-validation
        splits = data_spliter.train_test_k_fold(n_inner_folds, len(x_trainval))

        # Inner CV (I used 9-fold)
        for j, (train_indices, val_indices) in enumerate(splits):
            print("inner Folder", j)
            x_train = x_trainval[train_indices, :]
            y_train = y_trainval[train_indices]
            x_val = x_trainval[val_indices, :]
            y_val = y_trainval[val_indices]

            tree4 = Tree()
            tree4.fit(x_train, y_train)
            tree4.prune_and_update_tree(tree4.root, x_val, y_val)
            tree4_predictions = tree4.predict(x_test)

            total_depths3[i, j] = tree4.depth
            evaluators3[i, j] = Evaluator(tree4_predictions, y_test, classes)
            accuracies3[i, j] = evaluators3[i, j].accuracy()
            m_f1score3[i, j] = evaluators3[i, j].f1_score(evaluators3[i, j].confusion)[1]

    # 90 trees' averaged confusion matrix
    avg_confusion_matrix3 = np.mean([x.confusion for row in evaluators3 for x in row], axis=0)
    avg_confusion_by_label3 = [Evaluator.get_confusion_by_label_from_confusion(label, avg_confusion_matrix3)
                               for label in range(len(classes))]
    print(f"the average of 90 confusion matrix: \n {avg_confusion_matrix3}")

    # 90 trees' averaged accuracy
    avg_accuracy_by_label3 = [Evaluator.accuracy_from_confusion(x) for x in avg_confusion_by_label3]
    print('\n'.join([f"Room{classes[x]}'s average accuracy: {avg_accuracy_by_label3[x]} " for x in range(len(classes))]),
          f"the mean accuracy of 90 trees: {np.mean([x.accuracy() for row in evaluators3 for x in row])},"
          f"    std: {np.std([x.accuracy() for row in evaluators3 for x in row])}")

    # 90 trees' averaged depth
    print(f"average depth of the 90 trees: {np.mean(total_depths3)}")

    # 90 trees' averaged recall, precision, f1 and macro_f1
    print(f"(recall, macro_recall) rate from the average confusion matrix: \n {Evaluator.recall(avg_confusion_matrix3)}")
    print(f"(precision, macro_precision) rate from the average confusion matrix: \n {Evaluator.precision(avg_confusion_matrix3)}")
    print(f"(f1, macro_f1) from the average confusion matrix: \n {Evaluator.f1_score(avg_confusion_matrix3)}")
    print(f"average macro_f1_score of the 90 trees: {np.mean([x.f1_score(x.confusion)[1] for row in evaluators3 for x in row])}")

    # plot confusion matrix
    ticks = ['Room ' + str(i+1) for i in range(len(classes))]
    PlotTree.plot_confus_mat(avg_confusion_matrix3, ticks, 'Confusion Matrix')

    # plotting depth against performance of pre-prune tree for analysis
    PlotTree.plot_tree_performance(total_depths3, accuracies3, m_f1score3)


if __name__ == "__main__":
    main()
