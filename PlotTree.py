import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


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


def print_tree(root_node):
    """
    Print the nodes of the tree according to each depth.
    """
    cur_depth = 0
    nodes = [root_node]
    for node in nodes:
        if node.depth > cur_depth:
            print()
            cur_depth += 1
        print(f'X{node.attribute}<={node.value}  ', end='') if not node.is_leaf else print(
            f'label:{node.major_label}  ', end='')
        if not node.is_leaf:
            nodes.append(node.left)
            nodes.append(node.right)


def _get_num_leafs(node, num_leafs):
    '''
    Retrieve the total number of leafs in the tree.
    '''
    # recurse to find each left and right node if the node is not a leaf
    if not node.is_leaf:
        return _get_num_leafs(node.left, num_leafs) + _get_num_leafs(node.right, num_leafs)
    # if it is a leaf node, return 1 for theat leaf node and the above code will sum every leaf node
    else:
        return 1


def _plot_node(node_txt, centerPt, parentPt):
    '''
    function that plot node.
    '''
    # this is ax1 plot on draw_decision_tree plot
    draw_decision_tree.ax1.annotate(node_txt, xy=parentPt, xycoords='axes fraction',
                                    xytext=centerPt, textcoords='axes fraction',
                                    va='center', ha='center', bbox=dict(boxstyle="round", fc="white", edgecolor='blue'), arrowprops=dict(arrowstyle="<-", color='red'))


def _plot_tree(tree, parentPt, node_txt, depth):
    '''
    Plot the node using _plot_node_function, displaying the decision node and leaf node accordingly.
    '''

    # calculate number of leafs in the tree
    num_leafs = _get_num_leafs(tree, 0)
    depth = float(depth)

    # plot node based on node status, i.e a decision node or leaf node
    if tree.is_leaf:
        leaf_str = f"room {tree.label}"
        _plot_tree.x_off += 1./_plot_tree.total_w
        _plot_node(leaf_str, (_plot_tree.x_off, _plot_tree.y_off), parentPt)
    else:
        node_str = f"x{tree.attribute} <= {tree.value}"
        # set coords (a tuple)
        centerPt = (_plot_tree.x_off + (1.+float(num_leafs)) /
                    2./_plot_tree.total_w, _plot_tree.y_off)
        _plot_node(node_str, centerPt, parentPt)
        _plot_tree.y_off -= 1./_plot_tree.total_d

    # set a condition to stop recursive plotting
    if tree.is_leaf:
        return
    else:
        # recursively plot left and right node
        parentPt = centerPt
        _plot_tree(tree.left, parentPt, node_txt, depth-1)
        _plot_tree(tree.right, parentPt, node_txt, depth-1)

    _plot_tree.y_off += 1./_plot_tree.total_d


def draw_decision_tree(tree, depth):
    '''
    Create a canvas and plot the tree.
    '''
    figure = plt.figure(figsize=(40, 20), facecolor='white')
    figure.clf()
    axprops = dict(xticks=[], yticks=[])
    draw_decision_tree.ax1 = plt.subplot(111, frameon=False, **axprops)
    # determine total width and depth of the tree
    # set initial num_leafs to 0.
    _plot_tree.total_w = float(_get_num_leafs(tree, 0))
    _plot_tree.total_d = float(depth)
    # set x distance offset between each node
    _plot_tree.x_off = -0.5/_plot_tree.total_w
    _plot_tree.y_off = 1.
    _plot_tree(tree, (0.5, 1.), '', depth)

    plt.show()


if __name__ == "__main__":
    import DecisionTree as Tree
    clean_x, clean_y, clean_class = load_data('wifi_db/', 'clean')
    tree = Tree.DecisionTree()
    tree.fit(clean_x, clean_y)
    dep = tree.depth
    # print(dep)
    # print(_get_num_leafs(clean_tree, 0))
    print_tree(tree.root)  # print tree in python
    draw_decision_tree(tree.root, dep)  # Matplotlib Visualization



def plot_confus_mat(mat, ticks, title="Confusion Matrix"):
    """Display the confusion result after evaluating a decision tree
    Args:
        mat: Confusion matrix generated previously
        ticks: String labels of class for the data
        title: Title of the figure
    """
    plt.figure()
    plt.imshow(mat + 0.3, norm=matplotlib.colors.LogNorm(), cmap='Blues')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(ticks)))
    plt.xticks(num_local, ticks, rotation=45)
    plt.yticks(num_local, ticks)
    plt.ylabel('True label', fontdict={'size': 12})
    plt.xlabel('Predicted label', fontdict={'size': 12})
    thresh = mat.max() / 2.
    for i in range(4):
        for j in range(4):
            plt.text(j, i, '{:d}'.format(int(mat[i, j])), horizontalalignment="center",
                     color="white" if mat[i, j] > thresh else "black")
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.show()
    return


def plot_tree_performance(total_depth, accuracies, m_f1score):
    # plotting depth against performance of pre-prune tree for analysis
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.axis([5., 25., 0.5, 1.])
    ax1.scatter(total_depth, accuracies)
    ax1.axhline(y=np.mean(accuracies), color='r', linestyle='-', label='mean accuracies')
    ax1.axvline(x=np.mean(total_depth), color='b', linestyle='-', label='mean depth')
    ax1.set_xlabel("Depth")
    ax1.set_ylabel("Accuracies")
    ax1.legend(loc='lower right')
    ax2.axis([5., 25., 0.5, 1.])
    ax2.scatter(total_depth, m_f1score)
    ax2.axhline(y=np.mean(m_f1score), color='r', linestyle='-', label='mean f1 score')
    ax2.axvline(x=np.mean(total_depth), color='b', linestyle='-', label='mean depth')
    ax2.set_xlabel("Depth")
    ax2.set_ylabel("F1-score")
    ax2.legend(loc='lower right')
    plt.show()
    return
