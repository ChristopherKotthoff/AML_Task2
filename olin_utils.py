from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def confMat(pred_labels, true_labels, visualize = False):
    '''
    Takes two arrays of ints that represent classes.
    It is assumed that the classes are labeled ascending from 0.

    The outputs are indexed by: [true_label, predicted_label]

    Input:
        pred_labels (np.array of ints): The predicted labels.
        true_labels (np.array of ints): The true labels - assumed to contain all the existing classes.

    Output: 
    1. conf_mat (2D array of  ints): for a given case [true_label, predicted_label] this will return the amount of (miss-)classifications of this type
    2. index_conf_mat (2D array of lists): for a given case [true_label, predicted_label] this will return a list of the indeces that fall into this type of (miss-)classification
    '''

    nr_classes = len(np.unique(true_labels))

    conf_mat = np.zeros((nr_classes, nr_classes))
    index_conf_mat = [[[] for cell in range(nr_classes)] for row in range(nr_classes)]
    for i in range(len(pred_labels)):
        pred_l = int(pred_labels[i])
        true_l = int(true_labels[i])
        conf_mat[true_l, pred_l] += 1
        index_conf_mat[true_l][pred_l].append(i)
    
    conf_mat = conf_mat.astype(int)

    if visualize:    
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot(values_format='.4g')
        plt.show()

    return conf_mat, np.array(index_conf_mat, dtype=object)