"""
A utility script which contains tools useful for notebooks and analysis when running
AI lit solutions.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_and_evaluate(university, model_name, evaluation_name):
    """
    This is a convenience method to train, if needed, and evaluate a university.
    :param university: The university to potentially train and evaluate.
    :param model_name: The name of the model supported in this event.
    :param evaluation_name: The name of the evaluation produced in this event.
    :return: The accuracy, F1-measure and confusion matrix of the evaluation.
    """
    latest_run = university.get_latest_run_dir()
    if latest_run is None:
        latest_run = university.train()
    targets, predictions = university.get_or_perform_evaluation(latest_run, evaluation_name)

    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    print("Accuracy:", accuracy)
    print("F1:", f1)

    cnf_matrix = confusion_matrix(targets, predictions)
    plot_confusion_matrix(cnf_matrix, classes=university.subjects, normalize=True,
                          title=model_name + '-' + evaluation_name + ' Confusion Matrix')
    return accuracy, f1, cnf_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plot the provided confusion matrix for an AI lit analysis notebook.
    :param cm: The confusion matrix to plot.
    :param classes: The set of classes we are analysing.
    :param normalize: True to normalize the CM plot, False to not.
    :param title: The title of the plot.
    :param cmap: The confusion matrix theme.
    """
    np.set_printoptions(precision=2)
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
