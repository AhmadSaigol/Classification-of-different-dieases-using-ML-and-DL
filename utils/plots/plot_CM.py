"""
Plots Confusion Matrix

"""

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np

def plot_CM(y_true, y_pred, path_to_results):
    """
    Plots and save Confusion Matrix
    
    y_true: numpy array of shape (num_of_images,)
    y_pred: numpy array of shape (num_of_images, )
    path_to_results: path where plot will be saved

    """

    ConfusionMatrixDisplay.from_predicitons(y_pred=y_pred, y_true =y_true)

    plt.savefig(path_to_results + "_CM")



if __name__ == "__main__":
    """
    # binary
    y_true = np.array(["aa", "a", "ab", "b", "ac", "a"])
    y_pred = np.array(["aa", "b", "ab", "b", "ac", "b"])
    """                    
    
     
    # multi
    y_true = np.array(["aa", "a", "ab", "b", "ac", "c"])
    y_pred = np.array(["aa", "b", "ab", "c", "ac", "b"])


    plot_CM(
        y_true=y_true,
        y_pred=y_pred,
        path_to_results="/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"
    )                 

