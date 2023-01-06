"""
Plots learning curves for cross entropy loss, balanced accuracy and mcc score for training and validation data

"""

from matplotlib import pyplot as plt
import numpy as np

def plot_LC(metric_score, path_to_results, path_to_images):

    """
    Plots and saves learning curves
    
    metric_score: numpy array of shape (metrics, 2, epochs)
    path_to_results: path where plot will be saved
    
    """
    num_metrics = metric_score.shape[0]
    metrics = ["CELoss", "Bal_Accu", "MCC"]
    
    fig, axes = plt.subplots(num_metrics, 1)
    
    fig.suptitle("Learning Curves")

    for met in range(num_metrics):

        # plot training curve
        train_line, = axes[met].plot(metric_score[met][0], color='blue', label='Training')

        # plot validation curve
        valid_line, = axes[met].plot(metric_score[met][1], color= 'orangered', label='Validation')

        # setup x-axis
        if met != num_metrics -1 :
            axes[met].set_xticks([])
        else:
            axes[met].set_xlabel("Epochs")

        # setup y-axis
        axes[met].set_ylabel(metrics[met])

    fig.legend(handles= [train_line, valid_line])

    plt.savefig(path_to_results + "_LC")
    plt.close()

if __name__ == "__main__":

    x = np.random.rand(3, 2, 10)
    x[0,0] = np.arange(1, 11)
    x[1, 1] = np.arange(11,21)
    
    print(x)
    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/code_test/"

    plot_LC(
        x, path_to_results, x
    )