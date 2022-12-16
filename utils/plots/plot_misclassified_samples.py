"""
Plots Misclassified samples

"""

from matplotlib import pyplot as plt
import numpy as np

def plot_MS(y_true, y_pred, path_to_results, path_to_images):
    """
    Plots and save missclassified samples
    
    y_true: numpy array of shape (num_of_images,2)
    y_pred: numpy array of shape (num_of_images,2)
    path_to_results: path where plot will be saved
    path_to_images: folder containing images

    """
    classes = np.unique(y_true[:,1])
    num_classes = len(classes)

    fig, axes = plt.subplots(num_classes,num_classes, figsize=(15,15))
    
    

    plt.savefig(path_to_results + "_CM")

if __name__ == "__main__":
    path_to_txt = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/histogram_new/multiclass/train/0/valid_svm.txt"
    y_pred = np.loadtxt(path_to_txt, dtype=str, delimiter=" ")

    print(y_pred)

    path_to_true = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train_multi.txt"
    y = np.loadtxt(path_to_true, dtype=str, delimiter=" ")
    print(y)

    pred_ids = y_pred[:,0].tolist()

    y_true = []
    for i in range(y.shape[0]):
        t = y[i,0]
        #print(t)
        if t in pred_ids:
            #print(y[i])
            y_true.append(y[i])

    y_true = np.array(y_true)
    print(y_true)
    print(y_pred.shape)
    print(y_true.shape)
    print(np.count_nonzero(y_true[:,0] == y_pred[:,0]))