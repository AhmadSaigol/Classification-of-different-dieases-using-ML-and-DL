"""
Plots Misclassified samples

"""

from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg


def plot_MS(y_true, y_pred, path_to_results, path_to_images):
    """
    Plots and save missclassified samples
    
    y_true: numpy array of shape (num_of_images,2)
    y_pred: numpy array of shape (num_of_images,2)
    path_to_results: path where plot will be saved
    path_to_images: folder containing images


    Only works with (num_samples)^2 = whole number

    """
    num_samples_to_plot = 4

    classes = np.unique(y_true[:,1])

    if np.sqrt(num_samples_to_plot) %1 !=0:
        raise ValueError("Currently, this function only supports those number of samples whose sqaure is a whole number")

    
    
    cm = create_dict(classes)

    for true_label in classes:
        
        # get all images for a class in y_true
        pos = y_true[np.where(y_true[:,1] == true_label)]
        
        pred_classes = classes.tolist()
      
        for sample in pos:
            
            # find given image in y_pred
            img_id = y_pred[np.where(y_pred[:,0] == sample[0])]
            
            # store img_ids in their respective col
            for pred_label in pred_classes:
                if img_id[0,1] == pred_label:
                    cm[true_label][pred_label].append(img_id[0,0])

            # check whether there are requried number of samples in each col
            for l in cm[true_label].keys():
                
                if l in pred_classes and len(cm[true_label][l]) == num_samples_to_plot:
                    pred_classes.remove(l)
                
            if not len(pred_classes):
                break
    
    plot_CM_images(cm, num_samples_to_plot, path_to_images, path_to_results)



def plot_CM_images(cm, num_samples, path_to_images, path_to_results):
    """
    Plots and saves images
    
    """
    num_classes = len(cm.keys())
    
    num_imgs_axis = int(np.sqrt(num_samples)) 
    
    num_rows= num_imgs_axis * num_classes
    num_cols = num_imgs_axis * num_classes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10,10))
    
    fig.suptitle("Confusion Matrix of misclassified images")
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.text(0.5, 0.04, 'Predicted Labels', ha='center', va='center')
    fig.text(0.06, 0.5, 'True Labels', ha='center', va='center', rotation='vertical')

    for i, true_label in enumerate(cm.keys()):
        
        pred_labels = cm[true_label]
        
        if i==0:
            row_i = i
        

        for j, pred_label in enumerate(pred_labels.keys()):

            img_ids = pred_labels[pred_label]

            empty_img_ids = num_samples - len(img_ids)

            if j==0:
                col_j = j
            
            row = 0
            col = 0

            for id in img_ids:

                img = mpimg.imread(os.path.join(path_to_images, id))
                img_shape = img.shape
                
                axes[row_i+row, col_j+col].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
                axes[row_i+row, col_j+col].set_xticks([])
                axes[row_i+row, col_j+col].set_yticks([])

                if row_i + row == num_rows-1:
                    axes[row_i+row, col_j+col].set_xlabel(pred_label)

            
                if col_j + col == 0:
                    axes[row_i+row, col_j+col].set_ylabel(true_label)




                if col ==num_imgs_axis-1:
                    col =0
                    row +=1
                else:
                    col +=1

            if len(img_ids) ==0:
                img_shape = (299,299)        
            
            if empty_img_ids > 0:
                temp = np.full(img_shape, 255)

                for _ in range(empty_img_ids):
                    
                    axes[row_i+row, col_j+col].imshow(temp, cmap='gray', vmin=0, vmax=255, aspect='auto')
                    axes[row_i+row, col_j+col].set_xticks([])
                    axes[row_i+row, col_j+col].set_yticks([])

                    if row_i + row == num_rows-1:
                        axes[row_i+row, col_j+col].set_xlabel(pred_label)

                
                    if col_j + col == 0:
                        axes[row_i+row, col_j+col].set_ylabel(true_label)
                        
                    
                    if col ==num_imgs_axis-1:
                        col =0
                        row+=1
                    else:
                        col +=1
                   
            if empty_img_ids<0:
                raise ValueError("There are more image ids in the dict than number of samples to be plotted")
            
            col_j = col_j + num_imgs_axis
        
        row_i += num_imgs_axis
    
    plt.savefig(path_to_results + "_MS")
    plt.close()

    
def create_dict(classes):
    """
    Setups a dictionary for storing image ids in confusion matrix
    
    """

    output_dict = dict()
    for i in classes:
        output_dict[i] = {}
        for j in classes:
            output_dict[i][j]= []

    return output_dict        


if __name__ == "__main__":
    path_to_txt = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase1_results/histogram_new/multiclass/train/0/valid_svm.txt"
    y_pred = np.loadtxt(path_to_txt, dtype=str, delimiter=" ")
    print("y-pred")
    print(y_pred)
    
    path_to_true = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train_multi.txt"
    y = np.loadtxt(path_to_true, dtype=str, delimiter=" ")
    #print(y)

    pred_ids = y_pred[:,0].tolist()

    y_true = []
    for i in range(y.shape[0]):
        t = y[i,0]
        #print(t)
        if t in pred_ids:
            #print(y[i])
            y_true.append(y[i])

    print("y-true")
    y_true = np.array(y_true)
    
    
    #print("y_true")
    print(y_true)
    #print(y_pred.shape)
    #print(y_true.shape)
    #print(np.count_nonzero(y_true[:,0] == y_pred[:,0]))

    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/code_test/"
    path_to_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train"

    plot_MS(y_true, y_pred, path_to_results, path_to_images)
