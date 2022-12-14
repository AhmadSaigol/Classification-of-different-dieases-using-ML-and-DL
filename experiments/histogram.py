"""
Looking into histogram plots of the graph

"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


path_to_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train_multi.txt"
#img_ids = ["0a1dd587-1656-4fe9-97ab-d29d99d368a8.png", "0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png", "0a0d4a73-a868-4078-8a27-3aa1d69323ad.png", "0a01d14b-2c8b-4155-ae95-095f625315bd.png"]

#labels = ["Normal",  "Lung_Opacity", "pneumonia", "COVID"]

multilabels = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")
img_ids = multilabels[:,0]

labels = multilabels[:,1]


path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/experiments/histograms"

for i in range(len(img_ids)):

    print("Processing ", labels[i])

    fig, axes = plt.subplots(2,3, figsize=(15,15))
    
    img_path = os.path.join(path_to_data, img_ids[i])

    # read image
    img = cv2.imread(img_path, 0)
    print(img.shape)
    
    axes[0,0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title("Original Image")

    print("Ploting histogram")

    axes[1,0].hist(np.ravel(img), color='b', density=True)
    axes[1,0].set_title("Original Histogram")

    # equalize histogram
    print("Ploting Equalized histogram")
    equ_img = cv2.equalizeHist(img)
    axes[0,1].imshow(equ_img, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title("Histogram Equalized Image")

    axes[1,1].hist(np.ravel(equ_img), color='b', density=True)
    axes[1,1].set_title("Equalized Histogram")

    # adaptive histogram equalization.
    print("Ploting Adaptive histogram")
    clahe = cv2.createCLAHE()#clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)

    axes[0,2].imshow(clahe_img, cmap='gray', vmin=0, vmax=255)
    axes[0,2].set_title("Adaptive Histogram Equalized Image")

    axes[1,2].hist(np.ravel(clahe_img), color='b', density=True)
    axes[1,2].set_title("Adaptive Equalized Histogram")


    # save image
    plt.savefig(path_to_results + "/" + labels[i] + "_density_" + img_ids[i] + ".png")


   
print("Processing Completed")