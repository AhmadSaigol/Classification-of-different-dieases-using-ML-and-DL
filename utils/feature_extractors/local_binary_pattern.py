"""
Extracts local binary pattern

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern as lbp
import os


def calculate_lbp(X, parameters):
    """
    Calculates LBP image and returns its histogram as features
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
            P: number of neighbour set points
            R: radius of circle
            method: to determine the pattern. "ror", "var" or "uniform" (default)

    Returns:
        features: numpy array of shape (num_images, 13)
        config: dictionary with keys
    
    See more under:
    "https://cvexplained.wordpress.com/2020/07/22/10-6-haralick-texture/"
    "https://mahotas.readthedocs.io/en/latest/api.html"
    
    """
    config = {}

    # P
    if "P" in parameters.keys():
        p = parameters["P"]
    else:
        raise ValueError("parameter 'P' must be provided in local_binary_pattern")

    config["P"] = p


    # R
    if "R" in parameters.keys():
        r = parameters["R"]
    else:
        raise ValueError("parameter 'R' must be provided in local_binary_pattern")

    config["R"] = r

    # method
    if "method" in parameters.keys():
        method = parameters["method"]
    else:
        method = "uniform"

    config["method"] = method


    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating histogram is only supported for grayscale images")
    
    num_images = X.shape[0]

    feature = []

    for img in range(num_images):

        lbp_img = lbp(np.squeeze(X[img]), P=p, R=r, method=method)

        (hist, _) = np.histogram(lbp_img.ravel(), 
                                bins=np.arange(0, p + 3), 
                                range=(0, p + 2))
		
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        feature.append(hist)

    
    feature = np.array(feature)
    
    return feature, config

if __name__ == "__main__":

    path_to_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
    imgs = os.listdir(path_to_images)
    images = []
    for i in imgs:
        path_to_image = os.path.join(path_to_images, i)
        image_c =cv2.imread(path_to_image, 0)
    
    
        image_c = np.expand_dims(image_c, axis=0)
        image_c = np.expand_dims(image_c, axis=-1)
    
    
        images.append(image_c)

    images =np.array(images)

    
    pipeline={}
    pipeline["lbp"] = {}
    pipeline["lbp"]["function"] =0 #some function pointer
    pipeline["lbp"]["P"] = 60
    pipeline["lbp"]["R"] = 20
    pipeline["lbp"]["method"] = "uniform"
    

    #print(images.shape)
    #cv2.imshow("original", images[0])
    #cv2.waitKey(0)    

    new, config = calculate_lbp(images, parameters=pipeline["lbp"])
    
    print(new.shape)
    #print(new)
    
    print(config)