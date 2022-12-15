"""
Generates a feature vector using number of pixels in each bin in histogram

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def calculate_histogram(X, parameters):
    """
    Calculates and returns values of histogram (number of values in each bin or normalized, see 'denisty' parameter)
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
            bins: number of bins for histogram (defau1t=10)
            range: lower and upper range of bins (default = (0,256))
            denisty: If False, the result will contain the number of samples in each bin. 
                     If True, the result is the value of the probability density function at the bin, 
                     normalized such that the integral over the range is 1 (default=False)

    Returns:
        features: numpy array of shape (num_images, num_bins)
        config: dictionary with keys

    """
    config = {}

    # get number of bins
    if "bins" in parameters.keys():
        bins = parameters["bins"]
    else:
        bins = 10

    config["bins"] = bins

    # get range
    if "range" in parameters.keys():
        rng = parameters["range"]
    else:
        rng = (0,256)
        
    config["range"] = rng

    # get denisty
    if "density" in parameters.keys():
        density = parameters["density"]
    else:
        density = False
        
    config["density"] = density
    
    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating histogram is only supported for grayscale images")
    
    
    num_images = X.shape[0]

    feature = []
    for img in range(num_images):
        temp = np.histogram(X[img].flatten(), bins=bins, range=rng, density=density)[0]
        feature.append(temp)

    feature = np.array(feature)
        
    return feature, config


if __name__ == "__main__":

    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png"
    image_c =cv2.imread(path_to_image, 0)
    image_c = image_c /255

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    image_c = np.concatenate((image_c, image_c))
    
    pipeline={}
    pipeline["histogram"] = {}
    pipeline["histogram"]["function"] =0 #some function pointer
    pipeline["histogram"]["bins"] = 10
    pipeline["histogram"]["range"] = (0,1)
    pipeline["histogram"]["density"] = True


    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = calculate_histogram(image_c, parameters=pipeline["histogram"])
    
    print(new.shape)
    print(new)

    
    print(config)