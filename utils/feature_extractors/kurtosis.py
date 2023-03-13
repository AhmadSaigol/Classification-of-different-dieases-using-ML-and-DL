"""
Using Kurtosis as a feature

"""
import numpy as np
import cv2
from scipy.stats import kurtosis

def calculate_kurtosis(X, parameters):
    """
    Calculates kurtosis of the image
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
                method: how to calculate kurtosis. "fisher" (default) or "pearson"
                bias:(default=True) If it is False then the kurtosis is calculated using k statistics to eliminate bias coming from biased moment estimators
    
    Returns:
        features: numpy array of shape (num_images, 1)
        config: dictionary with parameters of the function (including default parameters)

    Additional Notes:
        - currently supports grayscale images only.
    """

    config = {}

    # get method
    if "method" in parameters.keys():
        method = parameters["method"]
    else:
        method = "fisher"
    config["method"] = method

    # get bias
    if "bias" in parameters.keys():
        bias = parameters["bias"]
    else:
        bias = True
    config["bias"] = bias
    
    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating contrast is only supported for grayscale images")
    
    if method == "fisher":
        fisher = True
    elif method == "pearson":
        fisher = False
    else:
        raise ValueError("Unknown Value encountered for parameter 'method' when calcualting kurtosis")
    
    num_images = X.shape[0]
    
    feature = np.zeros((num_images, 1))
    
    # calculate kurtosis
    for img in range(num_images):
        feature[img] = kurtosis(X[img].flatten(), fisher=fisher, bias=bias)
    
    return feature, config


if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image, 0)

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    image_c = np.concatenate((image_c, image_c*0.4))
    
    pipeline={}
    pipeline["kurtosis"] = {}
    pipeline["kurtosis"]["function"] =0 #some function pointer
    pipeline["kurtosis"]["method"] = "pearson"
    pipeline["kurtosis"]["bias"] = True


    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = calculate_kurtosis(image_c, parameters=pipeline["kurtosis"])
    
    print(new.shape)
    print(new)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)