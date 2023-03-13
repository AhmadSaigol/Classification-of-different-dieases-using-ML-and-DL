"""
Using Contrast as a feature

"""
import numpy as np
import cv2

def calculate_contrast(X, parameters):
    """
    Calculates contrast of an image with given method
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
                method: how to calculate contrast. Currently, it supports
                        ["michelson", "rms"]

    Returns:
        features: numpy array of shape (num_images, 1)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:
        
        - for method 'michelson', it is calculated using the formula:
           ( max(I) - min(I) )/(max(I) - min(I))
        
        - for method 'rms', it is calculated using the formula:
           std(X)

        - Currently supports only grayscale images.
        
    """
    
    config = {}

    # get method
    if "method" in parameters.keys():
        method = parameters["method"]
        config["method"] = method
    else:
        raise ValueError("'method' must be provided in the parameters when calculating contrast") 

    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating contrast is only supported for grayscale images")

    #calculate contrast
    if method == "michelson":
        min = np.min(X, axis=(1,2))
        max = np.max(X, axis=(1,2))
        feature = (max - min)/(max + min)
    
    elif method == "rms":
        feature = np.std(X, axis=(1,2))
    
    else:
        raise ValueError("Unknown Value encountered for parameter 'method' when calcualting contrast")

    return feature, config



if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image, 0)

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    image_c = np.concatenate((image_c, image_c*0.4))
    
    pipeline={}
    pipeline["contrast"] = {}
    pipeline["contrast"]["function"] =0 #some function pointer
    pipeline["contrast"]["method"] = "rms"

    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = calculate_contrast(image_c, parameters=pipeline["contrast"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)