"""
Counts non zero values

"""
import numpy as np
import cv2


def count_nonzeros(X, parameters):
    """
    Calculates and returns non zero values in image
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: 
            

    Returns:
        features: numpy array of shape (num_images, 1)
        config: dictionary with parameters of the function (including default parameters)

    Additional Notes:
        - currently supports grayscale images only
    
    """
    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating zernike moments is only supported for grayscale images")
    
    feature = np.count_nonzero(X, axis=(1,2,3))
    feature = np.expand_dims(feature, axis=-1)  

    return feature, parameters



if __name__ == "__main__":

    images = np.array([ 1,0,1,
                        0,1,1,
                        0,1,0,
                        0,1,1,  
                        
                        1,0,1,
                        0,0,0,
                        1,0,1,
                        1,0,1 
                       
                        ]).reshape(2,4,3,1)
    
    pipeline={}
    pipeline["non_zero"] = {}
    pipeline["non_zero"]["function"] =0 #some function pointer
    
    print(images)

    new, config = count_nonzeros(images, parameters=pipeline["non_zero"])
    
    print(new.shape)
    print(new)

    
    print(config)