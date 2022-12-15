"""
Extracts haralick features

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import mahotas as mht

def calculate_haralick(X, parameters):
    """
    Calculates and returns haralick features
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
            blur: whether to image before extracting features or not (default=False)
            distance: how many pixels to consider at a time (default=1)

    Returns:
        features: numpy array of shape (num_images, 13)
        config: dictionary with keys
    
    See more under:
    "https://cvexplained.wordpress.com/2020/07/22/10-6-haralick-texture/"
    "https://mahotas.readthedocs.io/en/latest/api.html"
    
    """
    config = {}

    # determine whether to blur the image or not
    if "blur" in parameters.keys():
        blur = parameters["blur"]
    else:
        blur = False

    config["blur"] = blur


    # distance
    if "distance" in parameters.keys():
        distance = parameters["distance"]
    else:
        distance = 1

    config["distance"] = distance

    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating histogram is only supported for grayscale images")
    
    num_images = X.shape[0]

    feature = []

    for img in range(num_images):
        
        if blur:
            proc_img = cv2.bilateralFilter(np.squeeze(X[img], axis=-1) , d=5, sigmaColor=75, sigmaSpace=75)
        else:
            proc_img = np.squeeze(X[img], axis=-1)
        
        
        temp = mht.features.haralick(proc_img, distance=distance, return_mean=True)
        feature.append(temp)

    
    feature = np.array(feature)
    
    return feature, config

if __name__ == "__main__":

    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png"
    image_c =cv2.imread(path_to_image, 0)
    
    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a1b4148-5a4e-4f1f-8f85-9f4d975b2299.png"
    image_c2 =cv2.imread(path_to_image, 0)
    
    #image_c = image_c /255
    #image_c2 = image_c2 /255

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    
    image_c2 = np.expand_dims(image_c2, axis=0)
    image_c2 = np.expand_dims(image_c2, axis=-1)
    
    images = np.concatenate((image_c, image_c2))
    
    pipeline={}
    pipeline["haralick"] = {}
    pipeline["haralick"]["function"] =0 #some function pointer
    pipeline["haralick"]["blur"] = True
    pipeline["haralick"]["distance"] = 1
    

    print(images.shape)
    cv2.imshow("original", images[0])
    cv2.waitKey(0)    

    new, config = calculate_haralick(images, parameters=pipeline["haralick"])
    
    print(new.shape)
    print(new)

    
    print(config)