import numpy as np
import cv2


def normalize(images, parameters):
    """
    (for now) Changes brightness of image to the range [0, 1]

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                    method: "simple" or "minmax"
                    
    Returns:
        (same as input)
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Note:
        when method is "simple", all values are just divided by 255
        when method is "minmax", it maps values to [0,1] using min and max value of an image

        For additional normalization methods, see
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd
    """
    #setup output config
    config = {}


    if "method" in parameters.keys():
        method = parameters["method"]
        config["method"] = method
    else:
        raise ValueError("'method' must be provided in the parameters when normalizating the image.") 

    
    if method == "simple":

        results = images / 255

   
    elif method == "minmax":
        flag = True

        for img in range(images.shape[0]):

            temp = cv2.normalize(images[img], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            temp = np.expand_dims(temp, axis=0)

            if flag:
                results = temp
                flag = False
            else:
                results = np.concatenate((results, temp))  
    else:
        raise ValueError("Unknown Value encountered for the parameter'method' while normalizing the image.")


    
    return results, config



if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image)

    image_c = np.expand_dims(image_c, axis=0)
    #image_c = np.expand_dims(image_c, axis=-1)
    pipeline={}
    pipeline["normalize_image"] = {}
    pipeline["normalize_image"]["function"] =0 #some function pointer
    pipeline["normalize_image"]["method"] = "minmax"
    #pipeline["resize_image"]["interpolation"] = "bilinear"

    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = normalize(image_c, parameters=pipeline["normalize_image"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)