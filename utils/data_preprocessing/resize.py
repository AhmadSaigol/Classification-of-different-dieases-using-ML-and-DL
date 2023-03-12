import cv2
import numpy as np

def resize(images, parameters):
    """
    Resizes the image 

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                        output_size: tuple of ints (width, height)
                        interpolation: interpolation method (default="area")

    Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:
        According to the opencv docu, it is best to use "area" as interpolation method for shrinking the image
        and for enlarging the image, use "bicubic" (slow) or "bilinear" (faster)

        For more info, see
            https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    """
    # set up output config
    config = {}
    
    para_keys = parameters.keys()

    # get output size
    if "output_size" in para_keys:
        dsize = parameters["output_size"]
        config["output_size"] = dsize
    else:
        raise ValueError("'output_size' must be provided in the parameters when resizing the image.")

    # get interpolation method    
    interpolations = ["nearest_neighbor", "bilinear", "bicubic", "area" ]

    interpolation_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
 
    if "interpolation" in para_keys:
        interpolation = parameters["interpolation"]
    else:
        interpolation = "area"

    config["interpolation"] = interpolation
    
    if interpolation in interpolations:
        index = interpolations.index(interpolation)
    else:
        raise ValueError("Unknown Value encountered for the parameter 'interpolation' while resizing the image.")


    # apply resizing
    results = []
    for img in range(images.shape[0]):

        temp = cv2.resize(images[img], dsize=dsize, interpolation=interpolation_methods[index])
        
        # add dim when image is grayscale for consistency
        if images[img].shape[-1] == 1:
            temp = np.expand_dims(temp, axis=-1)

        results.append(temp)

    results = np.array(results)

    return results, config

if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image)

    image_c = np.expand_dims(image_c, axis=0)
    #image_c = np.expand_dims(image_c, axis=-1)
    pipeline={}
    pipeline["resize_image"] = {}
    pipeline["resize_image"]["function"] =0 #some function pointer
    pipeline["resize_image"]["output_size"] = (360, 600)
    #pipeline["resize_image"]["interpolation"] = "bilinear"

    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = resize(image_c, parameters=pipeline["resize_image"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)