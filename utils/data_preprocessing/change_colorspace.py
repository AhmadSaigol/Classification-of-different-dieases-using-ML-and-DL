import cv2
import numpy as np

def change_colorspace(images, parameters):
    """
    Changes the colorspace of image

    Parameters:
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary containing following keys
                conversion: defines input color space to output color space 
                    currently supports:["BGR2RGB", "RGB2BGR", 
                                        "BGR2GRAY", "RGB2GRAY", "GRAY2BGR", "GRAY2RGB",
                                        "BGR2YCrCb", "RGB2YCrCb", "YCrCb2BGR", "YCrCb2RGB",
                                        "BGR2HSV", "RGB2HSV", "HSV2BGR", "HSV2RGB",
                                        "BGR2HLS", "RGB2HLS", "HLS2BGR", "HLS2RGB"]

     Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)


    Note: 
        For more conversions, see
        https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0   
    
    """
    # setup output config
    config = {}

    if "conversion" in parameters.keys():
        conversion = parameters["conversion"]
        config["conversion"] = conversion 
    else:
        raise ValueError("'conversion' must be provided in the parameters when changing colorspace.")

    conversion_codes = ["BGR2RGB", "RGB2BGR", 
            "BGR2GRAY", "RGB2GRAY", "GRAY2BGR", "GRAY2RGB",
            "BGR2YCrCb", "RGB2YCrCb", "YCrCb2BGR", "YCrCb2RGB",
            "BGR2HSV", "RGB2HSV", "HSV2BGR", "HSV2RGB",
            "BGR2HLS", "RGB2HLS", "HLS2BGR", "HLS2RGB"]
    
    codes = [cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR, 
            cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY, cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB,
            cv2.COLOR_BGR2YCrCb, cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2BGR, cv2.COLOR_YCrCb2RGB,
            cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB,
            cv2.COLOR_BGR2HLS, cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2BGR, cv2.COLOR_HLS2RGB]
    
    if conversion in conversion_codes:
        index = conversion_codes.index(conversion)    
    else:
        raise ValueError("Unknown Value encountered for the parameter'conversion' while changing the colorspace.")


    results = []
    for img in range(images.shape[0]):
        results.append(cv2.cvtColor(images[img], codes[index]))
  
    results =np.array(results)

    # add dim when converting to grayscale for consistency
    if conversion in ["BGR2GRAY", "RGB2GRAY"]:
        results = np.expand_dims(results, axis=-1)  
        
        
    return results, config


if __name__ == "__main__":

    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/test/0a23fc8b-01c1-4f0b-a33c-d749811da434.png"
    image_c =cv2.imread(path_to_image, 1)

    image_c = np.expand_dims(image_c, axis=0)
    #image_c = np.expand_dims(image_c, axis=-1)
    pipeline={}
    pipeline["map_to_RGB"] = {}
    pipeline["map_to_RGB"]["function"] =0 #some function pointer
    pipeline["map_to_RGB"]["conversion"] = "RGB2GRAY"
    print(image_c.shape)

    new, config = change_colorspace(image_c, parameters=pipeline["map_to_RGB"])
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)