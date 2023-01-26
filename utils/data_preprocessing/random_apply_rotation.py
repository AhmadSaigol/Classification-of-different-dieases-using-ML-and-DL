import cv2
import numpy as np
from torchvision import transforms
import torch


def random_apply_rotation(images, parameters):
    """
    Randomly applies rotation

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                    degrees: 
                    expand: default = False
                     p = 0.5
                    

    Returns:
        (same as input)
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
  
    """
    # set up output config
    config = {}
    
    para_keys = parameters.keys()
    
    num_images, _, _, _ = images.shape


    if "degrees" in para_keys:
        degrees = parameters["degrees"]
        config["degrees"] = degrees
    else:
        raise ValueError("degrees must be provided while random rotation")


    if "expand" in para_keys:
        expand = parameters["expand"]
    else:
        expand = False
    config["expand"] = expand

     # get output size
    if "p" in para_keys:
        p = parameters["p"]

    else:
        p = 0.5
        
    config["p"] = p
                
    rf =transforms.RandomApply([transforms.RandomRotation(degrees=90, expand=True)], p=p)
    tensor = transforms.ToTensor()


    # apply resizing
    results = []
    for img in range(num_images):

        result = tensor(images[img])
        result = rf(result)
        result = torch.permute(result, (1,2,0))
        
        results.append(result.numpy())

    results = np.array(results)

    return results, config


if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image)

    image_c = np.expand_dims(image_c, axis=0)
    #image_c = np.expand_dims(image_c, axis=-1)
    pipeline={}
    pipeline["random_rotation"] = {}
    pipeline["random_rotation"]["function"] =0 #some function pointer
    pipeline["random_rotation"]["degrees"] = 90
    pipeline["random_rotation"]["expand"] = True
    pipeline["random_rotation"]["p"] = 1

    
    #pipeline["resize_image"]["interpolation"] = "bilinear"

    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = random_apply_rotation(image_c, parameters=pipeline["random_rotation"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)