import cv2
import numpy as np
from torchvision import transforms
import torch


def random_apply_affine(images, parameters):
    """
    Applies affine transformation to the image with given probability

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                        degrees: range of degrees to seleect from. (min, max)
                        scale: scaling factor. (min, max)
                        translate: range of translation in absolute fractions (min, max)
                        p: probability (default= 0.5)
                        

    Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:
        for more info, see
            https://pytorch.org/vision/main/generated/torchvision.transforms.RandomApply.html
            https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine
  
    """
    # set up output config
    config = {}
    
    para_keys = parameters.keys()
    
    num_images, _, _, _ = images.shape


    if "degrees" in para_keys:
        degrees = parameters["degrees"]
        config["degrees"] = degrees
    else:
        raise ValueError("degrees must be provided while random affine")

    if "translate" in para_keys:
        translate = parameters["translate"]
        config["translate"] = translate
    else:
        raise ValueError("translate must be provided while random affine")

    if "scale" in para_keys:
        scale = parameters["scale"]
        config["scale"] = scale
    else:
        raise ValueError("scale must be provided while random affine")

     # get probability
    if "p" in para_keys:
        p = parameters["p"]
    else:
        p = 0.5
    config["p"] = p
                        
    rf =transforms.RandomApply([transforms.RandomAffine(degrees=degrees, scale=scale, translate=translate)], p=p)
    tensor = transforms.ToTensor()

    results = []
    for img in range(num_images):
        
        # convert to tensor
        result = tensor(images[img])

        # apply affine transformation 
        result = rf(result)
      
        # pytorch moves the channel on first axis while in our case channel is the last axis so     
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
    pipeline["random_affine"] = {}
    pipeline["random_affine"]["function"] =0 #some function pointer
    pipeline["random_affine"]["degrees"] = (-15, 15)
    pipeline["random_affine"]["scale"] = (0.1, 0.3)
    pipeline["random_affine"]["translate"] = (0.1, 0.3)
    pipeline["random_affine"]["p"] = 1

    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = random_apply_affine(image_c, parameters=pipeline["random_affine"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)