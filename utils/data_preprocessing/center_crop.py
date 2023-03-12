import cv2
import numpy as np
from torchvision import transforms
import torch


def center_crop(images, parameters):
    """
    center crop of the image

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                    output_size: size of the crop (int or tuple of int)
                    

    Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:

        for more info, see 
        https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html
   
    """
    # set up output config
    config = {}
    
    para_keys = parameters.keys()
    
    num_images, _, _, _ = images.shape

    # get output size
    if "output_size" in para_keys:
        dsize = parameters["output_size"]
        config["output_size"] = dsize

    else:
        raise ValueError("'output_size' must be provided in the parameters when center cropping the image.")

    
    cc = transforms.CenterCrop(dsize)
    tensor = transforms.ToTensor()

    results = []
    for img in range(num_images):
        
        # convert to tensor
        result = tensor(images[img])

        # generate center crops
        result = cc(result)

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
    pipeline["cc"] = {}
    pipeline["cc"]["function"] =0 #some function pointer
    pipeline["cc"]["output_size"] = (250,250)
   
    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = center_crop(image_c, parameters=pipeline["cc"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)