import cv2
import numpy as np
from torchvision import transforms
import torch


def random_resized_crop(images, parameters):
    """
    Randomly crops a portion of an image and resizes it to a given shape

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                        output_size: tuple of ints (width, height)
                    

    Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:
        for more info, see
            https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
  
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
        raise ValueError("'output_size' must be provided in the parameters when random crop resizing the image.")

    
    rrc = transforms.RandomResizedCrop(dsize)
    tensor = transforms.ToTensor()

    results = []
    for img in range(num_images):

        # convert to tensor
        result = tensor(images[img])
        
        # apply transformation
        result = rrc(result)

        # pytorch moves the channel on first axis while in our case channel is the last axis      
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
    pipeline["random_resized_crop"] = {}
    pipeline["random_resized_crop"]["function"] =0 #some function pointer
    pipeline["random_resized_crop"]["output_size"] = (360, 600)
   
    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = random_resized_crop(image_c, parameters=pipeline["random_resized_crop"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)