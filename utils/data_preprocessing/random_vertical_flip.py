import cv2
import numpy as np
from torchvision import transforms
import torch


def random_vertical_flip(images, parameters):
    """
    Randomly flips an image vertically with given probability

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary with keys:
                    p: probabilty (default = 0.5)
                    

    Returns:
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)

    Additional Notes:
        for more info, see
            https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html 
   
    """
    # set up output config
    config = {}
    
    para_keys = parameters.keys()
    
    num_images, _, _, _ = images.shape

    # get probability
    if "p" in para_keys:
        p = parameters["p"]
    else:
        p = 0.5
    config["p"] = p
    
    rvf = transforms.RandomVerticalFlip(p)
    tensor = transforms.ToTensor()


    results = []
    for img in range(num_images):
        
        # convert to tensor
        result = tensor(images[img])
        
        # apply transformation
        result = rvf(result)

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
    pipeline["random_vertical_flip"] = {}
    pipeline["random_vertical_flip"]["function"] =0 #some function pointer
    pipeline["random_vertical_flip"]["p"] = 1
   
    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = random_vertical_flip(image_c, parameters=pipeline["random_vertical_flip"])
    
    print(new.shape)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)