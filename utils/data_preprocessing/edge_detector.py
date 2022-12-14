import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def canny_edge_detector(images, parameters):
    """
    Detectes edges using canny edge detection algo

    Parameters:
        images: numpy array of shape(num_images, height, width, channel)
        parameters: dictionary containing following keys:
            blur: whether to blur the image before finding edges or not (defaul=False)
            threshold1: for hystersis procedure (defualt:50)
            threshold2: for hystersis procedure (default:100)
            apertureSize: for Sobel operator (default=3)
            L2gradient: whether to use L2 norm or L1 norm to calculate gradient magnitude (default= False)

     
     Returns:
        (same as input)
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)


    Note: 
        For more conversions, see
       https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga2a671611e104c093843d7b7fc46d24af   
    
    """

    # setup output config
    config = {}

    # determine whether to blur the image or not
    if "blur" in parameters.keys():
        blur = parameters["blur"]
    else:
        blur = False

    config["blur"] = blur

    # threshold1
    if "threshold1" in parameters.keys():
        th1 = parameters["threshold1"]
    else:
        th1 = 50

    config["threshold1"] = th1

    # threshold2
    if "threshold2" in parameters.keys():
        th2 = parameters["threshold2"]
    else:
        th2 = 100

    config["threshold2"] = th2
    

    # determine apertureSize
    if "apertureSize" in parameters.keys():
        apertureSize = parameters["apertureSize"]
    else:
        apertureSize = 3

    config["apertureSize"] = apertureSize


    # determine L2gradient
    if "L2gradient" in parameters.keys():
        L2gradient = parameters["L2gradient"]
    else:
        L2gradient = False

    config["L2gradient"] = L2gradient

    # make sure grayscale image is given
    if images.shape[-1] != 1:
        raise ValueError("Currently, calculating edge detction is only supported for grayscale images")
    
    num_images = images.shape[0]

    for img in range(num_images):
        
        if blur:
            proc_img = cv2.bilateralFilter(np.squeeze(images[img], axis=-1) , d=5, sigmaColor=75, sigmaSpace=75)
        else:
            proc_img = np.squeeze(images[img], axis=-1)
        
        temp = cv2.Canny(proc_img, threshold1=th1, threshold2=th2, apertureSize=apertureSize, L2gradient=L2gradient)

        temp = np.expand_dims(temp, axis=0)
        temp = np.expand_dims(temp, axis=-1)
        


        if img == 0:
            results = temp
        else:
            results = np.concatenate((results, temp), axis=0)    
    
    
    
    return results, config


if __name__ == "__main__":
    """
    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png"
    image_c =cv2.imread(path_to_image, 0)
    
    path_to_image = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a1b4148-5a4e-4f1f-8f85-9f4d975b2299.png"
    image_c2 =cv2.imread(path_to_image, 0)
    
    image_c = image_c /255
    image_c2 = image_c2 /255

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    
    image_c2 = np.expand_dims(image_c2, axis=0)
    image_c2 = np.expand_dims(image_c2, axis=-1)
    
    images = np.concatenate((image_c, image_c2))
    
    pipeline={}
    pipeline["canny_edges"] = {}
    pipeline["canny_edges"]["function"] =0 
    pipeline["canny_edges"]["blur"] = False
    pipeline["canny_edges"]["threshold1"] = 50
    pipeline["canny_edges"]["threshold2"] = 100
    pipeline["canny_edges"]["apertureSize"] = 5
    pipeline["canny_edges"]["L2gradient"] = True
    

    print(images.shape)
    cv2.imshow("original", images[0])
    cv2.waitKey(0)    

    new, config = canny_edge_detector(images, parameters=pipeline["canny_edges"])
    
    print(new.shape)
    print(new)

    
    print(config)
    """


    path_to_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
    path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train_multi.txt"
    #img_ids = ["0a1dd587-1656-4fe9-97ab-d29d99d368a8.png", "0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png", "0a0d4a73-a868-4078-8a27-3aa1d69323ad.png", "0a01d14b-2c8b-4155-ae95-095f625315bd.png"]

    #labels = ["Normal",  "Lung_Opacity", "pneumonia", "COVID"]

    multilabels = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")
    img_ids = multilabels[:,0]

    labels = multilabels[:,1]


    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/experiments/canny_edges"

    pipeline={}
    pipeline["canny_edges"] = {}
    pipeline["canny_edges"]["function"] =0 
    pipeline["canny_edges"]["blur"] = True
    pipeline["canny_edges"]["threshold1"] =250
    pipeline["canny_edges"]["threshold2"] = 500
    pipeline["canny_edges"]["apertureSize"] = 5
    pipeline["canny_edges"]["L2gradient"] = True

    for i in range(len(img_ids)):

        print("Processing Image: ", img_ids[i])
        
        img_path = os.path.join(path_to_data, img_ids[i])

        # read image
        img = cv2.imread(img_path, 0)
        
        #img = img /255
        
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        
        if  i== 0:
            images = img
        else:
            images = np.concatenate((images, img))

    results, config = canny_edge_detector(images=images, parameters=pipeline["canny_edges"])

    for i in range(len(img_ids)):
        
        print("Procesing label: ", labels[i])

        fig, axes = plt.subplots(1,2, figsize=(15,15))

        axes[0].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Original Image")

        axes[1].imshow(results[i], cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Canny Edges")

        # save image
        plt.savefig(path_to_results + "/" + labels[i] + "_" + img_ids[i] + ".png")


    
    print("Processing Completed")