import numpy as np 
import os
import json
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
import torch
from matplotlib import pyplot as plt
import pandas as pd
import cv2


def data_augment(path_to_labels, path_to_images, transformations, save_results, total_num_images, random_size = 0.1):
    """
    Offline Data Augmentation

    Parameters:
        path_to_labels: path to .txt containing image ids and labels
        path_to_images: path to the folder containing images
        transformations: to be applied. dictionaray with following structure:
        
            transformations["transformations_1"]["name"] = name of the transformation
            transformations["transformations_1"]["parameter_1"] = value
            transformations["transformations_1"]["parameter_2"] = value

            transformations["transformations_2"]["name"] = name of the transformation
            transformations["transformations_2"]["parameter_1"] = value
            transformations["transformations_2"]["parameter_2"] = value

            Currently it supports:
                "resize": 
                        keys: output_shape, 

                "random_resized_crop": 
                        keys: output_shape, 
            
                "random_perspective"
                            keys: distortion_scale (default = 0.3)                   

                "random_adjust_sharpness":
                            keys: sharpness_factor
                                    
                "random_affine"
                            keys: "translate" (between 0 and 1)
                                "scale" (between 0 and 1)
                                "degrees"

                and "random_horizontal_flip", "random_vectical_flip", "rotate", "random_auto_contrast".          
        
        save_results: path to the folder where results will be stored
        total_num_images: total number of images to generate
        random_size: fraction([0,1]) of total number of images to generate, that will have more than one transformation applied randomly

    Additional Notes:
        - The distribution of classes in the new dataset will be same as of the original dataset
        - This function generates new data as follows:
            - calculates number of images for a class in new data by multiplying ratio of class in original data with total_num_images.
            - These number of images are selected randomly from all the images with that class in the original data.  
            - Out of these images, 'random_size' is reserved for applying non random transformation while the remaining is kept for random transformation
            - Incase of non random transformation
                - number of images required per transfomration is calculated and are selected randomly from the reserved images
                - each image is loaded, each transfomration is applied on it and all the results are in saved in folder 'non_random_images'
            - Incase of random transformation,
                - calcualates random number of transformations to apply
                - gets required transformations randomly from all tranformations passed.
                - each image is loaded, all selected transformations are applied sequentially and the final result is saved in the folder 'random_images'
        - It saves two txt files:
                    'data_augmented.txt': contains image ids and labels for newly generated data
                    'data_augmented_combined.txt': combined image ids and labels of original data and newly generated data.
        - The transformations are saved in .json
        - When dealing with smaller datasets or generating small number of images, because of rounding, you may run into an issue where it would not select any file for a class and would raise an error
            

    """
    rng = np.random.default_rng()

    # load image ids and labels
    y = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")

    num_images = y.shape[0]

    # get unique labels and their counts
    unique_labels, counts = np.unique( y[:,1], return_counts=True)
    
    print("Original Data stats")
    for ul, cs in zip(unique_labels, counts):
      print(f"label {ul} ratio {cs*100/num_images}")

    files = []
    
    for label, label_count in zip(unique_labels,counts):

        # get files with specific label
        label_files = y[y[:,1] == label]
        
        # get number of files for that label in new data
        new_label_files_num = np.round(total_num_images * label_count / num_images).astype(int) 

        # select random files for new data
        indexes =  rng.choice(label_files.shape[0], size= new_label_files_num, replace=False)
        new_label_files = label_files[indexes]

        # get number of non random transformed files
        non_rand_files_count = np.round(new_label_files_num * (1-random_size)).astype(int)
        
        # apply non random transformation and save them
        y_non_rand = apply_non_rand_transform(new_label_files[:non_rand_files_count], transformations, path_to_images, save_results)
        
        # apply random transformation and save them
        y_rand = apply_rand_transform(new_label_files[non_rand_files_count:], transformations, path_to_images, save_results)

        # combine random and non random img_id_labels
        files.extend(y_non_rand)
        files.extend(y_rand)

    files = np.array(files)
    
    # concatenate new labels generated with original labels
    combined = np.concatenate((y, files), axis=0)

    print("Generated Data stats")
    uni, co = np.unique(files[:,1], return_counts=True)
    for ul, cs in zip(uni,co):
      print(f"label {ul} ratio {cs*100/files.shape[0]}")

    print("Original and Generated data combined stats")
    uni, co =np.unique(combined[:,1], return_counts=True)
    for ul, cs in zip(uni, co):
      print(f"label {ul} ratio {cs*100/combined.shape[0]}")
    
    # save new data files txt and combined (y, new data files) txt
    np.savetxt(os.path.join(save_results, "data_augmented.txt"), files, fmt="%s")
    np.savetxt(os.path.join(save_results, "data_augmented_combined.txt"), combined, fmt="%s")

    # save transformations
    with open(os.path.join(save_results, "data_augmentations.json"), "w") as fp:
        json.dump(transformations, fp, indent=4)



def apply_non_rand_transform(files, transformations, path_to_images, save_results):
    """
    Applies each transformation to each image and saves the results 

    Parameters:
        files: numpy array of shape (num_of_imgs, 2) containing image ids (axis=0) and labels(axis=1)
        path_to_images: path to the folder containing images
        transformations: to be applied. dictionaray with following structure:
        
            transformations["transformations_1"]["name"] = name of the transformation
            transformations["transformations_1"]["parameter_1"] = value
            transformations["transformations_1"]["parameter_2"] = value

            transformations["transformations_2"]["name"] = name of the transformation
            transformations["transformations_2"]["parameter_1"] = value
            transformations["transformations_2"]["parameter_2"] = value

            Currently it supports:
                "resize": 
                        keys: output_shape, 

                "random_resized_crop": 
                        keys: output_shape, 
            
                "random_perspective"
                            keys: distortion_scale (default = 0.3)                   

                "random_adjust_sharpness":
                            keys: sharpness_factor
                                    
                "random_affine"
                            keys: "translate" (between 0 and 1)
                                "scale" (between 0 and 1)
                                "degrees"

                and "random_horizontal_flip", "random_vectical_flip", "rotate", "random_auto_contrast".          
         
        save_results: path to the folder where results will be stored

        Returns:
            image_ids_labels: numpy array of shape (num_images, 2) containing image ids and labels for newly generated data.


        Additional Notes:
            - The working of the function is as follows:
                - calculates number of images per tranformation and selects that number of files randomly from 'files'
                - Load an image, apply all transformation separately and save the results in the folder 'non_random_images'
            
    
    """
    # set up results directory
    save_results = os.path.join(save_results, "non_random_images")
    if not os.path.exists(save_results):
      os.mkdir(save_results)

    rng = np.random.default_rng()

    count = 0
    img_ids_labels = []

    # check only one label exist in data
    label =np.unique(files[:, 1])
    if len(label) != 1:
        raise  ValueError(f"no or more than one label({label}) found while applying non random transformation")

    #get number of files
    num_files = files.shape[0]

    # get transformations
    trans = transformations.keys()
    num_trans = len(trans)

    # find number of images per transformation
    num_images_per_trans = np.round(num_files/num_trans).astype(int)


    # select random files from the given files for transformation
    indexes =  rng.choice(num_files, size=num_images_per_trans, replace=False)
    files = files[indexes]

    # for all files
    for img, label in files:
        
        # read image
        path_to_img = os.path.join(path_to_images, img)
        image = read_image(path_to_img, ImageReadMode.GRAY).float()
        image = transforms.Normalize(mean=[0], std=[255] )(image)

        # apply transformations
        for tr in trans:
            file_name = img[:img.rindex(".png")] + "_" + str(count) + ".png"
            
            transform  = get_transfomration(transformations[tr])
            
            if transformations[tr]["name"] == "rotate":
                angle = transformations[tr]["angle"]
                int_method = InterpolationMode.BILINEAR 
                expand =True
                transformed_image = transform(image, angle, int_method, expand)
            else:
                transformed_image = transform(image)
            count +=1
            
            #save image
            norm = transforms.Normalize (mean=[0], std=[1/255] )
            save_img_path = os.path.join(save_results, file_name)
            
            cv2.imwrite(save_img_path, norm(transformed_image).numpy()[0])
            
            img_ids_labels.append([file_name, label])
    
    return img_ids_labels


def apply_rand_transform(files, transformations, path_to_images, save_results):
    """
    Applies random number of random tranformations in sequential order to each image and saves the results

    Parameters:
        files: numpy array of shape (num_of_imgs, 2) containing image ids (axis=0) and labels(axis=1)
        path_to_images: path to the folder containing images
        transformations: to be applied. dictionaray with following structure:
        
            transformations["transformations_1"]["name"] = name of the transformation
            transformations["transformations_1"]["parameter_1"] = value
            transformations["transformations_1"]["parameter_2"] = value

            transformations["transformations_2"]["name"] = name of the transformation
            transformations["transformations_2"]["parameter_1"] = value
            transformations["transformations_2"]["parameter_2"] = value

            Currently it supports:
                "resize": 
                        keys: output_shape, 

                "random_resized_crop": 
                        keys: output_shape, 
            
                "random_perspective"
                            keys: distortion_scale (default = 0.3)                   

                "random_adjust_sharpness":
                            keys: sharpness_factor
                                    
                "random_affine"
                            keys: "translate" (between 0 and 1)
                                "scale" (between 0 and 1)
                                "degrees"

                and "random_horizontal_flip", "random_vectical_flip", "rotate", "random_auto_contrast".          
         
        save_results: path to the folder where results will be stored

        Returns:
            image_ids_labels: numpy array of shape (num_images, 2) containing image ids and labels for newly generated data.


        Additional Notes:
            - The working of the function is as follows:
                - generate number of transformation to apply randomly
                - select the required number of transformations from all the transformations randomly 
                - Load an image, apply the selected transformations sequentially and save the result in the folder 'random_images'
            
    
    """
    #setup result directory    
    save_results = os.path.join(save_results, "random_images")
    if not os.path.exists(save_results):
      os.mkdir(save_results)

    rng = np.random.default_rng()

    count = 0
    img_ids_labels = []

    # make sure only one label is there in the data
    label =np.unique(files[:, 1])
    if len(label) != 1:
        raise  ValueError(f"no or more than one label({label}) found while applying random transformation")

    trans = list(transformations.keys())

    # get resize tranformation
    flag=False
    flag2 =False
    for t in trans:
        if transformations[t]["name"] == "rotate":
            flag2 = True

        if transformations[t]["name"] == "resize":
            resize_trans = get_transfomration(transformations[t])
            flag = True
            break
    if flag2:
        if not flag:
            raise ValueError("resize transformation must be given when using rotate transformation") 

    # generate number of transformation to apply
    num_trans = len(trans)
    num_trans_to_apply = np.random.randint(low=1, high=len(trans)+1)

    # get random transformations
    indexes =  rng.choice(num_trans, size=num_trans_to_apply, replace=False)
    trans_to_apply = np.array(trans)[indexes]


    # for all files
    for img, label in files:
        
        #read image
        file_name = img[:img.rindex(".png")] + "_" + str(count) + ".png"
        path_to_img = os.path.join(path_to_images, img)
        image = read_image(path_to_img, ImageReadMode.GRAY).float()
        image = transforms.Normalize(mean=[0], std=[255] )(image)
        
        transformed_image = image

        # apply the transformation
        for tr in trans_to_apply:

            transform  = get_transfomration(transformations[tr])

            if transformations[tr]["name"] == "rotate":
                angle = transformations[tr]["angle"]
                int_method = InterpolationMode.BILINEAR 
                expand =True
                transformed_image = transform(transformed_image, angle, int_method, expand)
                
            else:
                transformed_image = transform(transformed_image)

        count +=1

        # save image
        norm= transforms.Normalize (mean=[0], std=[1/255] )
        save_img_path = os.path.join(save_results, file_name)
        
        cv2.imwrite(save_img_path, norm(transformed_image).numpy()[0] )
        
        img_ids_labels.append([file_name, label])

    return img_ids_labels



def get_transfomration(transformations):
    """
    Returns transforms
    
    Parameters:
        transformations: Ordered dict with structure
        transformations = {}
        transformations["name"] = value
        transformations["parameter_1"] = value
        transformations["parameter_2"] = value
        
    Currently supports transformation: 
       
        "resize": 
                    keys: output_shape, 

        "random_resized_crop": 
                    keys: output_shape, 
        
        "random_horizontal_flip"
                    keys: 
        
        "random_vectical_flip"
                    keys:          
        
        "random_perspective"
                    keys: distortion_scale = 0.3                   
                    
        "rotate"
                    keys: 

        "random_adjust_sharpness":
                    keys: sharpness_factor
                          

        "random_auto_contrast":
                keys:
    
        "random_affine"
                    keys: "translate" (between 0 and 1)
                          "scale" (between 0 and 1)
                          "degrees"

        
    """
    
    # get name of transfomration and associated keys
    tran_name = transformations["name"]
    trans_keys = transformations.keys()
    
    # resize
    if tran_name == "resize":
        
        if "output_shape" in trans_keys:
            output_shape = transformations["output_shape"]
        else:
            raise ValueError("Output shape must be provided while resizing")
        
        return transforms.Resize(output_shape)

    # random perspective
    if tran_name == "random_perspective":
        
        if "distortion_scale"  in trans_keys:
            distortion_scale  = transformations["distortion_scale"]
        else:
            distortion_scale = 0.3
        
        return transforms.RandomPerspective(distortion_scale=distortion_scale, p=1)

    # random horizontal flip
    elif tran_name == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip(p=1)
      
    # random vertical flip
    elif tran_name == "random_vertical_flip":
        return transforms.RandomVerticalFlip(p=1)
            
    
    # random_rotation
    elif tran_name == "rotate":
        return transforms.functional.rotate


    elif tran_name == "random_auto_contrast":
        return transforms.RandomAutocontrast(p=1)
    
    elif tran_name == "random_adjust_sharpness":

        if "sharpness_factor" in trans_keys:
            sharpness_factor = transformations["sharpness_factor"]
        else:
            raise ValueError("sharpness_factor must be provided while random rotation")

        return transforms.RandomAdjustSharpness(sharpness_factor, p=1)


    elif tran_name == "random_affine":
        
        if "degrees" in trans_keys:
            degrees = transformations["degrees"]
        else:
            raise ValueError("degrees must be provided while random affine")

        if "translate" in trans_keys:
            translate = transformations["translate"]
        else:
            raise ValueError("translate must be provided while random affine")
        
        if "scale" in trans_keys:
            scale = transformations["scale"]
        else:
            raise ValueError("scale must be provided while random affine")
        
        return transforms.RandomAffine(degrees=degrees, scale=scale, translate=translate)

    elif tran_name == "random_resized_crop":
        
        if "output_shape" in trans_keys:
            output_shape = transformations["output_shape"]
        else:
            raise ValueError("output_shape must be provided while random random resizing crop")

        return transforms.RandomResizedCrop(output_shape)

    
    else:
        raise ValueError("Unknown transformation passed")


def generate_y_txt(path_to_images, path_to_orig_y, save_path):

    """
    Generates .txt file which contains image ids and labels of both directory and original y

    Parameters:
        path_to_images: path to the folder containing images
        path_to_orig_y: path to the .txt file containing image_ids and labels
        save_path: path to the folder where results will be saved

    Additional Notes:
        - It combines image ids and labels in the txt file provided by 'path_to_images' and 'path_to_orig_y' and saves it to 'train_multi.txt'
        - The label for images in 'path_to_images' is fonud as follows:
            Assuming the files in the folder are named as 'imageId_number.png', it will get imageId and search for it in 'path_to_orig_y'

    
    """

    # load y txt file
    y = np.loadtxt(path_to_orig_y, dtype=str, delimiter=" ")

    # read names of files in the folder
    y_aug =sorted(os.listdir(path_to_images))

    results = []
    for ya in y_aug:
        
        # get image id
        curr_path = ya[:ya.rindex("_")]+".png"
        
        # find label
        label = y[y[:,0] == curr_path][0][1]

        results.append([ya, label])

    
    results = np.array(results)
    combined = np.concatenate((y, results), axis=0)

    # save combined (y, new data files) to txt file
    np.savetxt(os.path.join(save_path, "train_multi.txt"), combined, fmt="%s")
    

    
if __name__ == "__main__":
    
    transformations = {}
    transformations["transformations"] = {}

    transformations["transformations"]["random_adjust_sharpness"] = {}
    transformations["transformations"]["random_adjust_sharpness"]["name"] = "random_adjust_sharpness"
    transformations["transformations"]["random_adjust_sharpness"][ "sharpness_factor"] = 2

    transformations["transformations"]["random_auto_contrast"] = {}
    transformations["transformations"]["random_auto_contrast"]["name"] = "random_auto_contrast"

    transformations["transformations"]["random_affine"] = {}
    transformations["transformations"]["random_affine"]["name"] = "random_affine"
    transformations["transformations"]["random_affine"]["degrees"] = (-15, 15)
    transformations["transformations"]["random_affine"]["translate"] = (0.1, 0.3)
    transformations["transformations"]["random_affine"]["scale"] = (0.8, 1)

    transformations["transformations"]["rotate90"] = {}
    transformations["transformations"]["rotate90"]["name"] = "rotate"
    transformations["transformations"]["rotate90"]["angle"] = 90

    transformations["transformations"]["rotate-90"] = {}
    transformations["transformations"]["rotate-90"]["name"] = "rotate"
    transformations["transformations"]["rotate-90"]["angle"] = -90

    transformations["transformations"]["random_perspective"] = {}
    transformations["transformations"]["random_perspective"]["name"] = "random_perspective"
    transformations["transformations"]["random_perspective"]["distortion_scale"] = 0.2

    transformations["transformations"]["horizontal_flip"] = {}
    transformations["transformations"]["horizontal_flip"]["name"] = "random_horizontal_flip"

    transformations["transformations"]["vertical_flip"] = {}
    transformations["transformations"]["vertical_flip"]["name"] = "random_vertical_flip"

    transformations["transformations"]["random_resized_crop"] = {}
    transformations["transformations"]["random_resized_crop"]["name"] = "random_resized_crop"
    transformations["transformations"]["random_resized_crop"]["output_shape"] = (256,256)

    transformations["transformations"]["resize"] = {}
    transformations["transformations"]["resize"]["name"] = "resize"
    transformations["transformations"]["resize"]["output_shape"] = (256,256)

    path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train_multi.txt"
    save_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/aug_dataset-21-01"
    
    if not os.path.exists(save_results):
        os.mkdir(save_results)
    path_to_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train"
    total_num_images = 4235*2
    random_size = 0.15
    
    data_augment(path_to_labels, path_to_images, transformations['transformations'], save_results, total_num_images, random_size)
