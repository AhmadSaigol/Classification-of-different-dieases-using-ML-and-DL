import numpy as np 
import os
import json
from torchvision import transforms
from torchvision.io import read_image, write_png, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
import torch
from matplotlib import pyplot as plt
import pandas as pd
import cv2


def data_augment(path_to_labels, path_to_images, transformations, save_results, total_num_images, random_size = 0.1):
    """
    
    
    total_num_images: number of images to generate
    random_size: ratio reserved for generating random transfomrated images
    
    """
    rng = np.random.default_rng()

    # load y txt file
    y = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")

    num_images = y.shape[0]

    # get unique labels and their counts
    unique_labels, counts = np.unique( y[:,1], return_counts=True)
    
    print("initial stats")
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
    combined = np.concatenate((y, files), axis=0)

    print("random stats")
    uni, co = np.unique(files[:,1], return_counts=True)
    for ul, cs in zip(uni,co):
      print(f"label {ul} ratio {cs*100/files.shape[0]}")

    print("combined stats")
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
    
    save_results = os.path.join(save_results, "non_random_images")
    if not os.path.exists(save_results):
      os.mkdir(save_results)

    rng = np.random.default_rng()

    count = 0
    img_ids_labels = []

    # check only one label exist in data
    label =np.unique(files[:, 1])
    if len(label) != 1:
        raise  ValueError("more than one label found")

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

        #print("Non random")
        #plt.imshow(image[0])
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
            #print(tr)
            #plt.imshow(transformed_image[0])
            count +=1
            
            #save image
            norm = transforms.Normalize (mean=[0], std=[1/255] )
            save_img_path = os.path.join(save_results, file_name)
            
            cv2.imwrite(save_img_path, norm(transformed_image).numpy()[0])
            #plt.imshow(norm(transformed_image).numpy()[0], cmap='gray', vmin=0, vmax=255)
            #plt.savefig(save_img_path)
            #write_png(norm(transformed_image).to(torch.uint8), save_img_path)
            img_ids_labels.append([file_name, label])
    
    return img_ids_labels


def apply_rand_transform(files, transformations, path_to_images, save_results):
    
    save_results = os.path.join(save_results, "random_images")
    if not os.path.exists(save_results):
      os.mkdir(save_results)

    rng = np.random.default_rng()

    count = 0
    img_ids_labels = []

    label =np.unique(files[:, 1])
    if len(label) != 1:
        raise  ValueError("more than one label found")

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

        #print("rand")
        #plt.imshow(image[0])
        
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

            #print(tr)
            #plt.imshow(transformed_image[0])

        count +=1

        # save image
        norm= transforms.Normalize (mean=[0], std=[1/255] )
        save_img_path = os.path.join(save_results, file_name)
        
        cv2.imwrite(save_img_path, norm(transformed_image).numpy()[0] )
        #plt.imshow(norm(transformed_image).numpy()[0], cmap='gray', vmin=0, vmax=255)
        #plt.savefig(save_img_path)
        #write_png(norm(transformed_image).to(torch.uint8), save_img_path)
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

    # load y txt file
    y = np.loadtxt(path_to_orig_y, dtype=str, delimiter=" ")

    y_aug =sorted(os.listdir(path_to_images))

    results = []
    for ya in y_aug:

        curr_path = ya[:ya.rindex("_")]+".png"
        #print(curr_path)

        label = y[y[:,0] == curr_path][0][1]
        #print(label)

        results.append([ya, label])

    
    results = np.array(results)
    combined = np.concatenate((y, results), axis=0)

    # save new data files txt and combined (y, new data files) txt
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


    #path_to_orig_y="/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train_multi.txt"
    #save_path = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/incomplete_data_augmented"
    #path_to_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/incomplete_data_augmented/final"

   #generate_y_txt(path_to_images, path_to_orig_y, save_path)

    #images = sorted(os.listdir(path_to_images))

    #for img in images:
        
        # read image
    #    path_to_img = os.path.join(path_to_images, img)
    #    image = read_image(path_to_img, ImageReadMode.GRAY).float()
    #    print(f"img {img} shape: {image.shape}")
