import numpy as np
import cv2 
import os
import pickle
from copy import deepcopy

from utils.compare_dicts import compare_dict
from utils.data_preprocessing.change_colorspace import change_colorspace
from utils.data_preprocessing.resize import resize
from utils.data_preprocessing.normalize import normalize
from utils.data_preprocessing.split_data import split_data

def load_and_preprocess_data(data_config, path_to_results=None):
    """
        Loads and preprocess data

        Parameters:
            path_to_results= path to the folder where processed data will be saved (required when saving to npy)

            data_config: dictionary with this structure:

                data_config["data"]: dictionary with following keys
                    path_to_images: path to the folder containing .png files (or .npy)
                    path_to_labels: path to the labels associated with the images. This could be a .txt file with each row containing
                                    file name(ends with.png) and its label (not required when reading images from .npy)
                    split_type: how to split the data : "simple", "simpleStratified" "kfold", "kfoldStratified"
                    save_to_pkl: whether to save results in npy format or not

                data_config["data_preprocessing"]: dictionary with following structure:
                    preprocessing["name_of_processing_1"]["function"] = pointer to the function
                    preprocessing["name_of_processing_1"]["parameter1"] = value
                    preprocessing["name_of_processing_1"]["parameter2"] = value
                    .
                    .
                    preprocessing["name_of_processing_2"]["function"] = pointer to the function
                    preprocessing["name_of_processing_2"]["parameter1"] = value
                    preprocessing["name_of_processing_2"]["parameter2"] = value
                    
        
        Returns:
            data: numpy array of shape (folds, num_of_images, height, width, num_of_channels)
            labels: numpy array with image ids (axis=0) and/or image labels(axis=1) of shape (fold, num_of_images, 1) or (fold, num_of_images, 2)
            output_config: dictionary with parameters of the function (including default parameters)

        Additional Notes:
           Images are initially read in BGR format  with shape (1, Height, width, color_channels) in uint8 format.
           Furthermore, this function load all images in the memory. For large dataset, it can make python kernel crash. 
                
        """ 

    verbose=False

    # setup output config and make sure data_config has correct structure
    output_config ={}

    for k in ["data", "data_preprocessing"]:
        if k in data_config.keys():
            output_config[k] = {}
        else:
            raise ValueError(f"'{k}' must be provided in 'data_config'")

    # Set up default values for data options
    data = data_config["data"]
    data_keys = data.keys()
    
    
    if "path_to_images" in data_keys:
        path_to_images = data["path_to_images"]
        output_config["data"]["path_to_images"] = path_to_images
    else:
        raise ValueError("'path_to_images' must be provided in the parameter 'data'.")
    
    
    if "path_to_labels" in data_keys:
        path_to_labels = data["path_to_labels"]
    else:
        path_to_labels = None
    output_config["data"]["path_to_labels"] = path_to_labels

    
    if "split_type" in data_keys:
        split_type = data["split_type"]
    else:
        split_type = None
    output_config["data"]["split_type"] = split_type

    # for making sure that splitting does not take place for testing data
    if split_type and not path_to_labels:
        raise ValueError ("Parameter 'split_type' is provided but 'path_to_labels' is not provided. ")


    #save to pkl
    if "save_to_pkl" in data_keys:
        save_pkl = data["save_to_pkl"]
    else:
        save_pkl = False
    output_config["data"]["save_to_pkl"] = save_pkl

    # load from the folder
    if os.path.isdir(path_to_images):

        # get image ids
        image_ids = [id for id in os.listdir(path_to_images) if ".png" in id]
        if not image_ids:
            raise ValueError(f"No Image(.png) found in {path_to_images}")

        print(f"{len(image_ids)} images found.")
        
        # get image labels
        if path_to_labels:

            img_ids_labels = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")

            # make sure number of images and names of files are same in dir and .txt file 
            if not (sorted(image_ids) == sorted(img_ids_labels[:, 0])):
                raise ValueError("Files in directory do not match with .txt file.")

            # get unique labels and their counts
            data_labels = {}
            unique_labels, counts =np.unique(img_ids_labels[:,1], return_counts=True)
            print(f"Unique labels found and their frequencies: ")
            for ul, c in zip(unique_labels, counts):
                data_labels[ul] = format(c*100/img_ids_labels.shape[0], '.2f')
                print(f"Label: {ul}, %age: {format(c*100/img_ids_labels.shape[0], '.2f')}")

            output_config["original_labels"] = data_labels
        else:
            img_ids_labels = np.array(image_ids)
            img_ids_labels = np.expand_dims(img_ids_labels, axis=-1)

            print(f"No labels found.")


        if split_type:
            print( f"Spliting the data: type: {split_type}")
            
            y_train, y_valid = split_data(y=img_ids_labels, split_type=split_type)

            print(f"Shape of y_train: {y_train.shape} y_valid:  {y_valid.shape}")
            
            
            num_folds = y_train.shape[0]

            X_train= []
            X_valid = []
            for fold_no in range(num_folds):
                print("Loading Images of Fold No: ", fold_no)
                
                train_images, train_config = load_images(path_to_images=path_to_images, 
                                                        image_ids=y_train[fold_no,:,0],
                                                        data_preprocessing=data_config["data_preprocessing"])

                print(f"\nFold No: {fold_no} Shape of Training Images: {train_images.shape}")

                valid_images, valid_config = load_images(path_to_images=path_to_images, 
                                            image_ids=y_valid[fold_no,:,0],
                                            data_preprocessing=train_config)
                
                print(f"\nFold No: {fold_no} Shape of Validation Images: {valid_images.shape}")

                if not compare_dict(train_config, valid_config):
                    raise ValueError("While loading images for validation data, a function changed the values of the configuration")

                # store images
                X_train.append(train_images)
                X_valid.append(valid_images)
            
            
            X_train = np.array(X_train)
            X_valid = np.array(X_valid)

            output_config ["data_preprocessing"] = train_config

            train_labels = {}
            valid_labels = {}
            for fold_no in range(y_train.shape[0]):
                
                # ratio of labels in y_train
                train_labels[str(fold_no)] = {}
                unique_labels, counts =np.unique(y_train[fold_no, :,1], return_counts=True)
                
                print(f"\nFold No: {fold_no} Training data: Unique labels: ")
                
                for ul, c in zip(unique_labels, counts):
                    train_labels[str(fold_no)][ul] = format(c*100/y_train.shape[1], '.2f')
                    print(f"Label: {ul}, %age: {format(c*100/y_train.shape[1], '.2f')}")
                
                # ratio of ylabels in y_valid
                valid_labels[str(fold_no)] = {}
                unique_labels, counts =np.unique(y_valid[fold_no, :,1], return_counts=True)
                
                print(f"\nFold No: {fold_no} Validation data: Unique labels: ")

                for ul, c in zip(unique_labels, counts):
                    valid_labels[str(fold_no)][ul] = format(c*100/y_valid.shape[1], '.2f')
                    print(f"Label: {ul}, %age: {format(c*100/y_valid.shape[1], '.2f')}")
    
            output_config["processed_labels"] = {}
            output_config["processed_labels"]["train"] = train_labels
            output_config["processed_labels"]["valid"] = valid_labels

            if save_pkl:
                if path_to_results:
                    print("Saving processed data to ", path_to_results)
                    path_to_processed_dataset = path_to_results + "/processed_data.pkl"
                    
                    with open(path_to_processed_dataset, 'wb') as file:
                        pickle.dump([X_train, y_train, X_valid, y_valid], file)
                else:
                    raise ValueError("Path to results must be provided when saving to npy")

            return X_train, y_train, X_valid, y_valid, output_config
        
        else:

            y = img_ids_labels
            X, config = load_images(path_to_images=path_to_images,
                                        image_ids=y[:,0],
                                        data_preprocessing=data_config["data_preprocessing"])
            X = np.expand_dims(X, axis=0)
            
            if not compare_dict(data_config["data_preprocessing"], config):
                raise ValueError("While loading images for the data, a function changed the values of the configuration")
              
            output_config["data_preprocessing"] = config

            y= np.expand_dims(y, axis=0)
            
            # save to npy
            if save_pkl:
                if path_to_results:
                    print("Saving processed data to ", path_to_results)
                    path_to_processed_dataset = path_to_results + "/processed_data.pkl"
                    
                    with open(path_to_processed_dataset, 'wb') as file:
                        pickle.dump([X, y], file)
                else:
                    raise ValueError("Path to results must be provided when saving to npy")
            
            return X, y, output_config

    
    elif ".pkl" in path_to_images:
        print("Reading data from npy")

        with open(path_to_images, 'rb') as file:
            datas = pickle.load(file)
            if len(datas) == 4:
                #X_train, y_train, X_valid, y_valid
                return datas [0], datas [1], datas [2], datas [3], output_config
            
            elif len(datas) == 2:
                # X, y
                return datas[0],datas[1], output_config
            else:
                raise ValueError("Unknown value for length of data in pickle encountered")
    else:
        raise ValueError("Unrecognised Value for the parameter 'path_to_images'. ")




def load_images(path_to_images, image_ids, data_preprocessing):
    """
    Loads images and applies preprocesssing

    Parameters:
        path_to_images: path to the folder containing .png files
        image_ids:  numpy array with image ids of shape(num_of_images,)
        data_preprocessing: dictionary with following structure:
                    preprocessing["name_of_processing_1"]["function"] = pointer to the function
                    preprocessing["name_of_processing_1"]["parameter1"] = value
                    preprocessing["name_of_processing_1"]["parameter2"] = value
                    .
                    .
                    preprocessing["name_of_processing_2"]["function"] = pointer to the function
                    preprocessing["name_of_processing_2"]["parameter1"] = value
                    preprocessing["name_of_processing_2"]["parameter2"] = value

    Returns:
        images:numpy array of shape (num_images, H, W, channels)
        config: dictionary with parameters of the function (including default parameters)
    """
    verbose=False
    # for storing preprocessing default values
    config = {}

    for index, img_id in enumerate(image_ids):

        # read image
        path_to_image = os.path.join(path_to_images, img_id)
        img = cv2.imread(path_to_image)
        if verbose:
            print(f"\nLoaded image {img_id} Shape: {img.shape} ")
        
        # make sure that every image has same shape
        if index == 0:
            temp_shape = img.shape
        
        #if img.shape != temp_shape:
           # print(f"Warning: Different shape encountered. Image ID: {img_id} shape: {img.shape}")

        img = np.expand_dims(img, axis=0)  

        # apply preprocessing on the data
        for preprocessing in data_preprocessing.keys():

            fnt_pointer = data_preprocessing[preprocessing]["function"]
            if verbose:
                print(f"Applying Data Preprocessing: {preprocessing}")
           
            img, fnt_config = fnt_pointer(img, data_preprocessing[preprocessing])
            #print(img.shape)            
            
            fnt_config["function"] = fnt_pointer

            config[preprocessing] = fnt_config

        #store image      
        if index == 0:
            images = img
        else:
            images = np.concatenate((images, img))

    
    return images, config


if __name__ == "__main__":

    # training data

    # save paths
    # actual data
    #data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train"
    #label = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train.txt"
       
    pipeline = {}

    #------------------ setup data------------------------
    pipeline["data"] = {}

    # path to folder containing images 
    pipeline["data"]["path_to_images"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
    # can be to .txt
    pipeline["data"]["path_to_labels"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train_multi.txt"

    # split data
    pipeline["data"]["split_type"] =  "simple" #"simple" #"simpleStrafied", "kfold", "kfoldStratified"


    # ---------------------------------set up data preprocessing methods and parameters------------------------------------

    pipeline["data_preprocessing"] ={}

    # normalize image
    pipeline["data_preprocessing"]["normalize_image"] = {}
    pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
    pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

    # map to RGB
    pipeline["data_preprocessing"]["map_to_grayscale"] = {}
    pipeline["data_preprocessing"]["map_to_grayscale"]["function"] = change_colorspace 
    pipeline["data_preprocessing"]["map_to_grayscale"]["conversion"] ="BGR2GRAY" 

    # resize_image
    pipeline["data_preprocessing"]["resize_image"] = {}
    pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
    pipeline["data_preprocessing"]["resize_image"]["output_size"] = (100,100) #(width, height)
    pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"



    X_train, y_train, X_valid, y_valid, output_config = load_and_preprocess_data(data_config=deepcopy(pipeline))
        
    print("X_train ", X_train.shape)
    print("y_train ", y_train.shape)
    print("X_valid ", X_valid.shape)
    print("y_valid ", y_valid.shape)
    print("output config ", output_config)
    

    """
    #test data
    pipeline = {}

    pipeline["data"] = {}

    # can be to folder 
    pipeline["data"]["path_to_images"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/noisy_test"

    # ---------------------------------set up data preprocessing methods and parameters------------------------------------
    #default values for parameters not guven -8way of determining whether its training or predictin
    pipeline["data_preprocessing"] ={}

    # normalize image
    pipeline["data_preprocessing"]["normalize_image"] = {}
    pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
    pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

    # map to RGB
    pipeline["data_preprocessing"]["map_to_RGB"] = {}
    pipeline["data_preprocessing"]["map_to_RGB"]["function"] = change_colorspace 
    pipeline["data_preprocessing"]["map_to_RGB"]["conversion"] ="BGR2GRAY" 

    # resize_image
    pipeline["data_preprocessing"]["resize_image"] = {}
    pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
    pipeline["data_preprocessing"]["resize_image"]["output_size"] = (50,50) #(width, height)
    pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"



    X, y, config = load_and_preprocess_data(data_config=deepcopy(pipeline))
        
    print("X ", X.shape)
    print("y ", y.shape)
    print("output config ", config)

    """