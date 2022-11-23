import numpy as np
import cv2 
import os

def load_data(path_to_data, path_to_labels=None, save_images_path=None, save_labels_path=None, color=0, verbose=False):
    """
        Loads raw data 

        Parameters:
            path_to_data: path to the folder containing .png files or path to the npy file
            path_to_labels: path to the labels associated with the images. This could be a .txt file with each row containing
                            file name(ends with.png) and its label or path to npy file
            save_images_path: path to images.npy where the raw images will be saved.
            save_labels_path: path to labels.npy where the corresponding labels will be saved 
            color: 1 for BGR color or 0 for grayscale
            verbose: True/False
        
        Returns:
            data: numpy array containing all images of shape (num_of_images, height, width, num_of_channels)
            labels: numpy array with image ids (axis=0) and/or image labels(axis=1) of shape (num_of_images, 1) or (num_of_images, 2)
    
        Additional Notes:
            - if 'path_to_data' points to a pickle file, all other parameters except 'path_to_labels' will be ignored
            - if 'path_to_data' points to a folder, only .png files will be considered.
            - When loading data from the folder, if 'save_images_path' or 'save_labels_path' is not provided, the npy file will be saved to 'path_to_data/images.npy' or 'path_to_data/labels.npy'
            - When loading data from npy file, if 'save_labels_path' is not given, an empty array is returned for the labels

            - (Tested by Ahmad on his computer and kaggle) This code is not suitable for reading colored images as it writes to allocate large amount of memory and the kernel crashes.
            - for now, better to use it for gray scale. In later stage, maybe make a generator out of it so that images are loadad on demand instead of loading all of images in the memory
    """     

    # load npy files
    if ".npy" in path_to_data:
        
        # load images 
        data = np.load(path_to_data)

        print(f"Loaded Images from {path_to_data} successfully.") 
        print(f"The shape of images: {data.shape}.")
        
        # load labels
        if path_to_labels:
            if ".npy" in path_to_labels:
                img_ids_labels = np.load(path_to_labels)
            else:
                raise ValueError("When loading data from .npy file, path to labels must be also point to labels.npy file")
           
            if verbose:
                print(f"Loaded Labels from {path_to_labels}.")
                print(f"The shape of labels: {img_ids_labels.shape}")
            
        
        else:

            print("Returning empty array for the labels ")

            img_ids_labels = np.array([])

    # load data from the folder
    else:

        # get image ids
        image_ids = [id for id in sorted(os.listdir(path_to_data)) if ".png" in id]
        if not image_ids:
            raise ValueError(f"No Images(.png) found in {path_to_data}")
        
        print(f"{len(image_ids)} number of images found.")
        
        # get image labels
        if path_to_labels:
            img_ids_labels = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")
            img_ids_labels = np.sort(img_ids_labels, axis=0)

            # make sure number of images are equal to number of labels provided
            if img_ids_labels.shape[0] != len(image_ids):
                raise ValueError(f"Number of images ({len(image_ids)}) are not equal to number of provided labels ({img_ids_labels.shape[0]})")
            
            print(f"Unique labels found in the txt file: {np.unique(img_ids_labels[:,1])}")

        else:
            img_ids_labels = np.array(image_ids)
            img_ids_labels = np.expand_dims(img_ids_labels, axis=-1)

            print(f"No labels found.")


        for index, img_id in enumerate(image_ids):

            # read image
            path_to_image = os.path.join(path_to_data, img_id)
            img = cv2.imread(path_to_image, color)

            if verbose:
                print(f"Loaded image: {img_id} Shape: {img.shape}")

            # add dim to image for grayscale 
            if not color:
                img = np.expand_dims(img, axis=-1)


            # initialize data for storing images
            if index == 0:
                data = np.zeros(((len(image_ids), )+ img.shape))
                temp_shape = img.shape
            
            # make sure that every image has same shape
            if img.shape != temp_shape:
                raise ValueError(f"Unknown shape encountered. Image ID: {img_id} shape: {img.shape}")
            
            # make sure there is one to one mapping between image IDs in the folder and in the text file so that label allocation is done correctly
            if path_to_labels:
                if img_id != img_ids_labels[index][0]:
                    raise ValueError(f"Image IDs in the folder as well as in the text file are sorted. However, the image ID at {index} do not match with each other. \n Image ID in the folder:{img_id} \n Image ID in the text file {img_ids_labels[index]}.")
            
            #store image        
            data[index] = img

        # save images
        if not save_images_path:
            save_images_path = os.path.join(path_to_data, "images.npy")
        np.save(save_images_path, data)
        

        print(f"Final shape of images: {data.shape}")
        print(f"Saved images to {save_images_path}")


        # save labels
        if not save_labels_path:
            save_labels_path = os.path.join(path_to_data, "labels.npy")
        np.save(save_labels_path, img_ids_labels)
        

        print(f"Final shape of labels: {img_ids_labels.shape}")
        print(f"Saved labels to {save_labels_path}")

    return data, img_ids_labels

if __name__ == "__main__":
    # path to data folders
    path_to_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data"
    path_to_train_data = path_to_data + "/train"
    path_to_train_labels = path_to_data + "/train.txt"

    path_to_test_data = path_to_data + "/test"

    path_to_noisy_test_data = path_to_data + "/noisy_test"
    

    # save paths
    save_path = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/processed_data"
    train_npy = save_path + "/train_colored.npy"
    train_labels_npy = save_path + "/train_colored_labels.npy"
    
    tests_npy = save_path + "/tests_colored.npy"
    tests_labels_npy = save_path + "/tests_colored_labels.npy"
    
    noisy_tests_npy = save_path + "/noisy_tests_colored.npy"
    noisy_tests_labels_npy = save_path + "/noisy_tests_colored_labels.npy" 

    # load train data
    print("Working on training data")
    train_img, train_labels = load_data(path_to_data=path_to_train_data,
                                        path_to_labels=path_to_train_labels,
                                        save_images_path=train_npy,
                                        save_labels_path=train_labels_npy,
                                        color=1,
                                        verbose=True)
    
    print(f"Processing Completed for training: Image: {train_img.shape} Labels: {train_labels.shape}")
    
    # load test data
    print("Working on test data")
    test_img, test_labels = load_data(path_to_data=path_to_test_data,
                                            save_images_path=tests_npy,
                                            save_labels_path=tests_labels_npy,
                                            color=1,
                                            verbose=True)
        
    print(f"Processing Completed for test: Image: {test_img.shape} Labels: {test_labels.shape}")

    #load noisy data
    print("Working on noisy test data")
    noisy_test_img, noisy_test_labels = load_data(path_to_data=path_to_noisy_test_data,
                                        save_images_path=noisy_tests_npy,
                                        save_labels_path=noisy_tests_labels_npy,
                                        color=1,
                                        verbose=True)
    
    print(f"Processing Completed for noisy_test: Image: {noisy_test_img.shape} Labels: {noisy_test_labels.shape}")
        
    
