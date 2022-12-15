"""
generates feature vector using batches

"""


def generate_feature_vector_using_batches(pipeline):
    """
    Generates feature vector using batches:

    Parameters:
    pipeline:
    
    Returns: 
    feature vectors:
    ys
    config:
    """

    from utils.data_preprocessing.split_data import split_data
    import pickle
    import os 
    import numpy as np
    
    pipeline_ops = pipeline.keys()
    
    output_config ={}
    output_config["data"] = {}
    output_config["data_preprocessing"] = {}
     
    # Set up default values for data ops
    data = pipeline["data"]
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

    # get batch size
    if "batch_size" in pipeline_ops:
        batch_size = pipeline["batch_size"]
    else:
        raise ValueError("'batch_size' must be provided in the pipeline.")
    
    output_config["batch_size"] = batch_size

    # for making sure that splitting does not take place for testing data
    if split_type and not path_to_labels:
        raise ValueError ("Parameter 'split_type' is provided but 'path_to_labels' is not provided. ")

    #whether to save features or not
    if "save_to_pkl" in pipeline_ops:
        save_pkl = pipeline["save_to_pkl"]
    else:
        save_pkl = False
    output_config["save_to_pkl"] = save_pkl

    #path to results
    if "path_to_results" in pipeline_ops:
        path_to_results = pipeline["path_to_results"]
    else:
        path_to_results = False
    output_config["path_to_results"] = path_to_results

    # result dir
    if not os.path.exists(path_to_results):
        os.mkdir(path_to_results)
        print(f"Created directory {path_to_results}")
    else:
        print(f"Warning: {path_to_results} already exists. Contents may get overwritten")

    # load from the folder
    if os.path.isdir(path_to_images):
        
        # --------------------get image ids--------------------
        image_ids = [id for id in os.listdir(path_to_images) if ".png" in id]
        if image_ids:
            print(f"{len(image_ids)} images found.")
        else:    
            raise ValueError(f"No Image(.png) found in {path_to_images}")


        # --------------------------get image labels------------------
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
        
        #------------------- training--------------------------------
        
        if split_type:
            print( f"Spliting the data: type: {split_type}")
            
            y_train, y_valid = split_data(y=img_ids_labels, split_type=split_type)
            
            print(f"Shape of y_train: {y_train.shape} y_valid:  {y_valid.shape}")
            
            num_folds = y_train.shape[0]

            train_labels = {}
            valid_labels = {}
            for fold_no in range(num_folds):
                
                # ratio in y_train
                train_labels[str(fold_no)] = {}
                unique_labels, counts =np.unique(y_train[fold_no, :,1], return_counts=True)
                
                print(f"\nFold No: {fold_no} Training data: Unique labels: ")
                
                for ul, c in zip(unique_labels, counts):
                    train_labels[str(fold_no)][ul] = format(c*100/y_train.shape[1], '.2f')
                    print(f"Label: {ul}, %age: {format(c*100/y_train.shape[1], '.2f')}")
                
                # ratio in y_valid
                valid_labels[str(fold_no)] = {}
                unique_labels, counts =np.unique(y_valid[fold_no, :,1], return_counts=True)
                
                print(f"\nFold No: {fold_no} Validation data: Unique labels: ")

                for ul, c in zip(unique_labels, counts):
                    valid_labels[str(fold_no)][ul] = format(c*100/y_valid.shape[1], '.2f')
                    print(f"Label: {ul}, %age: {format(c*100/y_valid.shape[1], '.2f')}")
            
            output_config["processed_labels"] = {}
            output_config["processed_labels"]["train"] = train_labels
            output_config["processed_labels"]["valid"] = valid_labels
            
            
            # generate features
            for fold_no in range(num_folds):
                
                
                print("Processing Images of Fold No: ", fold_no)
                
                print ("Processing Training data")
                #training data
                features_train_fold, features_train_config = get_features(path_to_images=path_to_images,
                            y=y_train[fold_no], 
                            batch_size=batch_size, 
                            data_preprocessing=pipeline["data_preprocessing"], 
                            feature_extractors=pipeline["feature_extractors"])
                            

                print(f"Generated Features for training data successfully. Shape: {features_train_fold.shape}, y: {y_train.shape}")

                print ("Processing validation data")
                #valid data
                features_valid_fold, _ = get_features(path_to_images=path_to_images,
                            y=y_valid[fold_no], 
                            batch_size=batch_size, 
                            data_preprocessing=pipeline["data_preprocessing"], 
                            feature_extractors=pipeline["feature_extractors"])

                #if not compare_dict(features_train_config, features_valid_config):
                #    raise ValueError("during feature generation, features config values got changed")

                print(f"Generated Features for validation data successfully. Shape: {features_valid_fold.shape}, y: {y_valid.shape}")

                if fold_no ==0:
                    features_train = features_train_fold
                    features_valid = features_valid_fold
                else:
                    features_train =np.concatenate((features_train, features_train_fold), axis=0)
                    features_valid =np.concatenate((features_valid, features_valid_fold), axis=0)
                
            if save_pkl:
                if path_to_results:
                    
                    path_to_processed_dataset = path_to_results + "/processed_features.pkl"
                    print("Saving processed features to ", path_to_processed_dataset)

                    with open(path_to_processed_dataset, 'wb') as file:
                        pickle.dump([features_train, y_train, features_valid, y_valid], file)
                else:
                    raise ValueError("Path to results must be provided when saving features to pkl")

            print("Shape training features: ", features_train.shape)
            print("Shape valdidation features: ", features_valid.shape)
            output_config = {**output_config, **features_train_config}

            return features_train, y_train, features_valid, y_valid, output_config

        else:

            #------------------------testing----------------------------
            y = img_ids_labels
            
            print ("Processing Testing data")
            
            #get features
            features, features_config = get_features(path_to_images=path_to_images,
                        y=y, 
                        batch_size=batch_size, 
                        data_preprocessing=pipeline["data_preprocessing"], 
                        feature_extractors=pipeline["feature_extractors"])

            if save_pkl:
                if path_to_results:
                    path_to_processed_dataset = path_to_results + "/processed_features.pkl"
                    print("Saving processed features to ", path_to_processed_dataset)
                    with open(path_to_processed_dataset, 'wb') as file:
                        pickle.dump([features, y], file)
                else:
                    raise ValueError("Path to results must be provided when saving features to pkl")

            print("Shape of features: ", features.shape)

            output_config = {**output_config, **features_config}
            
            y= np.expand_dims(y, axis=0)

            return features, y, output_config
    else:
        raise ValueError("path_to_images must be a directory")


def get_features(path_to_images, y, batch_size, data_preprocessing, feature_extractors):
    """
    Load images,
    extracts features, using batches
    
    Parameters:
        path_to_images: path to the folder
        y = (num_images, 2) or (num_images, 1)
        batch_size: int
        data_preprocessing: dict with preprocessing as keys
        feature_extractors: dict with feature extractos as keys
    
    Returns:
        features: (1, num_images, num_features)
        config with data_preprocessing and feature_extractor keys 
    """

    from utils.data_preprocessing.split_data import get_batch
    from utils.load_and_preprocess_data import load_images
    from utils.generate_feature_vector import generate_feature_vector
    import numpy as np

    verbose =False

    batch_no = 0
    output_config = {}
                
    for y_batch in get_batch(y, batch_size):
        if verbose:
            print(f"Batch NO: {batch_no} y: {y_batch}")

        images_batch, images_batch_config = load_images(path_to_images=path_to_images,
                    image_ids=y_batch[:,0], 
                    data_preprocessing=data_preprocessing)
        if verbose:
            print(f"\nBatch No: {batch_no} Shape of Images: {images_batch.shape}")
        
        # jsut for stying consistent with what i have before
        images_batch = np.expand_dims(images_batch, axis=0)
        if verbose:
            print("\nGenerating Features . . . ")
        features_batch, features_batch_config = generate_feature_vector(X=images_batch,
                                                                                    extractors=feature_extractors)
        
        if verbose:
            print(f"Generated Features for the batch successfully. Shape: {features_batch.shape}")

        if batch_no ==0:
            features = features_batch
        else:
            features = np.concatenate((features, features_batch), axis=1)
       
        if verbose:
            print("features shape: ", features.shape)
      
        batch_no = batch_no +1
    
    output_config["data_preprocessing"] = images_batch_config
    output_config["feature_extractors"] = features_batch_config

    return features, output_config
