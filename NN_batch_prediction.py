"""
Generates prediction using batches

"""
# import libraries
import os

from utils.load_and_preprocess_data import load_and_preprocess_data
from utils.data_preprocessing.change_colorspace import change_colorspace
from utils.data_preprocessing.resize import resize
from utils.data_preprocessing.normalize import normalize
from utils.data_preprocessing.edge_detector import canny_edge_detector


from utils.generate_feature_vector import generate_feature_vector
from utils.generate_feature_vector_using_batches import generate_feature_vector_using_batches

from utils.feature_extractors.contrast import calculate_contrast
from utils.feature_extractors.kurtosis import calculate_kurtosis
from utils.feature_extractors.skewness import calculate_skew
from utils.feature_extractors.histogram import calculate_histogram
from utils.feature_extractors.haralick import calculate_haralick
from utils.feature_extractors.zernike import calculate_zernike
from utils.feature_extractors.non_zero_valules import count_nonzeros
from utils.feature_extractors.local_binary_pattern import calculate_lbp
from utils.feature_extractors.wavelet import feature_GLCM


from utils.normalize_features import normalize_features

from utils.apply_nn_classifiers import apply_nn_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree
from utils.classifiers.boosting import boosting

from utils.json_processing import load_from_json, save_to_json

from utils.misc import add_function_names, replace_function_names, generate_txt_file

def NN_batch_prediction(path_to_images, path_to_json, save_path, batch_size=None):
    """
    Generates prediction for the given data set

    Parameters:
        path_to_images: path to the folder containing images on which prediction will be generated
        path_to_json: path to the training pipeline json
        save_path: path to the folder where the results will be stored
        batch_size: since images are of different shape in noisy data, so cannot concatenate them. can set different batch size for prediction from trianing

    """
    if not os.path.isdir(path_to_images):
        raise ValueError("'path_to_images' must be a directory")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created directory {save_path}")
    else:
        print(f"Warning: {save_path} already exists. Content may get overwritten")

    # loading training pipeline
    pipeline = load_from_json(path_to_json)

    if batch_size:
        pipeline["batch_size"] = batch_size

    pipeline["data"]["path_to_images"] = path_to_images
    pipeline["path_to_results"] = save_path

    del pipeline["data"]['path_to_labels']

    del pipeline["data"]["split_type"]

    # replace function names with their pointers    
    data_preprocessing = [normalize, change_colorspace, resize, canny_edge_detector]
    feature_extractors= [calculate_contrast, calculate_kurtosis, calculate_skew, calculate_histogram, calculate_haralick, 
                        calculate_zernike,  count_nonzeros, calculate_lbp, feature_GLCM]
    norm_features = [normalize_features]
    classifiers = [svm, rftree, boosting]

    replace_function_names(
        pipeline, 
        functions = data_preprocessing + feature_extractors + norm_features + classifiers)


    # load images
    print("\nGenerating Features . . . ")
    features, y, features_config = generate_feature_vector_using_batches(pipeline)

    print(f"Generated Features for the data Sucessfully. Shape of feature: {features.shape} y: {y.shape}")
    output_config = features_config
   
    # normalize features
    print("\nNormalizing Features for the data . . . ")
    features_norm, features_norm_config = normalize_features(
        features=features, 
        parameters=pipeline["normalize_features"]
        )
    print(f"Normalized Features for the data successfully. Shape: {features_norm.shape}")

    print("\nGenerating labels for the data . . . ")
    y_pred, y_pred_probs, networks_config, networks_list = apply_nn_classifiers(
        X=features_norm, 
        y=y, 
        classes=pipeline["data"]["classes"], 
        networks=pipeline["networks"], 
        return_probs = pipeline["return_probs"]
        )
    output_config["data"]["classes"] = pipeline["data"]["classes"]
    output_config["path_to_results"] = save_path
    print(f"Generated labels for the data successfully. Shape: {y_pred.shape}")

    results_dict =output_config
    results_dict["normalize_features"] = features_norm_config
    results_dict["networks"] = networks_config

    # add function names 
    print("\nReplacing Function names in the dict")
    add_function_names (results_dict)

    # save pipeline
    print("\nsaving pipeline dictionary to json")
    save_to_json(
        results_dict, 
        save_path +"/pipeline"
        )
    
    # save labels
    print("\nGenerating text file for the data")
    generate_txt_file(
        y=y_pred, 
        path_to_results=save_path, 
        classifiers=networks_list, 
        name_of_file="labels",
        y_probs = y_pred_probs
        )

if __name__ == "__main__":
    
    path_to_json = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/NN_test/train/training_pipeline.json"
    
    #path_to_noisy_test_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/noisy_test"
    #save_path_noisy_test = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/prediction_test/noisy_test"

    path_to_test_images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/test"
    save_path_test = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/NN_test/test"
    
    
    NN_batch_prediction(
        path_to_json=path_to_json,
        path_to_images=path_to_test_images,
        save_path=save_path_test
    )