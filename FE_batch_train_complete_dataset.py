"""
Trains different classifiers using the features obtained from different feature extractors on the complete dataset (no splitting is performed).
Applies data preprocessing and feature extractors on batch of images.
Also, generates prediction on test and noisy dataset.

Performs both binary and multiclass classification

"""

# ------------------------------import libraries----------------------------
import numpy as np
import os
from copy import deepcopy
import pickle

from utils.compare_dicts import compare_dict

from utils.load_and_preprocess_data import load_images
from utils.data_preprocessing.split_data import split_data, get_batch
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

from utils.normalize_features import normalize_features

from utils.apply_classifiers import apply_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree

from utils.evaluate_metrics import evaluate_metrics
from utils.metrics.accuracy import accuracy
from utils.metrics.f1_score import F1_score
from utils.metrics.mcc import mcc
from utils.metrics.precision import precision
from utils.metrics.sensitivity import sensitivity

from utils.create_plots import create_plots
from utils.plots.plot_CM import plot_CM

from utils.json_processing import save_to_json

from utils.misc import add_function_names, generate_txt_file, save_results, change_txt_for_binary

from FE_batch_prediction import FE_batch_prediction

def FE_batch_train_complete_dataset(pipeline):
    """
    Trains models using complete dataset and generates results

    Parameters:
        pipeline: dict
    
    """

    #  load images and generate features
    features, y, features_config = generate_feature_vector_using_batches(pipeline)
    output_config = features_config

    # normalize vectors
    print("\nNormalizing Features for training data . . . ")
    features_norm, features_norm_config = normalize_features(
        features=features, 
        parameters=pipeline["normalize_features"]
        )
    print(f"Normalized Features for the data successfully. Shape: {features_norm.shape}")


    # train model

    print("\nGenerating labels for the data . . . ")
    y_pred, classifiers_config, classifers_list = apply_classifiers(
        X=features_norm, 
        y=y, 
        classes=pipeline["data"]["classes"], 
        classifiers=pipeline["classifiers"], 
        path_to_results= pipeline["path_to_results"]
        )
    output_config["data"]["classes"] = pipeline["data"]["classes"].tolist()
    output_config["path_to_results"] = pipeline["path_to_results"]
    print(f"Generated labels for the data successfully. Shape: {y_pred.shape}")

    print("\nEvaluating Metrics on the data . . . ")
    metrics, metrics_config, metrics_list = evaluate_metrics(
        y_true=y, 
        y_pred=y_pred, 
        metrics=pipeline["metrics"],
        classifiers = classifers_list,
        )
    print("\nResults")

    for cl_no, cl in enumerate(classifers_list):

        for fold_no in range(metrics.shape[1]):
            
            for met_no, met in enumerate(metrics_list):

                print(f"Classifer: {cl} Fold No: {fold_no} Metric: {met}  Score: {metrics[cl_no, fold_no, met_no]} ")

    print(f"Evaluated Metrics on the the data successfully. Shape:{metrics.shape}")

    print("\nCreating Plots for training data . . .")
    plots_config = create_plots(
        y_true=y, 
        y_pred=y_pred, 
        plots= pipeline["plots"], 
        path_to_results=pipeline["path_to_results"],
        path_to_images="place_holder",
        classifiers=classifers_list,
        name_of_file = "train"
        )
    print("Created Plots for the data")

    # concencate all the returuned configs and save it to json
    output_config["normalize_features"] = features_norm_config
    output_config["classifiers"] = classifiers_config
    output_config["metrics"] = metrics_config
    output_config["plots"] = plots_config

    # add function names 
    print("\nReplacing Function names in the dict")
    add_function_names (output_config)

    # save pipeline
    print("\nsaving pipeline dictionary to json")
    save_to_json(
        output_config, 
        pipeline["path_to_results"]+"/training_pipeline"
        )

    # save labels
    print("\nGenerating text file for the data")
    generate_txt_file(
        y=y_pred, 
        path_to_results=pipeline["path_to_results"], 
        classifiers=classifers_list, 
        name_of_file="train"
        )

        # save metrics train and metrics_valid
    print("\nSaving the results ")
    save_results(
        results=metrics,
        classifiers=classifers_list,
        metrics=metrics_list,
        path_to_results=pipeline["path_to_results"],
        name_of_file="train"
    )

    print("\nTraining Completed\n")


if __name__ == "__main__":
    #--------------------------------- raw data image dir------------------------------

    images_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data"
    
    #-------------------------------- Results directory----------------------------
    results_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results"
    
    run_name = "haralick_zernike_complete_dataset"
    
    run_path = os.path.join(results_folder, run_name)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
        print(f"Created directory {run_path}")
    else:
        print(f"Warning: {run_path} already exists. Contents may get overwritten")

        
    
    #----------------------------------------Binary Classification--------------------------------
    
    print("----------------------------_Binary Classification------------------------")
    
    # transform label txt file
    print("Generating txt file for binary classification")
    path_to_multi_labels = os.path.join(images_folder, "train_multi.txt")
    path_to_binary_labels = os.path.join(images_folder, "train_binary.txt")
    #change_txt_for_binary(path_to_multi_labels, path_to_binary_labels)
    
    # result dir
    binary_path = os.path.join(run_path, "binary")
    if not os.path.exists(binary_path):
        os.mkdir(binary_path)
        print(f"Created directory {binary_path}")
    else:
        print(f"Warning: {binary_path} already exists. Contents may get overwritten")

    #--------------------------------------Pipeline-------------------------------


    pipeline = {}
    pipeline["path_to_results"] = os.path.join(binary_path, "train")

    # whether to save features to pkl
    pipeline["save_to_pkl"] = True

    # number of images to process at a time
    pipeline["batch_size"] = 500
    
    #------------------ setup data------------------------

    pipeline["data"] = {}

    # path to folder containing images 
    pipeline["data"]["path_to_images"] = os.path.join(images_folder, "train")
    # path to labels.txt
    pipeline["data"]["path_to_labels"] = path_to_binary_labels

    # names of the class
    pipeline["data"]["classes"] =  np.array(["NO_COVID", "COVID"])

    # ---------------------------------set up data preprocessing methods and parameters------------------------------------
    pipeline["data_preprocessing"] ={}

    # normalize image
    #pipeline["data_preprocessing"]["normalize_image"] = {}
    #pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
    #pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

    # map to RGB
    pipeline["data_preprocessing"]["map_to_RGB"] = {}
    pipeline["data_preprocessing"]["map_to_RGB"]["function"] = change_colorspace 
    pipeline["data_preprocessing"]["map_to_RGB"]["conversion"] ="BGR2GRAY" 

    # canny edge detector
    #pipeline["data_preprocessing"]["canny_edges"] = {}
    #pipeline["data_preprocessing"]["canny_edges"]["function"] = canny_edge_detector
    #pipeline["data_preprocessing"]["canny_edges"]["blur"] = True
    #pipeline["data_preprocessing"]["canny_edges"]["threshold1"] =250
    #pipeline["data_preprocessing"]["canny_edges"]["threshold2"] = 500
    #pipeline["data_preprocessing"]["canny_edges"]["apertureSize"] = 5
    #pipeline["data_preprocessing"]["canny_edges"]["L2gradient"] = True

    # resize_image
    pipeline["data_preprocessing"]["resize_image"] = {}
    pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
    pipeline["data_preprocessing"]["resize_image"]["output_size"] = (250, 250) #(width, height)
    pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"


    # ---------------------------------set up feature extractor methods and parameters------------------------------------
    pipeline["feature_extractors"] ={}

    # contrast
    #pipeline["feature_extractors"]["contrast"] = {}
    #pipeline["feature_extractors"]["contrast"]["function"] = calculate_contrast
    #pipeline["feature_extractors"]["contrast"]["method"] = "michelson" 

    # skewness
    #pipeline["feature_extractors"]["skewness"] = {}
    #pipeline["feature_extractors"]["skewness"]["function"] =calculate_skew
    #pipeline["feature_extractors"]["skewness"]["bias"] = True

    # kurtosis
    #pipeline["feature_extractors"]["kurtosis"] = {}
    #pipeline["feature_extractors"]["kurtosis"]["function"] = calculate_kurtosis
    #pipeline["feature_extractors"]["kurtosis"]["method"] = "pearson"
    #pipeline["feature_extractors"]["kurtosis"]["bias"] = True

    # RMS
    #pipeline["feature_extractors"]["RMS"] = {}
    #pipeline["feature_extractors"]["RMS"]["function"] = calculate_contrast
    #pipeline["feature_extractors"]["RMS"]["method"] = "rms"

    # count non zeros
    #pipeline["feature_extractors"]["count_nonzeros"] = {}
    #pipeline["feature_extractors"]["count_nonzeros"]["function"] = count_nonzeros

    # histogram
    #pipeline["feature_extractors"]["histogram"] = {}
    #pipeline["feature_extractors"]["histogram"]["function"] = calculate_histogram
    #pipeline["feature_extractors"]["histogram"]["bins"] = 256
    #pipeline["feature_extractors"]["histogram"]["range"] = (0,256)
    #pipeline["feature_extractors"]["histogram"]["density"] = False

    # haralick
    pipeline["feature_extractors"]["haralick"] = {}
    pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
    pipeline["feature_extractors"]["haralick"]["blur"] = True
    pipeline["feature_extractors"]["haralick"]["distance"] = 10

    # zernike
    pipeline["feature_extractors"]["zernike_moments"] = {}
    pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
    pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
    #pipeline["feature_extractors"]["zernike_moments"]["radius"] = 100
    #pipeline["feature_extractors"]["zernike_moments"]["degree"] = 10
    #pipeline["feature_extractors"]["zernike_moments"]["cm"] = (100, 100)

    #lbp
    #pipeline["feature_extractors"]["lbp"] = {}
    #pipeline["feature_extractors"]["lbp"]["function"] = calculate_lbp
    #pipeline["feature_extractors"]["lbp"]["P"] = 40
    #pipeline["feature_extractors"]["lbp"]["R"] = 12
    #pipeline["feature_extractors"]["lbp"]["method"] = "uniform"


    #---------------------------------Normalize feature vectors-----------------------
    pipeline["normalize_features"] = {}
    pipeline["normalize_features"]["norm_type"] = "StandardScaler"

    # ---------------------------------set up classifiers and parameters------------------------------------
    pipeline["classifiers"] ={}

    # SVM
    pipeline["classifiers"]["svm"] = {}
    pipeline["classifiers"]["svm"]["function"] = svm 
    pipeline["classifiers"]["svm"]["trainAuto"] = False
    pipeline["classifiers"]["svm"]['svm_type'] =  'C_SVC' 
    pipeline["classifiers"]["svm"]['kernel'] =  'RBF'

    # kNN
    #pipeline["classifiers"]["kNN"] = {}
    #pipeline["classifiers"]["kNN"]["function"] =0 #some function pointer
    #pipeline["classifiers"]["kNN"]["some_parameter"] =0 #value of parameter

    # decision tree
    #pipeline["classifiers"]["decision_tree"] = {}
    #pipeline["classifiers"]["decision_tree"]["function"] =0 #some function pointer
    #pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter

    # random forest tree
    #pipeline["classifiers"]["RFTree"] = {}
    #pipeline["classifiers"]["RFTree"]["function"] =rftree
    #pipeline["classifiers"]["RFTree"]["ActiveVarCount"] =0 


    # ensemble learning
    #pipeline["classifiers"]["ensemble"] = {}
    #pipeline["classifiers"]["ensemble"]["function"] =["svm", "decision_tree"] #name of functions to be used for ensemblers
    #pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter


    #---------------------------------------------set up evaluation metrics and parameters------------------------


    pipeline["metrics"] = {}

    # accuracy
    pipeline["metrics"]["simple_accuracy"] = {}
    pipeline["metrics"]["simple_accuracy"]["function"] = accuracy
    pipeline["metrics"]["simple_accuracy"]["type"] = "simple"  

    pipeline["metrics"]["balanced_accuracy"] = {}
    pipeline["metrics"]["balanced_accuracy"]["function"] = accuracy
    pipeline["metrics"]["balanced_accuracy"]["type"] = "balanced"

    # precision
    pipeline["metrics"]["precision"] = {}
    pipeline["metrics"]["precision"]["function"] = precision
    pipeline["metrics"]["precision"]["class_result"] = "COVID"

    # recall
    pipeline["metrics"]["sensitivity"] = {}
    pipeline["metrics"]["sensitivity"]["function"] = sensitivity
    pipeline["metrics"]["sensitivity"]["class_result"]  = "COVID"

    # F1 score
    pipeline["metrics"]["f1_score"] = {}
    pipeline["metrics"]["f1_score"]["function"] = F1_score 
    pipeline["metrics"]["f1_score"]["class_result"] = "COVID"

    # mcc
    pipeline["metrics"]["mcc"] = {}
    pipeline["metrics"]["mcc"]["function"] = mcc 

    #------------------------------------------------Create Plots --------------------------
    pipeline["plots"] = {}
    pipeline["plots"]["CM"] = {}
    pipeline["plots"]["CM"]["function"] = plot_CM

    FE_batch_train_complete_dataset(pipeline)

    #-------prediction---------
    path_to_json = os.path.join(binary_path, "train", "training_pipeline.json")

    # test images
    print("Generating predictions on test data")
    test_images_path = os.path.join(images_folder, "test")
    test_save_path = os.path.join(binary_path, "test")

    FE_batch_prediction(
        path_to_images=test_images_path,
        path_to_json=path_to_json,
        save_path=test_save_path)


    # noisy test images
    print("Generating predictions on noisy test data")
    noisy_test_images_path = os.path.join(images_folder, "noisy_test")
    noisy_test_save_path = os.path.join(binary_path, "noisy_test")

    FE_batch_prediction(
        path_to_images=noisy_test_images_path,
        path_to_json=path_to_json,
        save_path=noisy_test_save_path)


    
    #------------------------------- Multiclass Classification --------------------------

    print("---------------------------- Multiclass Classification ------------------------")

    multi_path = os.path.join(run_path, "multiclass")
    if not os.path.exists(multi_path):
        os.mkdir(multi_path)
        print(f"Created directory {multi_path}")
    else:
        print(f"Warning: {multi_path} already exists. Contents may get overwritten")

    #----------------------Pipeline---------------------


    pipeline = {}
    pipeline["path_to_results"] = os.path.join(multi_path, "train")

    # whether to save features to pkl
    pipeline["save_to_pkl"] = True

    # number of images to process at a time
    pipeline["batch_size"] = 500

    #------------------ setup data------------------------

    pipeline["data"] = {}

    # path to folder containing images 
    pipeline["data"]["path_to_images"] = os.path.join(images_folder, "train")
    # path to labels.txt
    pipeline["data"]["path_to_labels"] = path_to_multi_labels


    # names of the class
    pipeline["data"]["classes"] = np.array(["Normal", "COVID", "pneumonia", "Lung_Opacity"]) 

    # ---------------------------------set up data preprocessing methods and parameters------------------------------------
    pipeline["data_preprocessing"] ={}

    # normalize image
    #pipeline["data_preprocessing"]["normalize_image"] = {}
    #pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
    #pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

    # map to RGB
    pipeline["data_preprocessing"]["map_to_RGB"] = {}
    pipeline["data_preprocessing"]["map_to_RGB"]["function"] = change_colorspace 
    pipeline["data_preprocessing"]["map_to_RGB"]["conversion"] ="BGR2GRAY" 

    # canny edge detector
    #pipeline["data_preprocessing"]["canny_edges"] = {}
    #pipeline["data_preprocessing"]["canny_edges"]["function"] = canny_edge_detector
    #pipeline["data_preprocessing"]["canny_edges"]["blur"] = True
    #pipeline["data_preprocessing"]["canny_edges"]["threshold1"] =250
    #pipeline["data_preprocessing"]["canny_edges"]["threshold2"] = 500
    #pipeline["data_preprocessing"]["canny_edges"]["apertureSize"] = 5
    #pipeline["data_preprocessing"]["canny_edges"]["L2gradient"] = True

    # resize_image
    pipeline["data_preprocessing"]["resize_image"] = {}
    pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
    pipeline["data_preprocessing"]["resize_image"]["output_size"] = (250, 250) #(width, height)
    pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"


    # ---------------------------------set up feature extractor methods and parameters------------------------------------
    pipeline["feature_extractors"] ={}

    # contrast
    #pipeline["feature_extractors"]["contrast"] = {}
    #pipeline["feature_extractors"]["contrast"]["function"] = calculate_contrast
    #pipeline["feature_extractors"]["contrast"]["method"] = "michelson" 

    # skewness
    #pipeline["feature_extractors"]["skewness"] = {}
    #pipeline["feature_extractors"]["skewness"]["function"] =calculate_skew
    #pipeline["feature_extractors"]["skewness"]["bias"] = True

    # kurtosis
    #pipeline["feature_extractors"]["kurtosis"] = {}
    #pipeline["feature_extractors"]["kurtosis"]["function"] = calculate_kurtosis
    #pipeline["feature_extractors"]["kurtosis"]["method"] = "pearson"
    #pipeline["feature_extractors"]["kurtosis"]["bias"] = True

    # RMS
    #pipeline["feature_extractors"]["RMS"] = {}
    #pipeline["feature_extractors"]["RMS"]["function"] = calculate_contrast
    #pipeline["feature_extractors"]["RMS"]["method"] = "rms"

    # count non zeros
    #pipeline["feature_extractors"]["count_nonzeros"] = {}
    #pipeline["feature_extractors"]["count_nonzeros"]["function"] = count_nonzeros

    # histogram
    #pipeline["feature_extractors"]["histogram"] = {}
    #pipeline["feature_extractors"]["histogram"]["function"] = calculate_histogram
    #pipeline["feature_extractors"]["histogram"]["bins"] = 256
    #pipeline["feature_extractors"]["histogram"]["range"] = (0,256)
    #pipeline["feature_extractors"]["histogram"]["density"] = False

    # haralick
    pipeline["feature_extractors"]["haralick"] = {}
    pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
    pipeline["feature_extractors"]["haralick"]["blur"] = True
    pipeline["feature_extractors"]["haralick"]["distance"] = 10

    # zernike
    pipeline["feature_extractors"]["zernike_moments"] = {}
    pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
    pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
    #pipeline["feature_extractors"]["zernike_moments"]["radius"] = 100
    #pipeline["feature_extractors"]["zernike_moments"]["degree"] = 10
    #pipeline["feature_extractors"]["zernike_moments"]["cm"] = (100, 100)

    #lbp
    #pipeline["feature_extractors"]["lbp"] = {}
    #pipeline["feature_extractors"]["lbp"]["function"] = calculate_lbp
    #pipeline["feature_extractors"]["lbp"]["P"] = 40
    #pipeline["feature_extractors"]["lbp"]["R"] = 12
    #pipeline["feature_extractors"]["lbp"]["method"] = "uniform"


    #-----------------Normalize feature vectors--------------------
    pipeline["normalize_features"] = {}
    pipeline["normalize_features"]["norm_type"] = "StandardScaler"


    # ---------------- Classifiers ---------------------------
    pipeline["classifiers"] ={}

    # SVM
    pipeline["classifiers"]["svm"] = {}
    pipeline["classifiers"]["svm"]["function"] = svm 
    pipeline["classifiers"]["svm"]["trainAuto"] = False
    pipeline["classifiers"]["svm"]['svm_type'] =  'C_SVC' 
    pipeline["classifiers"]["svm"]['kernel'] =  'RBF'


    # kNN
    #pipeline["classifiers"]["kNN"] = {}
    #pipeline["classifiers"]["kNN"]["function"] =0 #some function pointer
    #pipeline["classifiers"]["kNN"]["some_parameter"] =0 #value of parameter

    # decision tree
    #pipeline["classifiers"]["decision_tree"] = {}
    #pipeline["classifiers"]["decision_tree"]["function"] =0 #some function pointer
    #pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter

    # random forest tree
    #pipeline["classifiers"]["RFTree"] = {}
    #pipeline["classifiers"]["RFTree"]["function"] =rftree
    #pipeline["classifiers"]["RFTree"]["ActiveVarCount"] =0 


    # ensemble learning
    #pipeline["classifiers"]["ensemble"] = {}
    #pipeline["classifiers"]["ensemble"]["function"] =["svm", "decision_tree"] #name of functions to be used for ensemblers
    #pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter


    #--------------------Evaluation Metrics---------------------------


    pipeline["metrics"] = {}

    # accuracy
    pipeline["metrics"]["simple_accuracy"] = {}
    pipeline["metrics"]["simple_accuracy"]["function"] = accuracy
    pipeline["metrics"]["simple_accuracy"]["type"] = "simple"  

    pipeline["metrics"]["balanced_accuracy"] = {}
    pipeline["metrics"]["balanced_accuracy"]["function"] = accuracy
    pipeline["metrics"]["balanced_accuracy"]["type"] = "balanced"

    # precision
    pipeline["metrics"]["precision"] = {}
    pipeline["metrics"]["precision"]["function"] = precision
    pipeline["metrics"]["precision"]["average"] = "weighted"

    # recall
    pipeline["metrics"]["sensitivity"] = {}
    pipeline["metrics"]["sensitivity"]["function"] = sensitivity
    pipeline["metrics"]["sensitivity"]["average"]  = "weighted"

    # F1 score
    pipeline["metrics"]["f1_score"] = {}
    pipeline["metrics"]["f1_score"]["function"] = F1_score 
    pipeline["metrics"]["f1_score"]["average"] = "weighted"

    # mcc
    pipeline["metrics"]["mcc"] = {}
    pipeline["metrics"]["mcc"]["function"] = mcc 

    #---------------------Create Plots --------------------------
    pipeline["plots"] = {}
    pipeline["plots"]["CM"] = {}
    pipeline["plots"]["CM"]["function"] = plot_CM

    #------- train ------------
    FE_batch_train_complete_dataset(pipeline)

    #-------prediction---------
    path_to_json = os.path.join(multi_path, "train", "training_pipeline.json")

    # test images
    print("Generating predicitons on test data.")
    test_images_path = os.path.join(images_folder, "test")
    test_save_path = os.path.join(multi_path, "test")

    FE_batch_prediction(
        path_to_images=test_images_path,
        path_to_json=path_to_json,
        save_path=test_save_path)

    # noisy test images
    print("Generating predicitons on noisy test data.")
    noisy_test_images_path = os.path.join(images_folder, "noisy_test")
    noisy_test_save_path = os.path.join(multi_path, "noisy_test")

    FE_batch_prediction(
        path_to_images=noisy_test_images_path,
        path_to_json=path_to_json,
        save_path=noisy_test_save_path)


    print("Processing Completed")