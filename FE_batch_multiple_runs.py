
"""
Performs binary and multiclass classificaiton at the same time

"""
from FE_batch_training import FE_batch_training
from FE_batch_prediction import FE_batch_prediction

# import libraries
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
from utils.feature_extractors.wavelet import feature_GLCM

from utils.normalize_features import normalize_features

from utils.apply_classifiers import apply_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree
from utils.classifiers.boosting import boosting

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
#--------------------------------- raw data image dir------------------------------

images_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data"

#-------------------------------- Results directory----------------------------
results_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results"

run_name = "zernike"

run_path = os.path.join(results_folder, run_name)
if not os.path.exists(run_path):
    os.mkdir(run_path)
    print(f"Created directory {run_path}")
else:
    print(f"Warning: {run_path} already exists. Contents may get overwritten")

#-------------------------------------------- Binary Classification ----------------------------


print("---------------------------- Binary Classification ------------------------")

# transform label txt file
print("Generating txt file for binary classification")
path_to_multi_labels = os.path.join(images_folder, "train_multi.txt")
path_to_binary_labels = os.path.join(images_folder, "train_binary.txt")
change_txt_for_binary(path_to_multi_labels, path_to_binary_labels)

# result dir
binary_path = os.path.join(run_path, "binary")
if not os.path.exists(binary_path):
    os.mkdir(binary_path)
    print(f"Created directory {binary_path}")
else:
    print(f"Warning: {binary_path} already exists. Contents may get overwritten")


#----------------------Pipeline---------------------


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

# split data
pipeline["data"]["split_type"] = "simpleStratified" #"simple", "simpleStratified", "kfold", "kfoldStratified"


# names of the class
pipeline["data"]["classes"] =  np.array(["NO_COVID", "COVID"])

# ---------------------------------set up data preprocessing methods and parameters------------------------------------
pipeline["data_preprocessing"] ={}

# normalize image
#pipeline["data_preprocessing"]["normalize_image"] = {}
#pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
#pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

# map to RGB
pipeline["data_preprocessing"]["map_to_grayscale"] = {}
pipeline["data_preprocessing"]["map_to_grayscale"]["function"] = change_colorspace 
pipeline["data_preprocessing"]["map_to_grayscale"]["conversion"] ="BGR2GRAY" 

# canny edge detector
#pipeline["data_preprocessing"]["canny_edges"] = {}
#pipeline["data_preprocessing"]["canny_edges"]["function"] = canny_edge_detector
#pipeline["data_preprocessing"]["canny_edges"]["blur"] = True
#pipeline["data_preprocessing"]["canny_edges"]["threshold1"] =100
#pipeline["data_preprocessing"]["canny_edges"]["threshold2"] = 250
#pipeline["data_preprocessing"]["canny_edges"]["apertureSize"] = 5
#pipeline["data_preprocessing"]["canny_edges"]["L2gradient"] = True

# resize_image
#pipeline["data_preprocessing"]["resize_image"] = {}
#pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
#pipeline["data_preprocessing"]["resize_image"]["output_size"] = (250, 250) #(width, height)
#pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"


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
#pipeline["feature_extractors"]["histogram"]["bins"] = 10
#pipeline["feature_extractors"]["histogram"]["range"] = (0,256)
#pipeline["feature_extractors"]["histogram"]["density"] = False

# haralick
#pipeline["feature_extractors"]["haralick"] = {}
#pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
#pipeline["feature_extractors"]["haralick"]["blur"] = True
#pipeline["feature_extractors"]["haralick"]["distance"] = 1

# zernike
pipeline["feature_extractors"]["zernike_moments"] = {}
pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
pipeline["feature_extractors"]["zernike_moments"]["radius"] = 180
#pipeline["feature_extractors"]["zernike_moments"]["degree"] = 8
#pipeline["feature_extractors"]["zernike_moments"]["cm"] = (100, 100)

#lbp
#pipeline["feature_extractors"]["lbp"] = {}
#pipeline["feature_extractors"]["lbp"]["function"] = calculate_lbp
#pipeline["feature_extractors"]["lbp"]["P"] = 40
#pipeline["feature_extractors"]["lbp"]["R"] = 12
#pipeline["feature_extractors"]["lbp"]["method"] = "uniform"

# feature GLCM
#pipeline["feature_extractors"]["GLCM"] = {}
#pipeline["feature_extractors"]["GLCM"]["function"] = feature_GLCM
#pipeline["feature_extractors"]["GLCM"]["wavelet_type"] = 'bior1.3'


#-----------------Normalize feature vectors--------------------
pipeline["normalize_features"] = {}
pipeline["normalize_features"]["norm_type"] = "StandardScaler"


# ---------------- Classifiers ---------------------------
pipeline["classifiers"] ={}

# SVM
pipeline["classifiers"]["svm"] = {}
pipeline["classifiers"]["svm"]["function"] = svm 
pipeline["classifiers"]["svm"]["trainAuto"] = True
pipeline["classifiers"]["svm"]['svm_type'] =  'C_SVC' 
pipeline["classifiers"]["svm"]['kernel'] =  'RBF'
#pipeline["classifiers"]["svm"]['kernel'] =  'POLY'
#pipeline["classifiers"]["svm"]['Degree'] = 3

# kNN
#pipeline["classifiers"]["kNN"] = {}
#pipeline["classifiers"]["kNN"]["function"] =0 #some function pointer
#pipeline["classifiers"]["kNN"]["some_parameter"] =0 #value of parameter

# decision tree
#pipeline["classifiers"]["decision_tree"] = {}
#pipeline["classifiers"]["decision_tree"]["function"] =0 #some function pointer
#pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter

# random forest tree
pipeline["classifiers"]["RFTree"] = {}
pipeline["classifiers"]["RFTree"]["function"] =rftree
pipeline["classifiers"]["RFTree"]["ActiveVarCount"] =0 
pipeline["classifiers"]["RFTree"]['MaxDepth'] = 20 
pipeline["classifiers"]["RFTree"]['num_iters'] =  10000

# random forest tree
pipeline["classifiers"]["Boosting"] = {}
pipeline["classifiers"]["Boosting"]["function"] = boosting
pipeline["classifiers"]["Boosting"]["boost_type"] = "REAL"
pipeline["classifiers"]["Boosting"]['num_weak_classifiers'] =  100
pipeline["classifiers"]["Boosting"]['max_depth'] = 20 


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

#---------------------Create Plots --------------------------
pipeline["plots"] = {}
pipeline["plots"]["CM"] = {}
pipeline["plots"]["CM"]["function"] = plot_CM

#------- train ------------
FE_batch_training (pipeline)

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
#print("Generating predictions on noisy test data")
#noisy_test_images_path = os.path.join(images_folder, "noisy_test")
#noisy_test_save_path = os.path.join(binary_path, "noisy_test")

#FE_batch_prediction(
#     path_to_images=noisy_test_images_path,
#     path_to_json=path_to_json,
#     save_path=noisy_test_save_path)


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

# split data
pipeline["data"]["split_type"] = "simpleStratified" #"simple", "simpleStratified", "kfold", "kfoldStratified"


# names of the class
pipeline["data"]["classes"] = np.array(["Normal", "COVID", "pneumonia", "Lung_Opacity"]) 

# ---------------------------------set up data preprocessing methods and parameters------------------------------------
pipeline["data_preprocessing"] ={}

# normalize image
#pipeline["data_preprocessing"]["normalize_image"] = {}
#pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
#pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax" 

# map to RGB
pipeline["data_preprocessing"]["map_to_grayscale"] = {}
pipeline["data_preprocessing"]["map_to_grayscale"]["function"] = change_colorspace 
pipeline["data_preprocessing"]["map_to_grayscale"]["conversion"] ="BGR2GRAY" 

# canny edge detector
#pipeline["data_preprocessing"]["canny_edges"] = {}
#pipeline["data_preprocessing"]["canny_edges"]["function"] = canny_edge_detector
#pipeline["data_preprocessing"]["canny_edges"]["blur"] = True
#pipeline["data_preprocessing"]["canny_edges"]["threshold1"] =100
#pipeline["data_preprocessing"]["canny_edges"]["threshold2"] = 250
#pipeline["data_preprocessing"]["canny_edges"]["apertureSize"] = 5
#pipeline["data_preprocessing"]["canny_edges"]["L2gradient"] = True

# resize_image
#pipeline["data_preprocessing"]["resize_image"] = {}
#pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
#pipeline["data_preprocessing"]["resize_image"]["output_size"] = (250, 250) #(width, height)
#pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"


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
# pipeline["feature_extractors"]["RMS"] = {}
#pipeline["feature_extractors"]["RMS"]["function"] = calculate_contrast
#pipeline["feature_extractors"]["RMS"]["method"] = "rms"

# count non zeros
#pipeline["feature_extractors"]["count_nonzeros"] = {}
#pipeline["feature_extractors"]["count_nonzeros"]["function"] = count_nonzeros

# histogram
#pipeline["feature_extractors"]["histogram"] = {}
#pipeline["feature_extractors"]["histogram"]["function"] = calculate_histogram
#pipeline["feature_extractors"]["histogram"]["bins"] = 10
#pipeline["feature_extractors"]["histogram"]["range"] = (0,256)
#pipeline["feature_extractors"]["histogram"]["density"] = False

# haralick
#pipeline["feature_extractors"]["haralick"] = {}
#pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
#pipeline["feature_extractors"]["haralick"]["blur"] = True
# pipeline["feature_extractors"]["haralick"]["distance"] = 1

# zernike
pipeline["feature_extractors"]["zernike_moments"] = {}
pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
pipeline["feature_extractors"]["zernike_moments"]["radius"] = 180
#pipeline["feature_extractors"]["zernike_moments"]["degree"] = 8
#pipeline["feature_extractors"]["zernike_moments"]["cm"] = (100, 100)

#lbp
#pipeline["feature_extractors"]["lbp"] = {}
#pipeline["feature_extractors"]["lbp"]["function"] = calculate_lbp
#pipeline["feature_extractors"]["lbp"]["P"] = 40
#pipeline["feature_extractors"]["lbp"]["R"] = 12
# pipeline["feature_extractors"]["lbp"]["method"] = "uniform"

# feature GLCM
#pipeline["feature_extractors"]["GLCM"] = {}
#pipeline["feature_extractors"]["GLCM"]["function"] = feature_GLCM
#pipeline["feature_extractors"]["GLCM"]["wavelet_type"] = 'bior1.3'


#-----------------Normalize feature vectors--------------------
pipeline["normalize_features"] = {}
pipeline["normalize_features"]["norm_type"] = "StandardScaler"


# ---------------- Classifiers ---------------------------
pipeline["classifiers"] ={}

# SVM
pipeline["classifiers"]["svm"] = {}
pipeline["classifiers"]["svm"]["function"] = svm 
pipeline["classifiers"]["svm"]["trainAuto"] = True
pipeline["classifiers"]["svm"]['svm_type'] =  'C_SVC' 
pipeline["classifiers"]["svm"]['kernel'] =  'RBF'
#pipeline["classifiers"]["svm"]['kernel'] =  'POLY'
#pipeline["classifiers"]["svm"]['Degree'] = 3
#pipeline["classifiers"]["svm"]['num_iters'] =  10000


# kNN
#pipeline["classifiers"]["kNN"] = {}
#pipeline["classifiers"]["kNN"]["function"] =0 #some function pointer
#pipeline["classifiers"]["kNN"]["some_parameter"] =0 #value of parameter

# decision tree
#pipeline["classifiers"]["decision_tree"] = {}
#pipeline["classifiers"]["decision_tree"]["function"] =0 #some function pointer
#pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter

# random forest tree
pipeline["classifiers"]["RFTree"] = {}
pipeline["classifiers"]["RFTree"]["function"] =rftree
pipeline["classifiers"]["RFTree"]["ActiveVarCount"] =0 
pipeline["classifiers"]["RFTree"]['MaxDepth'] = 20 
pipeline["classifiers"]["RFTree"]['num_iters'] =  10000

# boosting
pipeline["classifiers"]["Boosting"] = {}
pipeline["classifiers"]["Boosting"]["function"] = boosting
pipeline["classifiers"]["Boosting"]["boost_type"] = "REAL"
pipeline["classifiers"]["Boosting"]['num_weak_classifiers'] =  200
pipeline["classifiers"]["Boosting"]['max_depth'] = 10 

# ensemble learning
#pipeline["classifiers"]["ensemble"] = {}
#pipeline["classifiers"]["ensemble"]["function"] =["svm", "decision_tree"] #name of functions to be used for ensemblers
#pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter


#--------------------Evaluation Metrics---------------------------


pipeline["metrics"] ={}

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
FE_batch_training (pipeline)

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
#print("Generating predicitons on noisy test data.")
#noisy_test_images_path = os.path.join(images_folder, "noisy_test")
#noisy_test_save_path = os.path.join(multi_path, "noisy_test")

#FE_batch_prediction(
#     path_to_images=noisy_test_images_path,
#     path_to_json=path_to_json,
#     save_path=noisy_test_save_path)


print("Processing Completed")