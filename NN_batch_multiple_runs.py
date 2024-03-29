"""
Trains different neural networks using the features obtained from different feature extractors.
Applies data preprocessing and feature extractors on batch of images.
Also generates prediction on test and noisy dataset

Performs both binary and multiclass classification

"""
from NN_batch_training import NN_batch_training
from NN_batch_prediction import NN_batch_prediction

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

from utils.data_preprocessing.center_crop import center_crop
from utils.data_preprocessing.random_apply_rotation import random_apply_rotation
from utils.data_preprocessing.random_vertical_flip import random_vertical_flip
from utils.data_preprocessing.random_resized_crop import random_resized_crop
from utils.data_preprocessing.random_horizontal_flip import random_horizontal_flip
from utils.data_preprocessing.random_apply_affine import random_apply_affine
from utils.data_preprocessing.random_auto_contrast import random_auto_contrast
from utils.data_preprocessing.random_adjust_sharpness import random_adjust_sharpness


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

from utils.evaluate_metrics import evaluate_metrics
from utils.metrics.accuracy import accuracy
from utils.metrics.f1_score import F1_score
from utils.metrics.mcc import mcc
from utils.metrics.precision import precision
from utils.metrics.sensitivity import sensitivity

from utils.create_plots import create_plots
from utils.plots.plot_CM import plot_CM
from utils.plots.plot_learning_curves import plot_LC
from utils.plots.plot_misclassified_samples import plot_MS

from utils.json_processing import save_to_json

from utils.misc import add_function_names, generate_txt_file, save_results, change_txt_for_binary
#--------------------------------- raw data image dir------------------------------

images_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data"

#-------------------------------- Results directory----------------------------
results_folder = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results"

run_name = "run_03_weighted_loss_no_reg_data_aug_online"

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
#change_txt_for_binary(path_to_multi_labels, path_to_binary_labels)

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
pipeline["data"]["test_size"] = 0.3

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

# random adjust sharpness
pipeline["data_preprocessing"]["random_adjust_sharpness"] = {}
pipeline["data_preprocessing"]["random_adjust_sharpness"]["function"] = random_adjust_sharpness
pipeline["data_preprocessing"]["random_adjust_sharpness"]["sharpness_factor"] = 2

#random auto contrast
pipeline["data_preprocessing"]["random_auto_contrast"] = {}
pipeline["data_preprocessing"]["random_auto_contrast"]["function"] = random_auto_contrast

# random appyly rotation
pipeline["data_preprocessing"]["random_apply_rotation"] = {}
pipeline["data_preprocessing"]["random_apply_rotation"]["function"] = random_apply_rotation 
pipeline["data_preprocessing"]["random_apply_rotation"]["degrees"] =90 
pipeline["data_preprocessing"]["random_apply_rotation"]["expand"] = True

# random affine
pipeline["data_preprocessing"]["random_apply_affine"] = {}
pipeline["data_preprocessing"]["random_apply_affine"]["function"] =random_apply_affine
pipeline["data_preprocessing"]["random_apply_affine"]["degrees"] = (-15, 15)
pipeline["data_preprocessing"]["random_apply_affine"]["scale"] = (0.1, 0.3)
pipeline["data_preprocessing"]["random_apply_affine"]["translate"] = (0.1, 0.3)

# random resized crop
pipeline["data_preprocessing"]["random_resized_crop"] = {}
pipeline["data_preprocessing"]["random_resized_crop"]["function"] = random_resized_crop
pipeline["data_preprocessing"]["random_resized_crop"]["output_size"] = (250, 250) #(width, height)

# horziontal flip
pipeline["data_preprocessing"]["horizontal_flip"] = {}
pipeline["data_preprocessing"]["horizontal_flip"]["function"] = random_horizontal_flip

# vertical flip
pipeline["data_preprocessing"]["vertical_flip"] = {}
pipeline["data_preprocessing"]["vertical_flip"]["function"] = random_vertical_flip

# normalize image
pipeline["data_preprocessing"]["normalize_image"] = {}
pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax_255" 

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
pipeline["feature_extractors"]["haralick"] = {}
pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
pipeline["feature_extractors"]["haralick"]["blur"] = True
pipeline["feature_extractors"]["haralick"]["distance"] = 1

# zernike
pipeline["feature_extractors"]["zernike_moments"] = {}
pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
pipeline["feature_extractors"]["zernike_moments"]["radius"] = 180
pipeline["feature_extractors"]["zernike_moments"]["degree"] = 8
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
pipeline["networks"] ={}

# NN1
pipeline["networks"]["NN1"] = {}
pipeline["networks"]["NN1"]["hidden_layers"] = [32]
#pipeline["networks"]["NN1"]["alpha"] = 0.2
pipeline["networks"]["NN1"]["batch_size"] = 256
pipeline["networks"]["NN1"]["lmbda"] = 0
pipeline["networks"]["NN1"]["lr"] = 0.001
pipeline["networks"]["NN1"]["epochs"] = 10
pipeline["networks"]["NN1"]["use_single_neuron"] = True
pipeline["networks"]["NN1"]["use_weighted_loss"] = True


# NN2
pipeline["networks"]["NN2"] = {}
pipeline["networks"]["NN2"]["hidden_layers"] = [32]
#pipeline["networks"]["NN2"]["alpha"] = 0.2
pipeline["networks"]["NN2"]["batch_size"] = 256
pipeline["networks"]["NN2"]["lmbda"] = 0
pipeline["networks"]["NN2"]["lr"] = 0.001
pipeline["networks"]["NN2"]["epochs"] = 10
pipeline["networks"]["NN2"]["use_weighted_loss"] = True

# NN3
pipeline["networks"]["NN3"] = {}
pipeline["networks"]["NN3"]["hidden_layers"] = [32, 16]
#pipeline["networks"]["NN3"]["alpha"] = 0.2
pipeline["networks"]["NN3"]["batch_size"] = 256
pipeline["networks"]["NN3"]["lmbda"] = 0
pipeline["networks"]["NN3"]["lr"] = 0.001
pipeline["networks"]["NN3"]["epochs"] = 10
pipeline["networks"]["NN3"]["use_single_neuron"] = True
pipeline["networks"]["NN3"]["use_weighted_loss"] = True

# NN4
pipeline["networks"]["NN4"] = {}
pipeline["networks"]["NN4"]["hidden_layers"] = [32, 16]
#pipeline["networks"]["NN4"]["alpha"] = 0.2
pipeline["networks"]["NN4"]["batch_size"] = 256
pipeline["networks"]["NN4"]["lmbda"] = 0
pipeline["networks"]["NN4"]["lr"] = 0.001
pipeline["networks"]["NN4"]["epochs"] = 10
pipeline["networks"]["NN4"]["use_weighted_loss"] = True

# NN5
pipeline["networks"]["NN5"] = {}
pipeline["networks"]["NN5"]["hidden_layers"] = [64, 32, 16]
#pipeline["networks"]["NN5"]["alpha"] = 0.2
pipeline["networks"]["NN5"]["batch_size"] = 256
pipeline["networks"]["NN5"]["lmbda"] = 0
pipeline["networks"]["NN5"]["lr"] =0.001
pipeline["networks"]["NN5"]["epochs"] = 10
pipeline["networks"]["NN5"]["use_single_neuron"] = True
pipeline["networks"]["NN5"]["use_weighted_loss"] = True

# NN6
pipeline["networks"]["NN6"] = {}
pipeline["networks"]["NN6"]["hidden_layers"] = [64, 32, 16]
#pipeline["networks"]["NN6"]["alpha"] = 0.2
pipeline["networks"]["NN6"]["batch_size"] = 256
pipeline["networks"]["NN6"]["lmbda"] = 0
pipeline["networks"]["NN6"]["lr"] = 0.001
pipeline["networks"]["NN6"]["epochs"] = 10
pipeline["networks"]["NN6"]["use_weighted_loss"] = True


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

# confusion matrix
pipeline["plots"]["CM"] = {}
pipeline["plots"]["CM"]["function"] = plot_CM

# learning curves
pipeline["plots"]["learning_curves"] = {}
pipeline["plots"]["learning_curves"]["function"] = plot_LC

# plot misidentified samples
pipeline["plots"]["misidentified samples"] = {}
pipeline["plots"]["misidentified samples"]["function"] = plot_MS

#------- train ------------
NN_batch_training (pipeline)

#-------prediction---------
path_to_json = os.path.join(binary_path, "train", "training_pipeline.json")

# test images
print("Generating predictions on test data")
test_images_path = os.path.join(images_folder, "test")
test_save_path = os.path.join(binary_path, "test")

NN_batch_prediction(
     path_to_images=test_images_path,
     path_to_json=path_to_json,
     save_path=test_save_path)


# noisy test images
print("Generating predictions on noisy test data")
noisy_test_images_path = os.path.join(images_folder, "noisy_test")
noisy_test_save_path = os.path.join(binary_path, "noisy_test")

NN_batch_prediction(
     path_to_images=noisy_test_images_path,
     path_to_json=path_to_json,
     save_path=noisy_test_save_path,
     batch_size=1)


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
pipeline["data"]["test_size"] = 0.3

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

# random adjust sharpness
pipeline["data_preprocessing"]["random_adjust_sharpness"] = {}
pipeline["data_preprocessing"]["random_adjust_sharpness"]["function"] = random_adjust_sharpness
pipeline["data_preprocessing"]["random_adjust_sharpness"]["sharpness_factor"] = 2

#random auto contrast
pipeline["data_preprocessing"]["random_auto_contrast"] = {}
pipeline["data_preprocessing"]["random_auto_contrast"]["function"] = random_auto_contrast

# random appyly rotation
pipeline["data_preprocessing"]["random_apply_rotation"] = {}
pipeline["data_preprocessing"]["random_apply_rotation"]["function"] = random_apply_rotation 
pipeline["data_preprocessing"]["random_apply_rotation"]["degrees"] =90 
pipeline["data_preprocessing"]["random_apply_rotation"]["expand"] = True

# random affine
pipeline["data_preprocessing"]["random_apply_affine"] = {}
pipeline["data_preprocessing"]["random_apply_affine"]["function"] =random_apply_affine
pipeline["data_preprocessing"]["random_apply_affine"]["degrees"] = (-15, 15)
pipeline["data_preprocessing"]["random_apply_affine"]["scale"] = (0.1, 0.3)
pipeline["data_preprocessing"]["random_apply_affine"]["translate"] = (0.1, 0.3)

# random resized crop
pipeline["data_preprocessing"]["random_resized_crop"] = {}
pipeline["data_preprocessing"]["random_resized_crop"]["function"] = random_resized_crop
pipeline["data_preprocessing"]["random_resized_crop"]["output_size"] = (250, 250) #(width, height)

# horziontal flip
pipeline["data_preprocessing"]["horizontal_flip"] = {}
pipeline["data_preprocessing"]["horizontal_flip"]["function"] = random_horizontal_flip

# vertical flip
pipeline["data_preprocessing"]["vertical_flip"] = {}
pipeline["data_preprocessing"]["vertical_flip"]["function"] = random_vertical_flip


# resize_image
#pipeline["data_preprocessing"]["resize_image"] = {}
#pipeline["data_preprocessing"]["resize_image"]["function"] = resize 
#pipeline["data_preprocessing"]["resize_image"]["output_size"] = (250, 250) #(width, height)
#pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"

# normalize image
pipeline["data_preprocessing"]["normalize_image"] = {}
pipeline["data_preprocessing"]["normalize_image"]["function"] = normalize
pipeline["data_preprocessing"]["normalize_image"]["method"] = "minmax_255" 


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
pipeline["feature_extractors"]["haralick"] = {}
pipeline["feature_extractors"]["haralick"]["function"] = calculate_haralick
pipeline["feature_extractors"]["haralick"]["blur"] = True
pipeline["feature_extractors"]["haralick"]["distance"] = 1

# zernike
pipeline["feature_extractors"]["zernike_moments"] = {}
pipeline["feature_extractors"]["zernike_moments"]["function"] = calculate_zernike
pipeline["feature_extractors"]["zernike_moments"]["blur"] = True
pipeline["feature_extractors"]["zernike_moments"]["radius"] = 140
pipeline["feature_extractors"]["zernike_moments"]["degree"] = 8
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
pipeline["networks"] ={}

# NN1
pipeline["networks"]["NN1"] = {}
pipeline["networks"]["NN1"]["hidden_layers"] = [32]
#pipeline["networks"]["NN1"]["alpha"] = 0.2
pipeline["networks"]["NN1"]["batch_size"] = 256
pipeline["networks"]["NN1"]["lmbda"] = 0
pipeline["networks"]["NN1"]["lr"] = 0.001
pipeline["networks"]["NN1"]["epochs"] = 10
pipeline["networks"]["NN1"]["use_weighted_loss"] = True


# NN2
pipeline["networks"]["NN2"] = {}
pipeline["networks"]["NN2"]["hidden_layers"] = [32, 16]
#pipeline["networks"]["NN2"]["alpha"] = 0.2
pipeline["networks"]["NN2"]["batch_size"] = 256
pipeline["networks"]["NN2"]["lmbda"] = 0
pipeline["networks"]["NN2"]["lr"] = 0.001
pipeline["networks"]["NN2"]["epochs"] = 10
pipeline["networks"]["NN2"]["use_weighted_loss"] = True

# NN3
pipeline["networks"]["NN3"] = {}
pipeline["networks"]["NN3"]["hidden_layers"] = [64,32,16]
#pipeline["networks"]["NN3"]["alpha"] = 0.2
pipeline["networks"]["NN3"]["batch_size"] = 256
pipeline["networks"]["NN3"]["lmbda"] = 0
pipeline["networks"]["NN3"]["lr"] = 0.001
pipeline["networks"]["NN3"]["epochs"] = 10
pipeline["networks"]["NN3"]["use_weighted_loss"] = True


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

# confusion matrix
pipeline["plots"]["CM"] = {}
pipeline["plots"]["CM"]["function"] = plot_CM

# learning curves
pipeline["plots"]["learning_curves"] = {}
pipeline["plots"]["learning_curves"]["function"] = plot_LC

# plot misidentified samples
pipeline["plots"]["misidentified samples"] = {}
pipeline["plots"]["misidentified samples"]["function"] = plot_MS

#------- train ------------
NN_batch_training (pipeline)

#-------prediction---------
path_to_json = os.path.join(multi_path, "train", "training_pipeline.json")

# test images
print("Generating predicitons on test data.")
test_images_path = os.path.join(images_folder, "test")
test_save_path = os.path.join(multi_path, "test")

NN_batch_prediction(
     path_to_images=test_images_path,
     path_to_json=path_to_json,
     save_path=test_save_path)

# noisy test images
print("Generating predicitons on noisy test data.")
noisy_test_images_path = os.path.join(images_folder, "noisy_test")
noisy_test_save_path = os.path.join(multi_path, "noisy_test")

NN_batch_prediction(
    path_to_images=noisy_test_images_path,
     path_to_json=path_to_json,
     save_path=noisy_test_save_path,
     batch_size=1)


print("Processing Completed")
