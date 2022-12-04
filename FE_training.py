"""
Performs Binary classification (COVID vs No-COVID).

Classicial methods are used for generating feature vectors and different classifiers are used to perform classifcation

"""

# import libraries
import numpy as np

from utils.load_and_preprocess_data import load_and_preprocess_data
from utils.data_preprocessing.change_colorspace import change_colorspace
from utils.data_preprocessing.resize import resize
from utils.data_preprocessing.normalize import normalize

from utils.generate_feature_vector import generate_feature_vector
from utils.feature_extractors.contrast import calculate_contrast
from utils.feature_extractors.kurtosis import calculate_kurtosis
from utils.feature_extractors.skewness import calculate_skew

from utils.normalize_features import normalize_features


verbose = True

# where to store the results
path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"


pipeline = {}

#------------------ setup data------------------------
pipeline["data"] = {}

# can be to folder 
pipeline["data"]["path_to_images"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
# can be to .txt
pipeline["data"]["path_to_labels"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train.txt"

# split data
pipeline["data"]["split_data"] = "simple" , #"simpleStrafied", "kfold", "kfoldStratified"


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
pipeline["data_preprocessing"]["resize_image"]["output_size"] = (150,150) #(width, height)
pipeline["data_preprocessing"]["resize_image"]["interpolation"] = "area"

# ---------------------------------set up feature extractor methods and parameters------------------------------------
pipeline["feature_extractors"] ={}

# contrast
pipeline["feature_extractors"]["contrast"] = {}
pipeline["feature_extractors"]["contrast"]["function"] = calculate_contrast
pipeline["feature_extractors"]["contrast"]["method"] = "michelson" 

# skewness
pipeline["feature_extractors"]["skewness"] = {}
pipeline["feature_extractors"]["skewness"]["function"] =calculate_skew
pipeline["feature_extractors"]["skewness"]["bias"] = True

# kurtosis
pipeline["feature_extractors"]["kurtosis"] = {}
pipeline["feature_extractors"]["kurtosis"]["function"] = calculate_kurtosis
pipeline["feature_extractors"]["kurtosis"]["method"] = "pearson"
pipeline["feature_extractors"]["kurtosis"]["bias"] = True

# RMS
pipeline["feature_extractors"]["RMS"] = {}
pipeline["feature_extractors"]["RMS"]["function"] = calculate_contrast
pipeline["feature_extractors"]["RMS"]["method"] = "rms"

#---------------------------------Normalize feature vectors-----------------------
pipeline["normalize_features"] = {}
pipeline["normalize_features"]["norm_type"] = "StandardScaler"

# ---------------------------------set up classifiers and parameters------------------------------------
pipeline["classifiers"] ={}

# SVM
pipeline["classifiers"]["svm"] = {}
pipeline["classifiers"]["svm"]["function"] =0 #some function pointer
pipeline["classifiers"]["svm"]["some_parameter"] =0 #value of parameter


# kNN
pipeline["classifiers"]["kNN"] = {}
pipeline["classifiers"]["kNN"]["function"] =0 #some function pointer
pipeline["classifiers"]["kNN"]["some_parameter"] =0 #value of parameter


# decision tree
pipeline["classifiers"]["decision_tree"] = {}
pipeline["classifiers"]["decision_tree"]["function"] =0 #some function pointer
pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter

# ensemble learning
pipeline["classifiers"]["ensemble"] = {}
pipeline["classifiers"]["ensemble"]["function"] =["svm", "decision_tree"] #name of functions to be used for ensemblers
pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter


#---------------------------------------------set up evaluation metrics and parameters------------------------

# accuracy
pipeline["metrics"] = {}

pipeline["metrics"]["accuracy"] = {}
pipeline["metrics"]["accuracy"]["function"] =0 #name of functions to be used for ensemblers
pipeline["metrics"]["accuracy"]["some_parameter"] =0 #value of parameter

# precision
pipeline["metrics"]["precision"] = {}
pipeline["metrics"]["precision"]["function"] =0 #name of functions to be used for ensemblers
pipeline["metrics"]["precision"]["some_parameter"] =0 #value of parameter


# recall
pipeline["metrics"]["sensitivity"] = {}
pipeline["metrics"]["sensitivity"]["function"] =0 #name of functions to be used for ensemblers
pipeline["metrics"]["sensitivity"]["some_parameter"] =0 #value of parameter

# Specificity
pipeline["metrics"]["specificity"] = {}
pipeline["metrics"]["specificity"]["function"] =0 #name of functions to be used for ensemblers
pipeline["metrics"]["specificity"]["some_parameter"] =0 #value of parameter


# f1_score
pipeline["metrics"]["f1_score"] = {}
pipeline["metrics"]["f1_score"]["function"] =0 #name of functions to be used for ensemblers
pipeline["metrics"]["f1_score"]["some_parameter"] =0 #value of parameter




#--------------------------------------------Build classifier and get results----------------------------

# Image is read in BGR format initially. 

# load data

print("\nLoading Training data . . . ")

X_train, y_train, X_valid, y_valid, data_config = load_and_preprocess_data(data_config={**pipeline["data"], **pipeline["data_preprocessing"]})

print("Loaded Training data Sucessfully")
print("the shape of X_train: ", X_train.shape)
print("the shape of y_train: ", y_train.shape)
print("the shape of X_valid: ", X_valid.shape)
print("the shape of y_valid: ", y_valid.shape)
               
# print unique labels and their counts
"""
num_images_train =y_train.shape[1]
num_images_valid =y_valid.shape[1]
total_num_samples = num_images_train + num_images_valid

unique_labels, counts = np.unique(y_train[:,1], return_counts=True)
print(f"Unique labels and their frequencies found in the training data: ")
for ul, c in zip(unique_labels, counts):
    print(f"Label: {ul} Count: {c}")
"""
# feature generation

print("\nGenerating Features for training data . . . ")
features_train, features_config = generate_feature_vector(X=X_train, extractors=pipeline["feature_extractors"])
print(f"Generated Features for training data successfully. Shape of feature vector: {features_train.shape}")

print("\nGenerating Features for validation data . . . ")
features_valid, _ =generate_feature_vector(X=X_valid, extractors=features_config)
print(f"Generated Features for validation data successfully. Shape of feature vector: {features_valid.shape}")


# normalize vectors
print("\nNormalizing Features for training data . . . ")
features_norm_train, features_norm_config = normalize_features(features=features_train, parameters=pipeline["normalize_features"])
print(f"Normalized Features for training data successfully. Shape of normalized feature vector: {features_norm_train.shape}")

print("\nNormalizing Features for validation data . . . ")
features_norm_valid, _ = normalize_features(features=features_valid, parameters=features_norm_config)
print(f"Normalized Features for validation data successfully. Shape of normalized feature vector: {features_norm_valid.shape}")


# get prediction

print("\nGenerating labels for training data . . . ")
y_pred_train, classifiers_config = apply_classifiers(X=features_norm_train, classifiers=pipeline["classifiers"])
print("\nGenerated labels for training data successfully. ")


print("\nGenerating labels for validation data . . . ")
y_pred_valid, _ = apply_classifiers(X=features_norm_valid, classifiers=classifiers_config)
print("\nGenerated labels for validation data successfully. ")

print("Evaluating Metrics on training data . . . ")
metrics_train, metrics_config = evaluate_metrics(y_true=y_train, y_pred=y_pred_train, metrics=pipeline["metrics"])
print("Evaluated Metrics on the training data successfully")

# print results
#for index in range(len())

print("Evaluating Metrics on validation data . . . ")
metrics_valid, _ = evaluate_metrics(y_true=y_valid, y_pred=y_pred_valid, metrics=metrics_config)
print("Evaluated Metrics on the validation data successfully")

# print results
#for index in range(len())
 
plot_CM(y_true=y_train, y_pred=y_pred_train, path_to_results=path_to_results)

plot_ROC()

# concencate all the returuned configs and save it json
save_to_json(config, path_to_results)

# save metrics train and metrics_valid