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

from utils.apply_classifiers import apply_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree

from utils.evaluate_metrics import evaluate_metrics
from utils.metrics.accuracy import accuracy
from utils.metrics.f1_score import F1_score
from utils.metrics.mcc import mcc
from utils.metrics.precision import precision
from utils.metrics.sensitivity import sensitivity

verbose = True

pipeline = {}
pipeline["path_to_results"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"


#------------------ setup data------------------------

pipeline["data"] = {}

# can be to folder 
pipeline["data"]["path_to_images"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
# can be to .txt
pipeline["data"]["path_to_labels"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train.txt"

# split data
pipeline["data"]["split_data"] = "simple" , #"simpleStrafied", "kfold", "kfoldStratified"

pipeline["data"]["classes"] = np.array(["Normal", "COVID", "pneumonia", "Lung_Opacity"]) # np.array(["No-COVID", "COVID"]) keep order same

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
pipeline["classifiers"]["svm"]["function"] = svm 
pipeline["classifiers"]["svm"]["trainAuto"] = True 
pipeline["classifiers"]["svm"]['svm_type'] =  'C_SVC' 
pipeline["classifiers"]["svm"]['kernel'] =  'RBF'
pipeline["classifiers"]["svm"]['Gamma'] =  0.2 


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


# ensemble learning
#pipeline["classifiers"]["ensemble"] = {}
#pipeline["classifiers"]["ensemble"]["function"] =["svm", "decision_tree"] #name of functions to be used for ensemblers
#pipeline["classifiers"]["decision_tree"]["some_parameter"] =0 #value of parameter


#---------------------------------------------set up evaluation metrics and parameters------------------------


pipeline["metrics"] = {}

# accuracy
pipeline["metrics"]["simple_accuracy"] = {}
pipeline["metrics"]["simple_accuracy"]["function"] = accuracy
pipeline["metrics"]["simple_accuracy"]["type"] = "simple"  #value of parameter

pipeline["metrics"]["balanced_accuracy"] = {}
pipeline["metrics"]["balanced_accuracy"]["function"] = accuracy
pipeline["metrics"]["balanced_accuracy"]["type"] = "balanced"

# precision
pipeline["metrics"]["precision"] = {}
pipeline["metrics"]["precision"]["function"] = precision #name of functions to be used for ensemblers
pipeline["metrics"]["precision"]["class_result"] = "COVID"
pipeline["metrics"]["precision"]["average"] = "weighted"

# recall
pipeline["metrics"]["sensitivity"] = {}
pipeline["metrics"]["sensitivity"]["function"] = sensitivity
pipeline["metrics"]["sensitivity"]["class_result"]  = "COVID"
pipeline["metrics"]["sensitivity"]["average"]  = "weighted"

# F1 score
pipeline["metrics"]["f1_score"] = {}
pipeline["metrics"]["f1_score"]["function"] = F1_score 
pipeline["metrics"]["f1_score"]["class_result"] = "COVID"
pipeline["metrics"]["f1_score"]["average"] = "weighted"

# mcc
pipeline["metrics"]["mcc"] = {}
pipeline["metrics"]["mcc"]["function"] = mcc 




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

print("data processing config ", data_config)

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
features_train, features_train_config = generate_feature_vector(X=X_train, extractors=pipeline["feature_extractors"])
print(f"Generated Features for training data successfully.")

print("the shape of features_train: ", features_train.shape)

print("features config ", features_train_config)


print("\nGenerating Features for validation data . . . ")
features_valid, features_valid_config =generate_feature_vector(X=X_valid, extractors=features_train_config)
print(f"Generated Features for validation data successfully.")

print("the shape of features_valid: ", features_valid.shape)

print("features config ", features_valid_config)


# normalize vectors
print("\nNormalizing Features for training data . . . ")
features_norm_train, features_norm_train_config = normalize_features(features=features_train, parameters=pipeline["normalize_features"])
print(f"Normalized Features for training data successfully. Shape of normalized feature vector: {features_norm_train.shape}")
print("the shape of features_norm_train: ", features_norm_train.shape)

print("features norm config ", features_norm_train_config)


print("\nNormalizing Features for validation data . . . ")
features_norm_valid, features_norm_valid_config = normalize_features(features=features_valid, parameters=features_norm_train_config)
print(f"Normalized Features for validation data successfully. Shape of normalized feature vector: {features_norm_valid.shape}")
print("the shape of features_norm_valid: ", features_norm_valid.shape)

print("features norm config ", features_norm_valid_config)



# get prediction

print("\nGenerating labels for training data . . . ")
y_pred_train, classifiers_train_config, classifers_train_list = apply_classifiers(X=features_norm_train, y=y_train, classes=pipeline["data"]["classes"], classifiers=pipeline["classifiers"], path_to_results= pipeline["path_to_results"])
data_config["data"]["classes"] = pipeline["data"]["classes"]
data_config["path_to_results"] = pipeline["path_to_results"]
print("\nGenerated labels for training data successfully. ")

print("the shape of y_pred_train: ", y_pred_train.shape)

print("classifiers_train config ", classifiers_train_config)

print("classifiers_list ", classifers_train_list)

print("\nGenerating labels for validation data . . . ")
y_pred_valid, classifiers_valid_config, classifers_valid_list = apply_classifiers(X=features_norm_valid, classifiers=classifiers_train_config)
print("\nGenerated labels for validation data successfully. ")

print("the shape of y_pred_valid: ", y_pred_valid.shape)

print("classifiers_valid config ", classifiers_valid_config)

print("classifiers_list ", classifers_valid_list)





print("Evaluating Metrics on training data . . . ")
metrics_train, metrics_train_config, metrics_train_list = evaluate_metrics(y_true=y_train, y_pred=y_pred_train, metrics=pipeline["metrics"])
print("Evaluated Metrics on the training data successfully")

print("shape of train metrics score ", metrics_train.shape)
print("metrics train config ", metrics_train_config)
print("metrics train list ", metrics_train_list)

for cl_no, cl in enumerate(classifers_train_list):

    for fold_no in range(metrics_train.shape[1]):
        
        for met_no, met in enumerate(metrics_train_list):

            print(f"Classifer: {cl} Fold No: {fold_no} Metric: {met}  Score: {metrics_train[cl_no, fold_no, met_no]} ")




print("Evaluating Metrics on validation data . . . ")
metrics_valid, metrics_valid_config, metrics_valid_list = evaluate_metrics(y_true=y_valid, y_pred=y_pred_valid, metrics=metrics_train_config)
print("Evaluated Metrics on the validation data successfully")

print("shape of valid metrics score ", metrics_valid.shape)
print("metrics train config ", metrics_valid_config)
print("metrics train list ", metrics_valid_list)

for cl_no, cl in enumerate(classifers_valid_list):

    for fold_no in range(metrics_valid.shape[1]):
        
        for met_no, met in enumerate(metrics_valid_list):

            print(f"Classifer: {cl} Fold No: {fold_no} Metric: {met}  Score: {metrics_valid[cl_no, fold_no, met_no]} ")



plot_CM(y_true=y_train, y_pred=y_pred_train, path_to_results=path_to_results)

plot_ROC()

# concencate all the returuned configs and save it json
save_to_json(config, path_to_results)

# save metrics train and metrics_valid