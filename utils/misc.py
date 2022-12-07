
import numpy as np
import os
import pandas as pd

from data_preprocessing.change_colorspace import change_colorspace
from data_preprocessing.normalize import normalize

from feature_extractors.contrast import calculate_contrast
from feature_extractors.skewness import calculate_skew
from classifiers.SVM import svm
from classifiers.RFTree import rftree

from metrics.accuracy import accuracy
from metrics.f1_score import F1_score
from metrics.mcc import mcc
from metrics.precision import precision



def add_function_names(dic):
    """
    Finds function in the dictionary, and replace its value with the function name
    
    Parameters:
        dict: dictionary

    """

    d_keys = dic.keys()

    for key in d_keys:
        #print(key)
        values = dic[key]
        #print(values)

        if type(values) == dict:
            add_function_names(values)
        
        else:
            if key== "function":
                dic[key] = values.__name__



def generate_txt_file(y, path_to_results, classifiers, name_of_file):
    """
    Generates a text file with structure as follows:
    file_name_1 label
    file_name_2 label


    Parameters:
        y: numpy array of shape(classifiers, folds, num_images, 2)
        path_to_results: path to the folder where to store results
        classifiers: names of classifiers
        name_of_file (without .txt)

    """

    num_classifiers = y.shape[0]
    num_folds = y.shape[1]
    num_images = y.shape[2]

    for cl in range(num_classifiers):

        for fold_no in range(num_folds):

            print(f"Processing Classifier: {classifiers[cl]} Fold No: {fold_no}")

            path_to_fold = os.path.join(path_to_results, str(fold_no))
            if not os.path.exists(path_to_fold):
                    os.mkdir(path_to_fold)
            
            path_to_file = path_to_fold + "/" + name_of_file + "_" + classifiers[cl] +".txt"

            if not os.path.exists(path_to_file):
                open(path_to_file, "w").close()
            
            with open(path_to_file, "a") as file:

                for img_no in range(num_images):
                    file.write(f"{y[cl, fold_no, img_no, 0]} {y[cl, fold_no, img_no, 1]}\n")


def save_results(results, classifiers, metrics, path_to_results, name_of_file):
    """
    Save results to csv file

    Parameters:
        results: numpy array of shape (classifiers, num_folds, metrics)
        classifiers: numpy array of name of the classifiers
        metrics: numpy array of name of the metrics
        path_to_results: path to the folder where the results will be stored
        name_of_file: file name

    """
    df = pd.DataFrame([], columns=["Classifier", "Fold No", "Metric", "Score"])
    num_classifiers = results.shape[0]
    num_folds = results.shape[1]
    num_metrics = results.shape[2]

    for cl_no in range(num_classifiers):
        for fold_no in range(num_folds):
            for metric_no in range(num_metrics):
                temp = pd.DataFrame([[classifiers[cl_no], fold_no, metrics[metric_no], results[cl_no, fold_no, metric_no]]], columns=["Classifier", "Fold No", "Metric", "Score"])
                df = pd.concat([df, temp])
    df.reset_index(inplace=True, drop=True ) 
       
    df.to_csv(path_to_results + "/" + name_of_file +".csv")
        
if __name__ == "__main__":
    pipeline = {}

    pipeline["data"] = {}

    # can be to folder 
    pipeline["data"]["path_to_images"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train"
    # can be to .txt
    pipeline["data"]["path_to_labels"] = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train.txt"

    # split data
    pipeline["data"]["split_data"] = "simple"  #"simpleStrafied", "kfold", "kfoldStratified"

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

    # random forest tree
    pipeline["classifiers"]["RFTree"] = {}
    pipeline["classifiers"]["RFTree"]["function"] =rftree
    pipeline["classifiers"]["RFTree"]["ActiveVarCount"] =0 


    #---------------------------------------------set up evaluation metrics and parameters------------------------


    pipeline["metrics"] = {}

    # accuracy
    pipeline["metrics"]["simple_accuracy"] = {}
    pipeline["metrics"]["simple_accuracy"]["function"] = accuracy
    pipeline["metrics"]["simple_accuracy"]["type"] = "simple"  #value of parameter

    # precision
    pipeline["metrics"]["precision"] = {}
    pipeline["metrics"]["precision"]["function"] = precision #name of functions to be used for ensemblers
    pipeline["metrics"]["precision"]["class_result"] = "COVID"
    pipeline["metrics"]["precision"]["average"] = "weighted"


    add_function_names(pipeline)

    print(pipeline)

    y_pred = np.array(["aa", "b", "ab", "b", "ac", "b",
                        "ad", "a", "ae", "c", "af", "a",
                        "ag", "a", "ah", "b", "ai", "c",
                        "aj", "b", "ak", "c", "al", "b",
                        "am", "c", "an", "a", "ao", "a",
                        "ap", "a", "aq", "c", "ar", "b"]).reshape(2,3,3,2)


    path = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"
    
    classifiers = ["cl1", "cl2"]

    name = "test"

    generate_txt_file(y_pred, path, classifiers, name)
