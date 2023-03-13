from copy import deepcopy


def compare_dict(d1, d2):
    """
    Compares two dictionaries and returns whether they are exactly same or not

    Parameters:
        d1: dict
        d2: dict
    
    """

    if d1.keys()!=d2.keys():
        return False
    
    # for each key
    for key in d1.keys():
        
        # check type of value at the key
        if type(d1[key]) != type(d2[key]):
            return False

        # if key has a dict, recursive call
        if type(d1[key]) == dict:
            result = compare_dict(d1[key], d2[key])
            if not result:
                return False

        # check values
        if d1[key] != d2[key]:
            return False
    
    return True
         

if __name__ == "__main__":

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

    # map to RGB
    pipeline["data_preprocessing"]["map_to_RGB"] = {}
    pipeline["data_preprocessing"]["map_to_RGB"]["function"] =0 #some function pointer
    pipeline["data_preprocessing"]["map_to_RGB"]["some_parameter"] =0 #value of parameter
    pipeline["data_preprocessing"]["map_to_RGB"]["some_parameter"] =0 #value of parameter


    # normalize image
    pipeline["data_preprocessing"]["normalize_image"] = {}
    pipeline["data_preprocessing"]["normalize_image"]["function"] =0 #some function pointer
    pipeline["data_preprocessing"]["normalize_image"]["some_parameter"] =0 #value of parameter

    # resize_image
    pipeline["data_preprocessing"]["resize_image"] = {}
    pipeline["data_preprocessing"]["resize_image"]["function"] =0 #some function pointer
    pipeline["data_preprocessing"]["resize_image"]["some_parameter"] =0 #value of parameter


    # ---------------------------------set up feature extractor methods and parameters------------------------------------
    pipeline["feature_extractors"] ={}

    # contrast
    pipeline["feature_extractors"]["contrast"] = {}
    pipeline["feature_extractors"]["contrast"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["contrast"]["some_parameter"] =0 #value of parameter
    pipeline["feature_extractors"]["contrast"]["some_parameter"] =0 #value of parameter


    # energy
    pipeline["feature_extractors"]["energy"] = {}
    pipeline["feature_extractors"]["energy"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["energy"]["some_parameter"] =0 #value of parameter

    # variance
    pipeline["feature_extractors"]["variance"] = {}
    pipeline["feature_extractors"]["variance"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["variance"]["some_parameter"] =0 #value of parameter

    # skewness
    pipeline["feature_extractors"]["skewness"] = {}
    pipeline["feature_extractors"]["skewness"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["skewness"]["some_parameter"] =0 #value of parameter

    # kurtosis
    pipeline["feature_extractors"]["kurtosis"] = {}
    pipeline["feature_extractors"]["kurtosis"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["kurtosis"]["some_parameter"] =0 #value of parameter

    # RMS
    pipeline["feature_extractors"]["RMS"] = {}
    pipeline["feature_extractors"]["RMS"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["RMS"]["some_parameter"] =0 #value of parameter

    # hue_moments
    pipeline["feature_extractors"]["hue_moments"] = {}
    pipeline["feature_extractors"]["hue_moments"]["function"] =0 #some function pointer
    pipeline["feature_extractors"]["hue_moments"]["some_parameter"] =0 #value of parameter


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


    pipeline2 = deepcopy(pipeline)
    pipeline["metrics"]["f1_score"] ["function"]=41

    if compare_dict(pipeline, pipeline2):
        print(True)
    else :
        print(False)