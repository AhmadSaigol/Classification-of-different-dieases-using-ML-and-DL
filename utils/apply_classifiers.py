"""
Applies each classifier and generate predictions

"""
import os
import numpy as np
from classifiers.SVM import svm
from classifiers.RFTree import rftree

def apply_classifiers(X, classifiers, classes, y, path_to_results=None):
    """
    Applies each classifier to the input and generates prediction

    Parameters:
        X: numpy array of shape (folds, num_images, num_features)
        classes: numpy array of names of classes
        classifiers: dictionary with following structure:
            classifiers["classifiers_1"]["function"] = pointer to the function
            classifiers["classifiers_1"]["parameter_1"] = value
            classifiers["classifiers_1"]["parameter_2"] = value

            classifiers["classifiers_2"]["function"] = pointer to the function
            classifiers["classifiers_2"]["parameter_1"] = value
            classifiers["classifiers_2"]["parameter_2"] = value

        y: labels of the data. numpy array of shape (folds, num_of_images, 2) or (folds,num_images, 1)
        if last axis = 2, then it would be inferred that the function is being called during training phase 
        path_to_results: (required in testing phase) where the results will be saved (required for training since trained_models will be saved)


    Returns:
        predictions: (classifer, folds, num_images, 2)
        output_config:
        list_of_classifiers: since data is incoming in dictionary format, so for easy mapping of which index represents to which classifier

    Additional Notes:

        currently the parameters in the output dictionary will be only of first fold for each classifier


    """

    output_config = {}

    num_folds = X.shape[0]
    
    flag = True
    
    # determine whether we are in training phase or testing phase
    if y.shape[-1] == 2 and path_to_results:
        train = True
    elif y.shape[-1] == 1 and not path_to_results :
        train = False
    else:
        raise ValueError("Unable to determine the phase. please check the parameters 'y' and 'path_to_results'. ")

    
    
    if train:
         
        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)
            print(f"Created directory {path_to_results}")
        else:
            print(f"Warning: {path_to_results} already exists. Contents may get overwritten")


    
    
    list_of_classifiers = []

    for classifier in classifiers.keys():
        
        output_config[classifier] ={}

        if train:
            output_config[classifier]["path_to_models"] = {}
        
        fnt_pointer = classifiers[classifier]["function"]

        for fold_no in range(num_folds):
            
            print(f"Applying Classifier: {classifier} Fold No: {fold_no}")
            
            
            if fold_no < 10:
                fn = "0" + str(fold_no)
            else:
                fn = str(fold_no)

            # get predictions

            if train:
                print("In Training Phase ")
                # create directory for fold
                path_to_fold = os.path.join(path_to_results, str(fold_no))
                if not os.path.exists(path_to_fold):
                    os.mkdir(path_to_fold)
                    
                # create directory for model
                save_model_path = os.path.join(path_to_fold, "models")
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)

                print("confing ", output_config) 
                output_config[classifier]["path_to_models"][fn] = save_model_path
            
                prediction, fnt_config = fnt_pointer(X=X[fold_no], parameters=classifiers[classifier], 
                                                    classes=classes, y=y[fold_no], save_model_path=save_model_path)

            
            else:
                print("In Testing Phase ")
                path_to_model = classifiers[classifier]["path_to_models"][fn]
                
                prediction, fnt_config = fnt_pointer(X=X[fold_no], parameters=classifiers[classifier], 
                                                    classes=classes, y=y[fold_no], path_to_model=path_to_model)
            
            print ("prediction ", prediction)
            
            # concatenate y_pred for folds  
            prediction = np.expand_dims(prediction, axis=0)

            if fold_no==0:
                fnt_config["function"] = fnt_pointer
                output_config[classifier] = {**output_config[classifier], **fnt_config}
                y_pred = prediction
            else:
                y_pred = np.concatenate((y_pred, prediction))
                
            print("After fold shape ", y_pred.shape)

        # concatenate y_pred for classifiers             
        y_pred = np.expand_dims(y_pred, axis=0)

        if flag:
            y_preds = y_pred
        else:
            y_preds = np.concatenate((y_preds, y_pred)) 
                
        print("After classifier shape ", y_pred.shape)
        

        # save classifier
        list_of_classifiers.append(classifier)

    return y_preds, output_config, list_of_classifiers


if __name__ == "__main__":

    print("Training ")

    X = np.random.rand(2,3,5)
    print("X ", X)
    print("X shape ", X.shape)

    y = np.array([  "aa", "b", "ac", "a", "da","b",
                    "cc","b", "dc", "a", "da", "b"]).reshape(2,3,2)
    print("y ", y)
    print("y shape ", y.shape)

    
    classes = np.array(["a", "b"])
    
    pipeline = {}

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
    
    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"

    pred, config, classif = apply_classifiers(X=X, classifiers=pipeline["classifiers"], classes=classes, y=y, path_to_results=path_to_results)

    print("pred ", pred)
    print("pred shape ", pred.shape)

    print ("classifiers: ", classif)

    print("config ", config)
    #-------------------------------------------------------

    print("Testing")


    X_test = np.random.rand(1,1,5)
    print("X ", X_test)
    print("X shape ", X_test.shape)

    y_test = np.array([ "test" ]).reshape(1,1,1)
    print("y ", y_test)
    print("y shape ", y_test.shape)

    test_pred, test_config, test_classifi = apply_classifiers (X=X_test, classifiers=config, classes=classes, y=y_test, )


    print("test pred ", test_pred)
    print("test pred shape ", test_pred.shape)

    print ("test classifiers: ", test_classifi)

    print("test config ", test_config)