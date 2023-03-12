import cv2
import numpy as np
import os
import sys

#import label encoder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from process_labels import label_encoder

def boosting(X, parameters, classes, y, save_model_path=None, path_to_model=None):
    """
        Boosting using decision trees

        Parameters:
            X: numpy array of shape (num_images, num_features)
            parameters: dictionary containing the keys:
                boost_type: "REAL" (default) or "DISCRETE" 
                num_weak_classifiers: default = 100
                max_depth: default = 5
            classes: numpy array with names of classes
            y: labels of the data. numpy array of shape (num_of_images, 2) or (num_of_images, 1)
            save_model_path: (optional) where to save the trained model
            path_to_model: (optional) path to the foler where trained model is saved
        
        Returns:
            y_pred : numpy array of shape(num_images, 2)
            config: dictionary with parameters of the function (including default values)

        Additional Notes:
            if save_model_path is provided, then program will train the model and generate predicitons (training phase)
            if path_to_model is provided, then program will load the model and generate the prediction (testing phase)


            For more info, see
            "https://docs.opencv.org/3.4/d6/d7a/classcv_1_1ml_1_1Boost.html#details"
            "https://docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_boost"
    
    """

    #changing datatype of array
    X=X.astype(np.float32)

    # determine phase
    if  save_model_path and not path_to_model:
        train = True
    elif not save_model_path and path_to_model:
        train = False
    else:
        raise ValueError("Unable to determine the phase. Check parameters 'y', 'save_model_path' and 'path_to_model'. ")


    if train:
        
        #setup model
        model = cv2.ml.Boost_create()
        
        clf_keys =parameters.keys()

        #encode labels 
        labels = label_encoder(y=y[:,1], classes=classes, to_numbers=True)

        # get boost type
        if "boost_type" in clf_keys:
            if parameters["boost_type"] == "REAL":
                model.setBoostType(cv2.ml.Boost_REAL )
            elif parameters["boost_type"] == "DISCRETE":
                model.setBoostType(cv2.ml.Boost_DISCRETE)
            else:
                raise ValueError("unknown value encountered while reading boost type from parameters dictionary.")
        else:
            model.setBoostType(cv2.ml.Boost_REAL )

        #get number of weak classifiers
        if "num_weak_classifiers" in clf_keys:
            model.setWeakCount(parameters["num_weak_classifiers"])
        else:
            model.setWeakCount(100)

        # get number of max depth of decision trees   
        if "max_depth" in clf_keys:
            model.setMaxDepth (parameters["max_depth"])
        else:
            model.setMaxDepth (5)

        # can get weights for each class
        # model.setPriors([])

         # train the model
        print("Training Boosting")
        model.train(X, cv2.ml.ROW_SAMPLE, labels)


        #save model
        model.save(os.path.join(save_model_path, "boosting.dat"))
    
    else:
        
        print(f"Loading Boosting from {path_to_model}")
        model = cv2.ml.Boost_load(os.path.join(path_to_model, "boosting.dat"))


    print("Using Boosting to generate the results")
    # get predictions 
    predictions = model.predict(X)[1]

    #get names of class
    predictions = label_encoder(predictions, classes=classes, to_numbers=False)

    y_pred = np.concatenate((y[:,0, None], predictions[:, None]), axis=-1)

    # setup output config
    config = {}
    
    boost_type = model.getBoostType()
    if boost_type == 0:
        config["boost_type"] = "DISCRETE"
    elif boost_type == 1:
        config["boost_type"] = "REAL"
    else:
        raise ValueError("Unknown Value encountered while reading boosting type from the model")

    config["num_weak_classifiers"] = model.getWeakCount()

    config["max_depth"] =model.getMaxDepth()    
    

    return y_pred, config

if __name__ == "__main__":

    y = np.array([  "aa", "b", "ac", "c", "da","b",
                    "cc","b", "dc", "a", "da", "b"]).reshape(-1,2)
    y  = np.concatenate((y, y, y), axis=0)

    y_test= np.array([  "aa","ac", "da",
                    "cc", "dc", "da"]).reshape(-1,1)
    trainingData = np.arange(12, dtype=np.float32).reshape(-1, 2)
    trainingData = np.concatenate((trainingData, trainingData, trainingData))
    
    testData = np.random.rand(6,2).astype(np.float32)

    classes =np.array(["a", "b", "c"])
    
    save_model_path= "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/final_test"
    
    # boosting
    clf={}
    clf["boosting"] = {}
    clf["boosting"]["function"] = boosting #some function pointer
    
    print("y_test")
    print(y_test)
    print(y_test.shape)
    
    print("testData")
    print(testData)
    print(testData.shape)
    
    y_pred, config = boosting (trainingData, clf["boosting"], classes, y=y ,save_model_path=save_model_path, path_to_model=None)
    
    print(y_pred)
    print(config)
