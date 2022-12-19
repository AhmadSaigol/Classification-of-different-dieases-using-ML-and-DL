import cv2
import numpy as np
import os
import sys

#import label encoder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from process_labels import label_encoder

def rftree(X, parameters, classes, y, save_model_path=None, path_to_model=None):
    """
        random forest tree

        Parameters:
            X: numpy array of shape (num_images, num_features)
            parameters: dictionary containing the keys:
                ActiveVarCount: (default=sqrt(num_features))The size of the randomly selected subset of features at each tree node and that are used to find the best split(s).
                MaxDepth: depth of trees (default = 5)

            classes: numpy array with names of classes
            y: labels of the data. numpy array of shape (num_of_images, 2) or (num_of_images, 1)
            save_model_path: (optional) where to save the trained model
            path_to_model: (optional) path to the foler where trained model is saved
        
        Returns:
            y_pred : numpy array of shape(num_images, 2)
            config: dictionary with parameters of the function (including default parameters)

        Additional Notes:
            if save_model_path is provided, then program will train the model and generate predicitons (training phase)
            if path_to_model is provided, then program will load the model and generate the prediction (testing phase)

            if "ActiveVarCount" is zero in the output config, that means the parameter was set to default value

            For more info, see
            "https://docs.opencv.org/3.4/d0/d65/classcv_1_1ml_1_1RTrees.html"

    """
    #changing datatype of array since rftree works on float32
    X=X.astype(np.float32)

    # determine phase
    if  save_model_path and not path_to_model:
        train = True
    elif  not save_model_path and path_to_model:
        train = False
    else:
        raise ValueError("Unable to determine the phase. Check parameters 'y', 'save_model_path' and 'path_to_model'. ")

    if train:

        #setup model
        model = cv2.ml.RTrees_create()
        
        clf_keys =parameters.keys()
        
        #encode labels 
        labels = label_encoder(y=y[:,1], classes=classes, to_numbers=True)

   
        if "ActiveVarCount" in clf_keys:
            model.setActiveVarCount( parameters["ActiveVarCount"])

        
        # can calculate each variable importance (during training phase)
        # model.setCalculateVarImportance(True)

        # can get OOB error calculated during training phase
        # model.calcOOBError ()
        
        # set depth of trees
        if "MaxDepth" in clf_keys:
            model.setMaxDepth (parameters["MaxDepth"])
        else:
            model.setMaxDepth (5)

        # can get weights for each class
        # model.setPriors([])
        """
        # get stopping critera
        if 'stopping_criteria' in clf_keys:
            sc = parameters['stopping_criteria'] 
        else:
            sc = 'both'

        if sc == 'max_iters':
            sc_type = cv2.TERM_CRITERIA_MAX_ITER
        elif sc == 'accu':
            sc_type = cv2.TERM_CRITERIA_EPS
        elif sc == 'both':
            sc_type = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
        else:
            raise ValueError("Unknown value encountered for 'stopping_criteria' in RFTree parameters")

        
        if 'max_nums_iter' in clf_keys:
            max_nums_iter = parameters['max_nums_iter']
        else:
            max_nums_iter = 10

        if 'epsilon' in clf_keys:
            ep = parameters['epsilon']
        else:
            ep = 1e-6
        """


        # can set termination criteria
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-6))

        print("Training Random Forest Tree")
        model.train(X, cv2.ml.ROW_SAMPLE, labels)

        #save model
        model.save(os.path.join(save_model_path, "RFTree.dat"))
    
    else:
        print(f"Loading Random Forest Tree from {path_to_model}")
        model = cv2.ml.RTrees_load(os.path.join(path_to_model, "RFTree.dat"))

    
    print("Using Random Forrest Tree to generate the results")
    # get predictions 
    predictions = model.predict(X)[1]

    #get names of class
    predictions = label_encoder(predictions, classes=classes, to_numbers=False)

    y_pred = np.concatenate((y[:,0, None], predictions[:, None]), axis=-1)

    # setup output config
    config = {}
    config["ActiveVarCount"] = model.getActiveVarCount()
    config["MaxDepth"] =model.getMaxDepth()
    
    sc, num_iters, eps = model.getTermCriteria()
    """
    if sc == cv2.TERM_CRITERIA_MAX_ITER:
        config['stopping_criteria'] = 'max_iters'
    elif sc == cv2.TERM_CRITERIA_EPS:
        config['stopping_criteria'] = 'accu'
    elif sc == (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS):
        config["stopping_criteria"] = 'both'
    else:
        raise ValueError ("Unknown Value Encountered while reading stopping critera type from RFTree model")
    """

    config['max_nums_iter'] = num_iters
    config['epsilon'] = eps

    
    return y_pred, config


if __name__ == "__main__":

    y = np.array([  "aa", "a", "ac", "c", "da","b",
                    "cc","b", "dc", "a", "da", "c"]).reshape(-1,2)
    y  = np.concatenate((y, y, y), axis=0)

    y_test= np.array([  "aa","ac", "da",
                    "cc", "dc", "da"]).reshape(-1,1)
    trainingData = np.random.rand(18,4).astype(np.float32)
    #trainingData = np.concatenate((train.ingData, trainingData, trainingData))
    
    testData = np.random.rand(6,2).astype(np.float32)

    classes =np.array(["a", "b", "c"])
    
    save_model_path= "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/experiments/stopping_criteria/"
    
    # SVM
    clf={}
    clf["rftree"] = {}
    clf["rftree"]["function"] = 0 #some function pointer
    clf["rftree"]["ActiveVarCount"] = 1

    
    print("y")
    print(y)
    print(y.shape)
    
    print("trainingData")
    print(trainingData)
    print(trainingData.shape)
    
    y_pred, config = rftree (trainingData, clf["rftree"], classes, y=y ,save_model_path=save_model_path, path_to_model=None)
    
    print(y_pred)
    print(config)

