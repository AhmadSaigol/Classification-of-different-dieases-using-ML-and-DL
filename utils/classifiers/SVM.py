import cv2
import numpy as np
import os
import sys

#import label encoder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from process_labels import label_encoder

def svm(X, parameters, classes, y, save_model_path=None, path_to_model=None):
    """
        Support vector machine

        Parameters:
            X: numpy array of shape (num_images, num_features)
            parameters: dictionary containing the keys:
                svm_type: (default = "C_SVC") type of svm. currently supports: "C_SVC", "NU_SVC", "EPS_SVR" or "NU_SVR" 
                kernel:(default = "RBF") kernel to be used. currently supports: "LINEAR", "POLY", "RBF", "SIGMOID", "CHI2" or "INTER"
                Gamma: (default = 1.0)
                Degree:(default = 0.0)
                P:(default = 0.0)
                Nu:(default = 0.0)
                Coef0: (default = 0.0)
                C:(default = 1.0)
                trainAuto: (default = False) whether to find the optimal parameters from grid of parameters

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


            output config can contain keys that are not relevant to the selected kernelType/SVMtype. They are to be ignored.

            For more info, see
            "https://docs.opencv.org/3.4/d1/d2d/classcv_1_1ml_1_1SVM.html#a6a86483c5518c332fedf6ec381a1daa7"
    
    """
    #changing datatype of array since svm works on float32
    X=X.astype(np.float32)

    # determine phase
    if  save_model_path and not path_to_model:
        train = True
    elif not save_model_path and path_to_model:
        train = False
    else:
        raise ValueError("Unable to determine the phase. Check parameters 'y', 'save_model_path' and 'path_to_model'. ")


    svm_types = ["C_SVC", "NU_SVC", "EPS_SVR", "NU_SVR"]
    svm_types_enums = [cv2.ml.SVM_C_SVC, cv2.ml.SVM_NU_SVC, cv2.ml.SVM_EPS_SVR, cv2.ml.SVM_NU_SVR]

    kernels = ["LINEAR", "POLY", "RBF", "SIGMOID", "CHI2", "INTER"]
    kernels_enums = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_POLY, cv2.ml.SVM_RBF, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER]


    if train:

        #setup model
        model = cv2.ml.SVM_create()
        
        clf_keys =parameters.keys()

        #encode labels 
        labels = label_encoder(y=y[:,1], classes=classes, to_numbers=True)

        if "svm_type" in clf_keys:
            index = svm_types.index (parameters["svm_type"])
            model.setType( svm_types_enums[index])
        
        if "kernel" in clf_keys:
            index =kernels.index(parameters["kernel"])
            model.setKernel(kernels_enums[index])
        
        if "Gamma" in clf_keys:
            model.setGamma (parameters["Gamma"])
        
        if "Degree" in clf_keys:
            model.setDegree (parameters["Degree"])

        if "P" in clf_keys:
            model.setP(parameters["P"])
        
        if "Nu" in clf_keys:
            model.setNu(parameters["Nu"])
        
        if "Coef0" in clf_keys:
            model.setCoef0(parameters["Coef0"])

        if "C" in clf_keys:
            model.setC(parameters["C"])

        if "trainAuto" in clf_keys:
            auto = parameters["trainAuto"]
        else:
            auto = False
        
        # set stopping criteria
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

        # can also class weights in case of misclassification of a class
        # svm.setClassWeights ()
    
        # train svm
        if auto:
            print("Training SVM while finding for optimal parameters")
            model.trainAuto(X, cv2.ml.ROW_SAMPLE, labels)
        
        else:

            print("Training SVM with provided paramters")
            model.train(X, cv2.ml.ROW_SAMPLE, labels)


        #save model
        model.save(os.path.join(save_model_path, "svm.dat"))
    
    else:
        print(f"Loading SVM from {path_to_model}")
        model = cv2.ml.SVM_load(os.path.join(path_to_model, "svm.dat"))


    print("Using SVM to generate the results")
    # get predictions 
    predictions = model.predict(X)[1]

    #get names of class
    predictions = label_encoder(predictions, classes=classes, to_numbers=False)

    y_pred = np.concatenate((y[:,0, None], predictions[:, None]), axis=-1)

    # setup output config
    config = {}
    
    index = svm_types_enums.index (model.getType())
    config["svm_type"] = svm_types[index]
        
    index =kernels_enums.index(model.getKernelType())
    config["kernel"] = kernels[index]
       
    config["Gamma"] = model.getGamma()     
    config["Degree"]= model.getDegree ()
    config["P"]= model.getP()     
    config["Nu"] = model.getNu()     
    config["Coef0"] = model.getCoef0()
    config["C"] = model.getC()
    
    sc, num_iters, eps = model.getTermCriteria()
    config['max_nums_iter'] = num_iters
    config['epsilon'] = eps


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
    
    save_model_path= "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing"
    
    # SVM
    clf={}
    clf["svm"] = {}
    clf["svm"]["function"] = svm #some function pointer
    clf["svm"]["trainAuto"] = True
    clf["svm"]['svm_type'] =  'C_SVC' 
    clf["svm"]['kernel'] =  'RBF' 
    clf["svm"]['Gamma'] =  5 
    clf["svm"]['Degree'] =  10 
    clf["svm"]["P"] =  0.5 
    
    print("y_test")
    print(y_test)
    print(y_test.shape)
    
    print("testData")
    print(testData)
    print(testData.shape)
    
    y_pred, config = svm (testData, clf["svm"], classes, y=y_test ,save_model_path=None, path_to_model=save_model_path)
    
    print(y_pred)
    print(config)

