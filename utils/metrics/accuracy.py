"""
Calculates accuracy

"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np

def accuracy(y_true, y_pred, parameters):
    """
    Calculates accuracy
    
    Parameters:
        y_true: numpy array of shape (num_images,)
        y_pred: numpy array of shape (num_images,)
        parameters: dictionary with the following keys:
            type: "simple" (default) or "balanced"
    
    Returns:
        score: float
        config:

    Additional Notes:

        Simple Accuracy: fraction of labels that are equal.
        Balanced Accuracy: 
            Binary class: equal to the arithmetic mean of sensitivity (true positive rate) 
                            and specificity (true negative rate)
                             = 1/2 ( (TP/TP+FN) + (TN/TN+FP))

            Multiclass: the macro-average of recall scores per class
                        recall for each class and then take the mean
                        recall = TP /(TP+FN)

            if the classifier predicts same label for all examples, the score will be equal to 1/num_classes (for binary: 0.5)
            for more info, see
                "https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score"
    """

    config={}
    if "type" in parameters.keys():
        accu_type = parameters["type"]
    else:
        accu_type = "simple"

    config["type"] = accu_type
    
    if accu_type == "simple":
        score = accuracy_score(y_true=y_true, y_pred=y_pred) 
    elif accu_type == "balanced":
        score = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError("Unknown Value encountered for parameter 'type' while calculating accuracy")#

    
    return score, config

if __name__ == "__main__":

    #binary
    y_true = np.array(["aa", "a", "ab", "b", "ac", "a"]).reshape(-1,2)
    y_pred = np.array(["aa", "b", "ab", "b", "ac", "b"]).reshape(-1,2)
    
    
     
    # multi
    #y_true = np.array(["aa", "a", "ab", "b", "ac", "c"]).reshape(-1,2)
    #y_pred = np.array(["aa", "c", "ab", "b", "ac", "c"]).reshape(-1,2)
    
    
    accu={}
    accu["accuracy"] = {}
    accu["accuracy"]["function"] =0 #name of functions to be used for ensemblers
    accu["accuracy"]["type"] = "balanced"
    print("true")
    print(y_true[:,1])

    print("pred")
    print(y_pred[:,1])

    score, config = accuracy(y_true=y_true[:,1], y_pred=y_pred[:,1], parameters=accu["accuracy"])

    print(score)
    print(config)
    



