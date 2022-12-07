"""
Calculates Sensitivtiy ( Recall )
"""
from sklearn.metrics import recall_score
import numpy as np

def sensitivity(y_true, y_pred, parameters):
    """
    Calculates sensitivity (recall)
    
    Parameters:
        y_true: numpy array of shape (num_images,)
        y_pred: numpy array of shape (num_images,)
        parameters: dictionary with the following keys:
            class_result: for binary classes, name of class for which metric will be calculated
            average: for mulitlcass, how to calculate metrics for each class: 'micro', 'macro', 'weighted' 
    
    Returns:
        score: float
        config:

    Additional Notes:
        recall = TP/ TP+FN

        average:
            'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives
            'macro': Calculate metrics for each label, and find their unweighted mean
            'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label)

    """

    config = {}

    if len(np.unique(y_true)) <= 2 and len(np.unique(y_pred)) <= 2:

        if "class_result" in parameters.keys():
            pos_label = parameters["class_result"]
        else:
           ytl = np.unique(y_true)
           ypl = np.unique(y_pred)
           labels = np.unique(np.concatenate((ytl, ypl))) 
           pos_label = labels[0]
        
        config["class_result"] = pos_label

        score = recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    
    else:
        
        if "average" in parameters.keys():
            average = parameters["average"]
        else:
            average = "weighted"

        config["average"] = average

        score = recall_score(y_true=y_true, y_pred=y_pred, average=average)

    return score, config



if __name__ == "__main__":

    #binary
    #y_true = np.array(["aa", "a", "ab", "b", "ac", "a"]).reshape(-1,2)
    #y_pred = np.array(["aa", "b", "ab", "b", "ac", "b"]).reshape(-1,2)
    
    
     
    # multi
    y_true = np.array(["aa", "a", "ab", "b", "ac", "c"]).reshape(-1,2)
    y_pred = np.array(["aa", "c", "ab", "b", "ac", "c"]).reshape(-1,2)
    
    
    pre={}
    pre["sensitivity"] = {}
    pre["sensitivity"]["function"] =0 #name of functions to be used for ensemblers
    pre["sensitivity"]["class_result"] = "b"
    pre["sensitivity"]["average"] = "micro"
    
    print("true")
    print(y_true[:,1])

    print("pred")
    print(y_pred[:,1])

    score, config = sensitivity(y_true=y_true[:,1], y_pred=y_pred[:,1], parameters=pre["sensitivity"])

    print(score)
    print(config)