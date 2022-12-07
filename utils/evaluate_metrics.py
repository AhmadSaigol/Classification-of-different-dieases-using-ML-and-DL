"""
Applies each evaluation metrics to generate evaluation score

"""

import numpy as np
from metrics.accuracy import accuracy
from metrics.f1_score import F1_score
from metrics.mcc import mcc
from metrics.precision import precision
from metrics.sensitivity import sensitivity

def evaluate_metrics(y_true, y_pred, metrics):
    """
    Applies each metric and generates evaluation score

    Parameters:
        y_true: numpy array of shape (folds, num_images, 2)
        y_pred: numpy array of shape (classifiers, folds, num_images, 2)
        metrics: dictionary with following structure:
            metrics["metrics_1"]["function"] = pointer to the function
            metrics["metrics_1"]["parameter_1"] = value
            metrics["metrics_1"]["parameter_2"] = value

            metrics["metrics_2"]["function"] = pointer to the function
            metrics["metrics_2"]["parameter_1"] = value
            metrics["metrics_2"]["parameter_2"] = value

        classes: numpy array consisting names of classes

    Returns:
        scores:numpy array of shape (classifiers, folds, metrics)
        output_config:
        list_of_metrics: since data is incoming in dictionary format, so for easy mapping of which index represents to which metric
    
    Additional Notes:

    """
    # check whether correct shapes of y_* are provided or not

    if len(y_true.shape) !=3:
        raise ValueError("Shape of y_true is not correct.")

    if len(y_pred.shape) == 4:
        num_classifiers = y_pred.shape[0]
        num_folds = y_pred.shape[1]
        met_keys = list(metrics.keys())
        num_metrics = len(met_keys)
    else:
        raise ValueError("Shape of y_pred is not correct.")

    config={}

    list_of_metrics = []

    scores = np.full((num_classifiers, num_folds, num_metrics), 1000, dtype=np.float32)
    
    flag = True

    for cl_no in range(num_classifiers):

        for fold_no in range(num_folds):

            for metric_no in range(num_metrics):

                print(f"Procesing Classifer No: {cl_no} Fold No: {fold_no} Metric: {met_keys[metric_no]}")

                fnt_pointer = metrics[ met_keys[metric_no] ]["function"]

                
                metric_score, fnt_config = fnt_pointer(y_true=y_true[fold_no, :, 1], 
                                                        y_pred=y_pred[cl_no, fold_no, :, 1], 
                                                        parameters=metrics[met_keys[metric_no]])

                
                scores[cl_no, fold_no, metric_no] = metric_score

                # setup output config
                if flag:
                    fnt_config["function"] = fnt_pointer
                    config[met_keys[metric_no]] = {} 
                    config[met_keys[metric_no]] = fnt_config
                    
                    list_of_metrics.append(met_keys[metric_no])

            flag=False
    
    return scores, config, list_of_metrics


if __name__ == "__main__":
    """
    # binary
    y_true = np.array(["aa", "a", "ab", "b", "ac", "a",
                        "ad", "a", "ae", "b", "af", "a",
                        "ag", "a", "ah", "b", "ai", "a"]).reshape(3,3, 2)
    y_pred = np.array(["aa", "b", "ab", "b", "ac", "b",
                        "ad", "a", "ae", "a", "af", "a",
                        "ag", "a", "ah", "b", "ai", "a",
                        "aj", "b", "ak", "b", "al", "b",
                        "am", "a", "an", "a", "ao", "a",
                        "ap", "a", "aq", "a", "ar", "b"]).reshape(2,3,3,2)
    """
    
     
    # multi
    y_true = np.array(["aa", "a", "ab", "b", "ac", "c",
                        "ad", "a", "ae", "c", "af", "a",
                        "ag", "c", "ah", "b", "ai", "a"]).reshape(3,3, 2)
    y_pred = np.array(["aa", "b", "ab", "b", "ac", "b",
                        "ad", "a", "ae", "c", "af", "a",
                        "ag", "a", "ah", "b", "ai", "c",
                        "aj", "b", "ak", "c", "al", "b",
                        "am", "c", "an", "a", "ao", "a",
                        "ap", "a", "aq", "c", "ar", "b"]).reshape(2,3,3,2)
    
    
    pipeline = {}

    pipeline["metrics"] = {}

    # accuracy
    pipeline["metrics"]["simple_accuracy"] = {}
    pipeline["metrics"]["simple_accuracy"]["function"] = accuracy
    pipeline["metrics"]["simple_accuracy"]["type"] = "simple"  

    pipeline["metrics"]["balanced_accuracy"] = {}
    pipeline["metrics"]["balanced_accuracy"]["function"] = accuracy
    pipeline["metrics"]["balanced_accuracy"]["type"] = "balanced"


    # precision
    pipeline["metrics"]["precision"] = {}
    pipeline["metrics"]["precision"]["function"] = precision 
    pipeline["metrics"]["precision"]["class_result"] = "a"
    pipeline["metrics"]["precision"]["average"] = "weighted"


    # recall
    pipeline["metrics"]["sensitivity"] = {}
    pipeline["metrics"]["sensitivity"]["function"] = sensitivity
    pipeline["metrics"]["sensitivity"]["class_result"]  = "a"
    pipeline["metrics"]["sensitivity"]["average"]  = "weighted"

    # f1_score
    pipeline["metrics"]["f1_score"] = {}
    pipeline["metrics"]["f1_score"]["function"] = F1_score 
    pipeline["metrics"]["f1_score"]["class_result"] = "a"
    pipeline["metrics"]["f1_score"]["average"] = "weighted"

    # mcc
    pipeline["metrics"]["mcc"] = {}
    pipeline["metrics"]["mcc"]["function"] = mcc 
    

    print("true labels")
    print(y_true)
    print(y_true.shape)

    print("pred labels")
    print(y_pred)
    print(y_pred.shape)

    scores, config, list_of_metrics =evaluate_metrics(y_true=y_true, y_pred=y_pred, metrics =pipeline["metrics"])

    print("scores ")
    print(scores)

    print("config ")
    print(config)
    
    print("metrics list ")
    print(list_of_metrics)

    