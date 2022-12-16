"""
Creates plots
"""
import os
import numpy as np
#from plots.plot_CM import plot_CM

def create_plots(y_true, y_pred, path_to_results, path_to_images, plots, classifiers, name_of_file):
    """
    Generates differenet plots
    Parameters:
        y_true: numpy array of shape (folds, num_images, 2)
        y_pred: numoy array of shape (classifiers, folds, num_images, 2)
        path_to_results: path to the folder where the results will be stored
        plots: dictionary with following structure:
                plots["plots_1"]["function"] = pointer to the function
                plots["plots_1"]["parameter_1"] = value
                plots["plots_1"]["parameter_2"] = value

                plots["plots_2"]["function"] = pointer to the function
                plots["plots_2"]["parameter_1"] = value
                plots["plots_2"]["parameter_2"] = value
        classifiers: numpay array with numes of classifiers
        name_of_file:  

    Returns:
        output_config:

    Additional Notes:


    """

    # check whether correct shapes of y_* are provided or not

    if len(y_true.shape) !=3:
        raise ValueError("Shape of y_true is not correct.")

    if len(y_pred.shape) == 4:
        num_classifiers = y_pred.shape[0]
        num_folds = y_pred.shape[1]
    else:
        raise ValueError("Shape of y_pred is not correct.")


    for cl in range(num_classifiers):

        for fold_no in range(num_folds):

            # create directory for fold
            path_to_fold = os.path.join(path_to_results, str(fold_no))
            if not os.path.exists(path_to_fold):
                os.mkdir(path_to_fold)
                
            # create directory for plots
            path_to_plots = os.path.join(path_to_fold, "plots")
            if not os.path.exists(path_to_plots):
                os.mkdir(path_to_plots)

            for pl in plots.keys():

                print(f"Creating Plot: {pl} Classifier: {classifiers[cl]} Fold No: {fold_no}")

                fnt_pointer = plots[pl]["function"]

                fnt_pointer(y_true=y_true[fold_no], y_pred=y_pred[cl, fold_no], path_to_results=path_to_plots+f"/{classifiers[cl]}_{name_of_file}")


    return plots


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

    pipeline["plots"] = {}

    # accuracy
    pipeline["plots"]["CM"] = {}
    pipeline["plots"]["CM"]["function"] = plot_CM

    config = create_plots(
        y_true=y_true,
        y_pred=y_pred,
        plots= pipeline["plots"],
        path_to_results="/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/new_code_testing",
        classifiers=["a", "b"]
    )                 

    print(config)


    