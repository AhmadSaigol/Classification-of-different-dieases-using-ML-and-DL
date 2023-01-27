
"""
Performs binary and multiclass classificaiton at the same time

"""


# import libraries
import numpy as np
import os
from copy import deepcopy

from utils.misc import replace_function_names
from utils.normalize_features import normalize_features

from utils.apply_classifiers import apply_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree
from utils.classifiers.boosting import boosting

from utils.evaluate_metrics import evaluate_metrics
from utils.metrics.accuracy import accuracy
from utils.metrics.f1_score import F1_score
from utils.metrics.mcc import mcc
from utils.metrics.precision import precision
from utils.metrics.sensitivity import sensitivity

from utils.create_plots import create_plots
from utils.plots.plot_CM import plot_CM
from utils.plots.plot_misclassified_samples import plot_MS

from utils.json_processing import save_to_json, load_from_json

from utils.misc import add_function_names, generate_txt_file, save_results, change_txt_for_binary
from utils.data_preprocessing.change_colorspace import change_colorspace
from utils.data_preprocessing.resize import resize
from utils.data_preprocessing.normalize import normalize
from utils.data_preprocessing.edge_detector import canny_edge_detector

import time
import datetime
from NN_batch_prediction import NN_batch_prediction

from utils.feature_extractors.contrast import calculate_contrast
from utils.feature_extractors.kurtosis import calculate_kurtosis
from utils.feature_extractors.skewness import calculate_skew
from utils.feature_extractors.histogram import calculate_histogram
from utils.feature_extractors.haralick import calculate_haralick
from utils.feature_extractors.zernike import calculate_zernike
from utils.feature_extractors.non_zero_valules import count_nonzeros
from utils.feature_extractors.local_binary_pattern import calculate_lbp
from utils.feature_extractors.wavelet import feature_GLCM


from utils.normalize_features import normalize_features

from utils.apply_classifiers import apply_classifiers
from utils.classifiers.SVM import svm
from utils.classifiers.RFTree import rftree
from utils.classifiers.boosting import boosting


# noisy dataset
path_to_noisy_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/final"
path_to_noisy_data_labels_multi = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/data_augmented.txt"
path_to_noisy_data_labels_binary = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/data_augmented_binary.txt"


y_noisy_binary = np.loadtxt(path_to_noisy_data_labels_binary, dtype=str, delimiter=" ")
y_noisy_binary = np.expand_dims(y_noisy_binary, axis=0)

y_noisy_multi = np.loadtxt(path_to_noisy_data_labels_multi, dtype=str, delimiter=" ")
y_noisy_multi = np.expand_dims(y_noisy_multi, axis=0)

# results
path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/documents/presentation"

phase = "phase2"
model_name = "NN_test"
type_aug= "offlien aug"
folder_name ="run_03_weighted_loss_no_reg_data_aug_offline"

json_name = "train/training_pipeline.json"


# binary
print("Processing Binary Classification")

path_to_json = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "binary", json_name)
save_path = os.path.join(path_to_results, phase,  model_name, type_aug, folder_name, "binary", "noisy_data-21-01")

# change json
print("Processing json . . .")
pb = load_from_json(path_to_json)
#pb["device"] = "cpu"
pb["path_to_results"] = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "binary", "train")
pb["batch_size"] = 32

path_to_models = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "binary", "train", "0", "models")
pb["networks"]["NN1"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN2"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN3"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN4"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN5"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN6"]["path_to_models"] ["00"]= path_to_models
save_to_json(pb, path_to_json[:path_to_json.rindex(".json")])

print("Generating predictions . . . ")
start = time.time()
NN_batch_prediction(path_to_noisy_data, path_to_json, save_path)
end = time.time()
print("Time taken to generate predictions on test data : ", datetime.timedelta(seconds=end-start))

y_pred_binary = []
for i in sorted(os.listdir(os.path.join(save_path, "0"))):
    temp = np.loadtxt(os.path.join(save_path, "0", i), dtype=str, delimiter=" ")
    y_pred_binary.append([temp])

y_pred_binary = np.array(y_pred_binary)
print("Loaded Labels txt: shape: ", y_pred_binary.shape)

#y_pred_binary = np.expand_dims(y_pred_binary, axis=0)
# replace function names with their pointers    
data_preprocessing_fnt = [normalize, change_colorspace, resize, canny_edge_detector]
feature_extractors_fnt= [calculate_contrast, calculate_kurtosis, calculate_skew, calculate_histogram, calculate_haralick, 
                    calculate_zernike,  count_nonzeros, calculate_lbp, feature_GLCM]
norm_features_fnt = [normalize_features]
classifiers_fnt = [svm, rftree, boosting]
metrics_fnt =[F1_score, precision, sensitivity, mcc, accuracy]
plots_fnt =[plot_CM, plot_MS]

replace_function_names(
    pb, 
    functions = data_preprocessing_fnt + feature_extractors_fnt + norm_features_fnt + classifiers_fnt + metrics_fnt + plots_fnt)

    
# setup for metrics
metrics = pb["metrics"]
print("\nEvaluating Metrics on data . . . ")
eval_score, _, metrics_list = evaluate_metrics(
    y_true=y_noisy_binary, 
    y_pred=y_pred_binary, 
    metrics=metrics,
    classifiers=list(pb["networks"].keys()),
    y_pred_probs = None
    )
print("\nResults")


for met_no, met in enumerate(metrics_list):
    print(f"Metric: {met}  Score: {np.around(eval_score[0, 0, met_no], 4)} ")

print(f"Evaluated Metrics on data successfully. Shape:{eval_score.shape}")

# setup plots
plots = pb["plots"]
print("\nCreating Plots for data . . .")
_ = create_plots(
    y_true=y_noisy_binary, 
    y_pred=y_pred_binary, 
    plots= plots, 
    path_to_results=save_path,
    path_to_images=path_to_noisy_data,
    name_of_file = "noisy_dataset",
    classifiers=list(pb["networks"].keys())
    )
print("Created Plots for the data")

print("\nSaving  results ")
save_results(
    results=eval_score,
    metrics=metrics_list,
    path_to_results=save_path,
    name_of_file="noisy_dataset",
    classifiers=list(pb["networks"].keys()),
)



# multi
print("Processing Multi Classification")

path_to_json = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "multi", json_name)
save_path = os.path.join(path_to_results, phase,  model_name, type_aug, folder_name, "multi", "noisy_data-21-01")

# change json
print("Processing json . . .")
pb = load_from_json(path_to_json)
#pb["device"] = "cpu"
pb["path_to_results"] = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "multi", "train")
pb["batch_size"] = 32

path_to_models = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "multi", "train", "0", "models")
pb["networks"]["NN1"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN2"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN3"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN4"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN5"]["path_to_models"] ["00"]= path_to_models
pb["networks"]["NN6"]["path_to_models"] ["00"]= path_to_models
save_to_json(pb, path_to_json[:path_to_json.rindex(".json")])

print("Generating predictions . . . ")
start = time.time()
NN_batch_prediction(path_to_noisy_data, path_to_json, save_path)
end = time.time()
print("Time taken to generate predictions on test data : ", datetime.timedelta(seconds=end-start))

y_pred_multi = []
for i in sorted(os.listdir(os.path.join(save_path, "0"))):
    temp = np.loadtxt(os.path.join(save_path, "0", i), dtype=str, delimiter=" ")
    y_pred_multi.append([temp])

y_pred_multi = np.array(y_pred_multi)
print("Loaded Labels txt: shape: ", y_pred_multi.shape)

#y_pred_binary = np.expand_dims(y_pred_binary, axis=0)
# replace function names with their pointers    


replace_function_names(
    pb, 
    functions = data_preprocessing_fnt + feature_extractors_fnt + norm_features_fnt + classifiers_fnt + metrics_fnt + plots_fnt)

    
# setup for metrics
metrics = pb["metrics"]
print("\nEvaluating Metrics on data . . . ")
eval_score, _, metrics_list = evaluate_metrics(
    y_true=y_noisy_multi, 
    y_pred=y_pred_multi, 
    metrics=metrics,
    classifiers=list(pb["networks"].keys()),
    y_pred_probs = None
    )
print("\nResults")


for met_no, met in enumerate(metrics_list):
    print(f"Metric: {met}  Score: {np.around(eval_score[0, 0, met_no], 4)} ")

print(f"Evaluated Metrics on data successfully. Shape:{eval_score.shape}")

# setup plots
plots = pb["plots"]
print("\nCreating Plots for data . . .")
_ = create_plots(
    y_true=y_noisy_multi, 
    y_pred=y_pred_multi, 
    plots= plots, 
    path_to_results=save_path,
    path_to_images=path_to_noisy_data,
    name_of_file = "noisy_dataset",
    classifiers=list(pb["networks"].keys())
    )
print("Created Plots for the data")

print("\nSaving  results ")
save_results(
    results=eval_score,
    metrics=metrics_list,
    path_to_results=save_path,
    name_of_file="noisy_dataset",
     classifiers=list(pb["networks"].keys())
)

print("Processing Completed")



