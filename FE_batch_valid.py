from utils.data_preprocessing.split_data import split_data
from utils.json_processing import save_to_json, load_from_json
import numpy  as np
import torch

output_config ={}
output_config["data"] = {}
output_config["data_preprocessing"] = {}

path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/train_multi.txt"
y = np.loadtxt(path_to_labels, dtype=str, delimiter=" ")

path_to_json = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/basic_extractors_26-01/binary/train/training_pipeline.json"

path_to_model = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/documents/presentation/phase2/NN/online aug/run_03_weighted_loss_no_reg_data_aug_online/multiclass/train/0/models/NN2.pt"

model = torch.load(path_to_model)
model.eval()

pipeline = load_from_json(path_to_json)

split_type = "simpleStratified"
test_size = 0.3


y_train, y_valid = split_data(y=y, split_type=split_type, test_size=test_size)

 with torch.no_grad():
        y_pred_train = model(X_train)
        if usn:
            y_pred_train =  torch.round(sig(y_pred_train.detach().clone())).numpy()
        else:
            y_pred_train = np.argmax(y_pred_train.detach().clone().numpy(), axis=1)

        y_pred_valid = best_model(X_valid)
        if usn:
            y_pred_valid =  torch.round(sig(y_pred_valid.detach().clone())).numpy()
        else:
            y_pred_valid = np.argmax(y_pred_valid.detach().clone().numpy(), axis=1)

    y_pred_train = label_encoder(y_pred_train, classes=classes, to_numbers=False)
    y_pred_train = np.concatenate((y[:,0,None], y_pred_train[:,None]), axis=-1)

    y_pred_valid = label_encoder(y_pred_valid, classes=classes, to_numbers=False)
    y_pred_valid = np.concatenate((valid_data[1][:,0,None], y_pred_valid[:,None]), axis=-1)

    return y_pred_train, y_pred_valid, eval_metrics_scores ,output_config
 