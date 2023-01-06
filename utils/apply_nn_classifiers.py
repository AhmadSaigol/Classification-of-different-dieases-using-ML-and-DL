"""
Applies nn classifiers and generate predictions

"""
import os
import numpy as np

def apply_nn_classifiers(X,y, networks, classes, path_to_results=None, valid_data=None, return_probs=None):
    """
    Applies each NN classifier to the input and generates prediction

    Parameters:
        X: numpy array of shape (folds, num_images, num_features)
        classes: numpy array of names of classes
        networks: dictionary with following structure:
            networks["NN1"]["parameter_1"] = value
            networks["NN1"]["parameter_2"] = value

            networks["NN2"]["parameter_1"] = value
            networks["NN2"]["parameter_2"] = value

        y: labels of the data. numpy array of shape (folds, num_of_images, 2) or (folds,num_images, 1)
        path_to_results: where the models will be saved ( if not provided, the code will work in testing phase)
        vald_data: [X_valid, y_valid] , required in training phase, will be used for evaluating models

    Returns:
        When training:
            y_preds_train: (networks, folds, num_images, 2)
            y_preds_valid: (networks, folds, num_images, 2)
            eval_metric_scores: list of numpy arrays with each having shape of(folds, metrics, 2, epochs). Each item in list represents results of a network.
            output_config: 
            list_of_networks:since data is incoming in dictionary format, so for easy mapping of which index represents to which classifier
        When testing:
            y_preds: (networks, folds, num_images, 2)    
            output_config:
            list_of_networks: since data is incoming in dictionary format, so for easy mapping of which index represents to which classifier

        
    Additional Notes:

        currently the parameters in the output dictionary will be only of first fold for each classifier


    """
    from classifiers.MLP import mlp

    
    output_config = {}

    num_folds = X.shape[0]
    
    flag = True
    
    # determine whether we are in training phase or testing phase
    if path_to_results and valid_data: 
        train = True
    elif not path_to_results and not valid_data:
        train = False
    else:
        raise ValueError("Unable to determine the phase in which the function is being called")
    
    # create directory for saving results
    if train:   
        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)
            print(f"Created directory {path_to_results}")
        else:
            print(f"Warning: {path_to_results} already exists. Contents may get overwritten")


    list_of_networks = []
    eval_metric_scores = []

    for network in networks.keys():
        
        output_config[network] ={}

        if train:
            output_config[network]["path_to_models"] = {}
        

        for fold_no in range(num_folds):
            
            print(f"Applying Network: {network} Fold No: {fold_no}")
            
            
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

                output_config[network]["path_to_models"][fn] = save_model_path
                
                prediction_train, prediction_valid, metric_scores, fnt_config = mlp(X=X[fold_no], parameters=networks[network], 
                                                    classes=classes, name_of_file =network, y=y[fold_no], valid_data = [valid_data[0][fold_no], valid_data[1][fold_no]], save_model_path=save_model_path)


                prediction_train = np.expand_dims(prediction_train, axis=0)
                prediction_valid = np.expand_dims(prediction_valid, axis=0)
                metric_scores = np.expand_dims(metric_scores,axis=0)

                if fold_no==0:
                    output_config[network] = {**output_config[network], **fnt_config}
                    y_pred_train = prediction_train
                    y_pred_valid = prediction_valid
                    eval_metric_score = metric_scores
                else:
                    y_pred_train = np.concatenate((y_pred_train, prediction_train))
                    y_pred_valid = np.concatenate((y_pred_valid, prediction_valid))
                    eval_metric_score = np.concatenate((eval_metric_score, metric_scores))

            
            else:
                print("In Testing Phase ")
                path_to_model =networks[network]["path_to_models"][fn]
                
                prediction, fnt_config = mlp(X=X[fold_no], parameters=networks[network], 
                                                    classes=classes, y=y[fold_no], name_of_file =network, path_to_model=path_to_model)
               
              
                prediction = np.expand_dims(prediction, axis=0)
            
                if fold_no==0:
                    output_config[network] = {**output_config[network], **fnt_config}
                    y_pred = prediction
                else:
                    y_pred = np.concatenate((y_pred, prediction))
                    
        
        # concatenate y_pred for networks             
        if train:
            y_pred_train = np.expand_dims(y_pred_train, axis=0)
            y_pred_valid = np.expand_dims(y_pred_valid, axis=0)
            eval_metric_scores.append(eval_metric_score)

            if flag:
                y_preds_train = y_pred_train
                y_preds_valid = y_pred_valid
                flag=False
            else:
                y_preds_train = np.concatenate((y_preds_train, y_pred_train)) 
                y_preds_valid = np.concatenate((y_preds_valid, y_pred_valid))
            
        else:
            y_pred = np.expand_dims(y_pred, axis=0)
        
            if flag:
                y_preds = y_pred
                flag=False
            else:
                y_preds = np.concatenate((y_preds, y_pred)) 
                    
        # save network
        list_of_networks.append(network)

    if train:
        y_preds_train_probs = None
        y_preds_valid_probs = None
        return y_preds_train, y_preds_valid, y_preds_train_probs, y_preds_valid_probs, eval_metric_scores, output_config, list_of_networks
    else:
        y_preds_probs = None
        return y_preds, y_preds_probs, output_config, list_of_networks


if __name__ == "__main__":
    
    print("Training ")

    X_train = np.random.rand(2,3,5)
    X_valid = np.random.rand(2,2,5)

    #print("X_train ", X_train)
    print("X_train shape ", X_train.shape)

    y_train = np.array([  "aa", "b", "ac", "a", "da","b",
                    "cc","b", "dc", "a", "da", "b"]).reshape(2,3,2)
    y_valid = np.array([  "aa", "b","cc","b", 
                        "dc", "a", "da", "b"]).reshape(2,2,2)

    #print("y_train ", y_train)
    print("y_train shape ", y_train.shape)

    #print("X_valid ", X_valid)
    print("X_valid shape ", X_valid.shape)

    #print("y_valid ", y_valid)
    print("y_valid shape ", y_valid.shape)


    classes = np.array(["a", "b"])
    
    pipeline = {}

    pipeline["classifiers"] ={}

    # NN1
    pipeline["classifiers"]["NN1"] = {}
    pipeline["classifiers"]["NN1"]["hidden_layers"] = [5]
    pipeline["classifiers"]["NN1"]["alpha"] = 0.2
    pipeline["classifiers"]["NN1"]["batch_size"] = 2
    pipeline["classifiers"]["NN1"]["lmbda"] = 0.1
    pipeline["classifiers"]["NN1"]["lr"] = 0.1
    pipeline["classifiers"]["NN1"]["epochs"] = 10
    pipeline["classifiers"]["NN1"]["use_single_neuron"] = True
    # NN2
    #pipeline["classifiers"]["NN2"] = {}
    #pipeline["classifiers"]["NN2"]["hidden_layers"] = [5]
    #pipeline["classifiers"]["NN2"]["alpha"] = 0.2
    #pipeline["classifiers"]["NN2"]["batch_size"] = 2
    #pipeline["classifiers"]["NN2"]["lmbda"] = 0.1
    #pipeline["classifiers"]["NN2"]["lr"] = 0.1
    #pipeline["classifiers"]["NN2"]["epochs"] = 10

    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/code_test"

    y_train_pred, y_valid_pred, metric_score, config, networks = apply_nn_classifiers(X=X_train, y=y_train, classes=classes, path_to_results=path_to_results, networks=pipeline["classifiers"], valid_data=[X_valid, y_valid])

    #print("train pred ", y_train_pred)
    print("train pred shape ", y_train_pred.shape)

    #print("valid pred ", y_train_pred)
    print("valid data shape ", y_train_pred.shape)

    #print ("networks: ", networks)

    #print("config ", config)
    #-------------------------------------------------------

    print("Testing")


    X_test = np.random.rand(1,2,5)
    #print("X_test ", X_test)
    print("X_test shape ", X_test.shape)

    y_test = np.array([ "test", "new_test" ]).reshape(1,2,1)
    #print("y_test ", y_test)
    print("y_test shape ", y_test.shape)

    test_pred, test_config, test_networks = apply_nn_classifiers (X=X_test, networks=config, classes=classes, y=y_test)


    #print("test pred ", test_pred)
    print("test pred shape ", test_pred.shape)

    #print ("test classifiers: ", test_networks)

    #print("test config ", test_config)