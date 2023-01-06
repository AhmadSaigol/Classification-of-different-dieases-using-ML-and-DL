"""
Generates and trains a neural network

Note: works with apply_nn_classifiers only
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from collections import OrderedDict

#import label encoder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from process_labels import label_encoder
from metrics.accuracy import accuracy
from metrics.mcc import mcc



def mlp(X, y, parameters, classes, name_of_file, save_model_path=None, path_to_model=None, valid_data=None):
    """
        Multi layer percepton

        Parameters:
            X: numpy array of shape (num_images, num_features)
            parameters: dictionary containing the keys:
                hidden_layers: list containing number of neurons in each hidden layer
                alpha: parameter of activation 'LReLU' (default = 0.01)
                batch_size: how many samples to use for gradient update (default = 24)
                epochs: how many times to go through the data default = 10
                betas: for Adam optimizer (default = (0.9,0.999))
                lmbda: L2 regularization term (default = 0)
                lr: learning rate (default = 1e-3)
                use_single_neuron: (default=False) for binary classification, one can have one or two neurons in the output layer. 
                use_weighted_loss: (default=False) whether to use weighted loss function or not. For more info, see notes.

            classes: numpy array with names of classes
            y: labels of the data. numpy array of shape (num_of_images, 2) or (num_of_images, 1)
            vald_data: [X_valid, y_valid] , required in training phase, will be used for evaluating models
            save_model_path: (optional) where to save the trained model
            path_to_model: (optional) path to the foler where trained model is saved
            
        
        Returns:
            When training:
                y_pred_train: numpy array of shape(num_images, 2)
                y_pred_valid: numpy array of shape(num_images, 2)
                eval_metric_scores: numpy array of shape (3, 2, epochs):
                    axis=0 -> metrics (CEloss, balanced_accuracy and mcc) 
                    axis=1 -> training and validation resuls
                config: dictionary with parameters of the function (including default parameters)
            
            When testing:
                y_pred: numpy array of shape(num_images, 2)
                config: dictionary with parameters of the function (including default parameters)
        
        Additional Notes:
            if save_model_path is provided, then program will train the model and generate predicitons (training phase)
            if path_to_model is provided, then program will load the model and generate the prediction (testing phase)
            
            example of hidden layers: [2,3] -> first hidden layer has 2 neurons and second hidden layer has 3 neurons. Does not include input and output layer
            Number of neurons in input layer is determined by last axis of X
            Number of neurons in output layer is determined by number of classes

            it makes use of LeakyReLU as activation function and softmax as activation of output layer
            
            saves the best model based on validation loss criteria

            When using weighted loss function, the weights for each class is as follows:
                Multiclass: 1 - (number of samples of class in training data/total number of samples)
                Binary: 
                    when using single neuron in output layer: number of samples in negative class(class at index 0 in 'classes')/ number of samples in positive class

            For more details on loss function, see
                https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
                https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
    """
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # set to 0 -> no info on console, 1 -> basic info, 2 -> detailed info.
    verbose =1

    # determine phase
    if  save_model_path and not path_to_model and valid_data:
        train = True
    elif not save_model_path and path_to_model and not valid_data:
        train = False
    else:
        raise ValueError("Unable to determine the phase. Check parameters 'valid_data', 'save_model_path' and 'path_to_model'. ")

    output_config = {}

    if train:
        # obtain parameters
        nn_keys = parameters.keys()
        
        if "hidden_layers" in nn_keys:
            hidden_layers = parameters["hidden_layers"]
        else:
            raise ValueError("'hidden_layers' must be provided in parameters for MLP")
        
        output_config["hidden_layers"] = hidden_layers
        
        if "alpha" in nn_keys:
            alpha = parameters["alpha"]
        else:
            alpha = 0.01

        output_config["alpha"] = alpha


        if "batch_size" in nn_keys:
            batch_size = parameters["batch_size"]
        else:
            batch_size = 24

        output_config["batch_size"] = batch_size


        if "epochs" in nn_keys:
            epochs = parameters["epochs"]
        else:
            epochs = 10

        output_config["epochs"] = epochs


        if "betas" in nn_keys:
            betas = parameters["betas"]
        else:
            betas = (0.9,0.999)
        
        output_config["betas"] = betas


        if "lmbda" in nn_keys:
            lmbda = parameters["lmbda"]
        else:
            lmbda = 0
        output_config["lmbda"] = lmbda

        if "lr" in nn_keys:
            lr = parameters["lr"]
        else: 
            lr = 1e-3
        output_config["lr"] = lr

        if "use_single_neuron" in nn_keys:
            usn = parameters["use_single_neuron"]
        else:
            usn = False
        output_config["use_single_neuron"] = usn

        if "use_weighted_loss" in nn_keys:
            use_weighted_loss = parameters["use_weighted_loss"]
        else:
            use_weighted_loss = False
        output_config["use_weighted_loss"] = use_weighted_loss


        save_model_path = os.path.join(save_model_path, name_of_file+'.pt')
        
        eval_metrics_scores = np.zeros((3, 2, epochs))
        num_input_neurons = X.shape[-1]
        
        
        #encode labels 
        y_train = label_encoder(y=y[:,1], classes=classes, to_numbers=True)
        y_valid = label_encoder(y=valid_data[1][:,1], classes=classes, to_numbers=True)


        # calculate weights for each class
        if use_weighted_loss:
            print("Using weighted loss function for training the model")

            _, counts = np.unique(y_train, return_counts=True)

            if usn:
                pos_weight = torch.tensor([counts[0]/counts[1]])
                output_config["pos_weight"] = pos_weight.tolist()
            else: 
                class_weights = torch.tensor(1 - ( counts / sum(counts) ), dtype=torch.float32)
                output_config["class_weights"] = class_weights.tolist()
                
        
        # determine number of output neurons and loss function
        if usn and len(classes) != 2:
            raise ValueError ("it is not possible to use single neuron in output layer for multiclass classificaiton.")
        
        if usn:
            num_output_neurons = 1
            sig = nn.Sigmoid()
            if use_weighted_loss:
                loss_fnt = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fnt = nn.BCEWithLogitsLoss()
        else:
            num_output_neurons = len(classes)
            if use_weighted_loss:
                loss_fnt= nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fnt= nn.CrossEntropyLoss()


          
                
        #validation_data = Data(valid_data[0], y_valid)

        X_train = torch.from_numpy(X).type(torch.float32)
        y_train = torch.from_numpy(y_train)
        X_valid = torch.from_numpy(valid_data[0]).type(torch.float32)
        y_valid = torch.from_numpy(y_valid)

        if usn:
            y_train = torch.unsqueeze(y_train, 1).float()
            y_valid = torch.unsqueeze(y_valid, 1).float()

        training_data = Data(X_train, y_train)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        #valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False) 

        # set up model    
        model = Model(num_input_neurons, hidden_layers, num_output_neurons, alpha)
        
        if verbose >=1:
            print(model)

        
        # set up optimizer
        optimizer = torch.optim.Adam(model.parameters(),  lr=lr, betas=betas, amsgrad=True, weight_decay=lmbda)

        for ep in range(epochs):

            # train the model
            iter=0
            for X_train_batch, y_train_batch in train_dataloader:
                
                # get prediction on training data
                y_pred_batch = model(X_train_batch)
 
                # calculate loss
                loss = loss_fnt(y_pred_batch, y_train_batch)
                
                
                if verbose >=2:
                    print(f'Training: Epoch:{ep+1}/{epochs} Iter: {iter} Loss: {loss.item()}')
                
                # reset gradients
                optimizer.zero_grad()
                
                # calculate gradients
                loss.backward()

                # update weights
                optimizer.step()

                iter = iter + 1

            # eval the model
            with torch.no_grad():
                
                # get prediction on training data
                y_train_pred_ep = model(X_train)
                
                if usn:
                    y_train_pred_ep_temp = torch.round(sig(y_train_pred_ep.detach().clone())).numpy()
                else:
                    y_train_pred_ep_temp = np.argmax(y_train_pred_ep.detach().clone().numpy(), axis=1)

                #Calculate CELoss, balanced accuracy on training prediciton
                eval_metrics_scores[0,0,ep] = loss_fnt(y_train_pred_ep, y_train).detach().clone().numpy()
                eval_metrics_scores[1,0,ep] = accuracy(y_true=y_train.numpy(), y_pred=y_train_pred_ep_temp, parameters={"type": "balanced"})[0]
                eval_metrics_scores[2,0,ep] = mcc(y_true=y_train.numpy(), y_pred=y_train_pred_ep_temp, parameters={})[0]

                # get prediction on validaton data
                y_valid_pred_ep = model(X_valid)
                if usn:
                    y_valid_pred_ep_temp = torch.round(sig(y_valid_pred_ep.detach().clone())).numpy()
                else:    
                    y_valid_pred_ep_temp = np.argmax(y_valid_pred_ep.detach().clone().numpy(), axis=1)

                eval_metrics_scores[0,1,ep] = loss_fnt(y_valid_pred_ep, y_valid).detach().clone().numpy()
                eval_metrics_scores[1,1,ep] = accuracy(y_true=y_valid.numpy(), y_pred=y_valid_pred_ep_temp, parameters={"type": "balanced"})[0]
                eval_metrics_scores[2,1,ep] = mcc(y_true=y_valid.numpy(), y_pred=y_valid_pred_ep_temp, parameters={})[0]

                if verbose >=1:
                    print(f'Evaluating: Epoch:{ep+1}/{epochs} Training: Loss: {eval_metrics_scores[0,0,ep]}, Bal_accu: {eval_metrics_scores[1,0,ep]}, mcc: {eval_metrics_scores[2,0,ep]} Validation: Loss: {eval_metrics_scores[0,1,ep]}, Bal_accu: {eval_metrics_scores[1,1,ep]}, mcc: {eval_metrics_scores[2,1,ep]}')
            
            
            # save best model
            if ep == 0:
                if verbose >=2:
                    print(f"Saving Model of first epoch to {save_model_path}")
                
                torch.save(model, save_model_path)
                
                # save model results
                best_model_dict = {}
                best_model_dict["epoch"] = ep+1

                best_model_dict["train"] = {}
                best_model_dict["train"]["cross_entropy_loss"] = eval_metrics_scores[0,0,ep]
                best_model_dict["train"]["balanced_accuracy"] = eval_metrics_scores[1,0,ep]
                best_model_dict["train"]["mcc"] = eval_metrics_scores[2,0,ep]
                
                best_model_dict["valid"] = {}
                best_model_dict["valid"]["cross_entropy_loss"] = eval_metrics_scores[0,1,ep]
                best_model_dict["valid"]["balanced_accuracy"] = eval_metrics_scores[1,1,ep]
                best_model_dict["valid"]["mcc"] = eval_metrics_scores[2,1,ep]
                
                
            elif  eval_metrics_scores[0,1,ep] < best_model_dict["valid"]["cross_entropy_loss"]:
                
                if verbose >=2:
                    print(f"Validation Loss score improved. Saving Model to {save_model_path}")
                
                torch.save(model, save_model_path)
                
                # save model results
                best_model_dict["epoch"] = ep+1

                best_model_dict["train"]["cross_entropy_loss"] = eval_metrics_scores[0,0,ep]
                best_model_dict["train"]["balanced_accuracy"] = eval_metrics_scores[1,0,ep]
                best_model_dict["train"]["mcc"] = eval_metrics_scores[2,0,ep]
                
                best_model_dict["valid"]["cross_entropy_loss"] = eval_metrics_scores[0,1,ep]
                best_model_dict["valid"]["balanced_accuracy"] = eval_metrics_scores[1,1,ep]
                best_model_dict["valid"]["mcc"] = eval_metrics_scores[2,1,ep]
                
            else:

                if verbose >=2:
                    print("Validaiton Loss score did not improve.")

        output_config["best_model"] = best_model_dict
        
        if verbose >=1:
            print(f"Best Validation Loss score recorded on epoch {best_model_dict['epoch']}: {best_model_dict['valid']['cross_entropy_loss']}")

        # generate prediction using the best model
        if verbose >=1:
            print(f"Loading best model from {save_model_path} to generate predictions")
        best_model = torch.load(save_model_path)
        best_model.eval()
        
        with torch.no_grad():
            y_pred_train = best_model(X_train)
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
    
    else:
        output_config = parameters

        X = torch.from_numpy(X).type(torch.float32)
        
        # generate prediction using the best model
        if verbose >=1:
            print(f"Loading best model from {path_to_model} to generate predictions")
        best_model = torch.load(os.path.join(path_to_model, name_of_file+".pt"))
        best_model.eval()

        
        with torch.no_grad():
            y_pred = best_model(X)
            
            if y_pred.shape[-1] == 1:
                sig = torch.nn.Sigmoid()
                y_pred =  torch.round(sig(y_pred.detach().clone())).numpy()
            else:
                y_pred = np.argmax(y_pred.detach().clone().numpy(), axis=1)
            
        y_pred = label_encoder(y_pred, classes=classes, to_numbers=False)
        y_pred = np.concatenate((y[:,0,None], y_pred[:,None]), axis=-1)

        return y_pred, output_config


class Data(Dataset):
    """
    Set up dataset for working with Pytorch
    """
    def __init__(self, X, y):
       self.X = X
       self.y = y
       self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len




class Model(nn.Module):
    """
    Set up Neural Network using Pytorch

    """

    def __init__(self, num_inputs, hidden_layers, num_outputs, alpha):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.seq_model = nn.Sequential(self.get_model())

    def forward(self, x):
        return self.seq_model(x)
    
    def get_model(self):
        
        neuron_block = []
        num_hidden_layers = len(self.hidden_layers)
        for i in range(num_hidden_layers+1):
            
            # input layer connection
            if i == 0:
                neuron_block.append((f'linear{i}', nn.Linear(self.num_inputs, self.hidden_layers[i])))
                neuron_block.append((f'lrelu{i}', nn.LeakyReLU(self.alpha)))
            
            # output layer connection
            elif i == num_hidden_layers:
                neuron_block.append((f'linear{i}', nn.Linear(self.hidden_layers[i-1], self.num_outputs)))

            # hidden layers connection    
            else:
                neuron_block.append((f'linear{i}', nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i])))
                neuron_block.append((f'lrelu{i}', nn.LeakyReLU(self.alpha)))

        return OrderedDict(neuron_block)


if __name__ == "__main__":
    X = np.arange(24, dtype=np.float32).reshape(-1, 3)
    X_valid = X[5:]
    X = X[0:5]

    y = np.array(["aa", "b", "ac", "c", "da","b",
                    "cc","b", "dc", "a"]).reshape(-1,2)
    
    y_valid = np.array(["op", "a", "de", "b", "da","b"]).reshape(-1,2)            

    classes =np.array(["b", "a", "c"])


    network = {}
    network["NN1"] = {}
    network["NN1"]["hidden_layers"] = [5, 4]
    network["NN1"]["alpha"] = 0.2
    network["NN1"]["batch_size"] = 2
    network["NN1"]["lmbda"] = 0.1
    network["NN1"]["lr"] = 0.1
    network["NN1"]["epochs"] = 10
    network["NN1"]["use_weighted_loss"] = True

    
                
    save_model_path = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/results/Phase2_results/code_test"
    
    valid_data = [X_valid, y_valid]

    print("X_train ", X)
    print("y_train ", y)
    print("X_valid ", X_valid)
    print("y_valid ", y_valid)
    y_pred_train, y_pred_valid, eval_metrics_scores ,output_config = mlp (X=X, y=y, parameters=network["NN1"],  name_of_file="test_loss", classes=classes, save_model_path=save_model_path, valid_data=valid_data)

    print("y_pred_train ", y_pred_train)
    print("y_pred_valid ", y_pred_valid)
    print("Eval_scores ", eval_metrics_scores)
    print("config ", output_config)

    print("testing")
    X_test = np.random.rand(6,2).astype(np.float32)
    y_test= np.array(["aa","ac", "da","cc", "dc", "da"]).reshape(-1,1)    

    print("X_test ", X_test)
    print("y_test ", y_test)

    y_pred, output_config = mlp(X, y, parameters=output_config, classes=classes, path_to_model=save_model_path, name_of_file="test_loss")
    print("y_pred", y_pred)
    print("config ", output_config)
