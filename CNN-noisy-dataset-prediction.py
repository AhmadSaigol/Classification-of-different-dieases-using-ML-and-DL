import numpy as np
import cv2 
import os

from collections import OrderedDict

import torch
from torch import nn
from torchvision.io import read_image,ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score

from matplotlib import pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.image as mpimg

import json

import pandas as pd

import time
import datetime


def label_encoder(y, classes, to_numbers):
    """
    encodes target labels with value between 0 and n_classes-1 and vice versa

    Parameter:
        y: labels of numpy array (num_images,)
        classes: numpy array of names of classes
        to_numbers True/False: whether to transfer from string to numbers or vice versa

    Returns:
        result: encoded labels of numpy array (num_images, 1)
                or labels of numpy array (num_images, 1)
    """

    le = preprocessing.LabelEncoder()
    
    if type(classes).__name__ == 'list':
        classes = np.array(classes)
    le.classes_ = classes
    
    if to_numbers:
        return le.transform(y.ravel())
    else:
        return le.inverse_transform(y.ravel().astype(int)) 


class ImageDataset(Dataset):
    """
    
    Parameters:
        labels: labels of shape(num_images, 2) or (num_images, 1)
        path_to_images: path to the folder containing the images
        color: whether to read image in "gray" or "rgb" (default="gray")
        transform: transform to be applied on a sample
        
        When combined with dataloader, it returns:
            image: tensor of shape (batch, 1, height, width)
            label: tensor of shape (batch)
    
    """
    def __init__(self, labels, path_to_images, color="gray", transform=None):
        self.img_labels = labels
        self.img_dir = path_to_images
        self.transform = transform
        
        if color =="gray":
            self.color = ImageReadMode.GRAY
        elif color =="rgb":
            self.color = ImageReadMode.RGB
        else:
            raise ValueError("unknown color passed to dataset")
        

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels[index, 0])
        
        # read image 
        image = read_image(img_path, self.color).float()
        
        if self.img_labels.shape[-1]==2:
            label = self.img_labels[index, 1].astype(int)
        else:
            label = -1.0 # just place holder for valid data
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

class Model(nn.Module):
    """
    Sets up the model
    layers: OrderedDict with format:
        layers["layer_1"] = {}
        layers["layer_1"]["name"] = value
        layers["layer_1"]["parameter1"] = value
        layers["layer_1"]["parameter2"] = value
        
        layers["layer_2"] = {}
        layers["layer_2"]["name"] = value
        layers["layer_2"]["parameter1"] = value
        layers["layer_2"]["parameter2"] = value
        
    currently supports layer names. "linear", "flatten", "relu", "lrelu"
    
    output layer must be added as well
    """
    def __init__(self, input_shape, layers, use_pretrained_model):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        if use_pretrained_model:
            self.seq_model = self.get_pretrained_model()
        else:
            self.seq_model = nn.Sequential(self.get_model())
    
    def forward(self, x):
        return self.seq_model(x)
    
    def get_model(self):
        
        arch = []
        self.layer = "Init"
        self.pre_layer_output = self.input_shape  
        
        for layer in self.layers.keys():
            self.layer = layer
            layer_name = self.layers[layer]["name"] 
            
            # Linear Layer 
            if layer_name == "linear":
                arch.append(self.add_linear(self.layers[layer]))
                
            # ReLU
            elif layer_name == "relu":
                arch.append(self.add_relu(self.layers[layer]))
                
            # LeakyReLU
            elif layer_name == "lrelu":
                arch.append(self.add_lrelu(self.layers[layer]))
            
            # flatten
            elif layer_name == "flatten":
                arch.append(self.add_flatten(self.layers[layer]))
                
            # conv
            elif layer_name == "conv":
                arch.append(self.add_conv(self.layers[layer]))
            
            # max pooling
            elif layer_name == "max_pool":
                arch.append(self.add_max_pool(self.layers[layer]))
            
            else:
                raise ValueError("Unknown layer encountered")

        return OrderedDict(arch)
    
    def get_pretrained_model(self):
        
        if len(self.layers.keys()) != 1:
            raise ValueError("Currently only single pretrained model is supported")
        
        for layer in self.layers.keys():
            self.layer = layer
            layer_name = self.layers[layer]["name"] 

            # mobile_net 
            if layer_name == "mobile_net":
                return self.add_mobile_net(self.layers[layer])

            # resnet18
            elif layer_name == "resnet18":
                return self.add_resnet18(self.layers[layer])
            
            # resnet50
            elif layer_name == "resnet50":
                return self.add_resnet50(self.layers[layer])

            else:
                raise ValueError("Unknown pretrained model name encountered")
    
    def set_parameter_requires_grad(self, features, flag):
        """
        Sets requires grad of layers to True/False
        """
        for param in features.parameters():
            param.requires_grad = flag

    def add_linear(self, parameters):
        """
        Adds Linear Layer
        
        parameters: dict containing keys:
            neurons: number of neurons in layer
            bias: whether to add bias term (default = True)
        """
        param = parameters.keys()
        
        if "neurons" in param:
            neurons = parameters["neurons"]
        else:
            raise ValueError ("Number of neurons must be provided")
            
        if "bias" in param:
            bias = parameters["bias"]
        else:
            bias = True
            
        linear = (f'{self.layer}', nn.Linear(
                                    in_features = self.pre_layer_output, 
                                    out_features= neurons, 
                                    bias = bias))
        self.pre_layer_output = neurons
        
        return linear
    
    def add_relu(self, parameters):
        """
        Adds ReLU layer
        """
        return (f'{self.layer}', nn.ReLU())
    
    def add_lrelu(self, parameters):
        """
        Adds ReLU layer
        Parameters:
            alpha: negative slope (default = 1e-2)
        """
        if "alpha" in parameters.keys():
            alpha = parameters["alpha"]
        else:
            alpha = 1e-2
        return (f'{self.layer}', nn.LeakyReLU(alpha))
    
    def add_flatten(self, parameters):
        """
        Adds Flatten layer
        """
        flatten_layer = (f'{self.layer}', nn.Flatten())
        in_shape = self.pre_layer_output
        
        # calculate output shape
        out_shape = 1
        for d in range(len(in_shape)):
            out_shape *= in_shape[d]
        
        self.pre_layer_output = out_shape
        
        return flatten_layer
    
    def add_conv(self, parameters):
        """
        Adds Conv layer
        
        Parameters:
            number_of_kernels: int
            kernel_size: int or tuple of ints
            stride: int or tuple of ints (default=1)
            padding: 'valid' or 'same' or tuple of ints (default='valid')
            bias: default=True
            
        """
        param_keys = parameters.keys()
        
        if "number_of_kernels" in param_keys:
            out_channels = parameters["number_of_kernels"]
        else:
            raise ValueError("When adding Convulational Layer, kernel_size must be provided")
        
        
        if "kernel_size" in param_keys:
            kernel_size = parameters["kernel_size"]
        else:
            raise ValueError("When adding Convulational Layer, kernel_size must be provided")
        
        if "stride" in param_keys:
            stride = parameters["stride"]
        else:
            stride = 1
            
        if "padding" in param_keys:
            padding = parameters["padding"]
        else:
            padding = 'valid'
        
        if "bias" in param_keys:
            bias = parameters["bias"]
        else:
            bias = True
        
        if len(self.pre_layer_output) == 2:
            in_channels = 1
            in_height, in_width =  self.pre_layer_output
        else:
            in_channels, in_height, in_width =  self.pre_layer_output
        
        if padding == 'valid':
            pad = 0
        
        if padding == 'same':
            pad =  ( (in_height - 1)* stride + kernel_size - in_height)/2
       
        
        out_height = (in_height + 2*pad - kernel_size)/ stride + 1
        out_width = (in_width + 2*pad - kernel_size)/ stride + 1
        
        if out_width %1 !=0 or out_height %1 !=0 :
            raise ValueError(f"Combination of pad, stride and kernel size lead to decimal number in conv layer. {out_height}")

        self.pre_layer_output = (out_channels, int(out_height), int(out_width))
        
        conv_layer = (f'{self.layer}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        
        return conv_layer
    
    def add_max_pool(self, parameters):
        """
        Adds Max pooling layer
        
        Parameters:
            kernel_size: int or tuple of ints
            stride: int or tuple of ints (default=2)
            padding:int or tuple of ints (default=0)
        
        """
        
        param_keys = parameters.keys()
        
        if "kernel_size" in param_keys:
            kernel_size = parameters["kernel_size"]
        else:
            raise ValueError("When adding Convulational Layer, kernel_size must be provided")
        
        if "stride" in param_keys:
            stride = parameters["stride"]
        else:
            stride = 2
            
        if "padding" in param_keys:
            padding = parameters["padding"]
        else:
            padding = 0
        
        
        if(len(self.pre_layer_output)) != 3:
            raise ValueError(f"The input shape to the Max Pooling Layer is not correct. {self.pre_layer_output}")
         
        in_channels, in_height, in_width =  self.pre_layer_output
        
        out_height = (in_height + 2*padding - kernel_size)/ stride + 1
        out_width = (in_width + 2*padding - kernel_size)/ stride + 1
        out_channels = in_channels
       
        if out_width %1 !=0 or out_height %1 !=0 :
            raise ValueError(f"Combination of pad, stride and kernel size lead to decimal number in conv layer. {out_height}")


        self.pre_layer_output = (out_channels, int(out_height), int(out_width))
        
        max_pool_layer = (f'{self.layer}', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        
        return max_pool_layer
    
    def add_mobile_net(self, parameters):
        """
        Returns a mobile net pretrained model. Currently supports V2 weights
        
        Parameters:
            weights: default = 'IMAGENET1K_V2'
            num_output_neurons = number of neurons in the last layer
            num_layers_to_train = number of layers to train. it starts counting from last layer. (default=0 -> only last layer is trained)
                                pretrained weights will be used for layers to be trained.
                                By passing -1, all layers will be trained. this is equal to finetuning.
                                
        """
        param_keys= parameters.keys()
        
        if "weights" in param_keys:
            weights = parameters["weights"]
        else:
            weights = 'IMAGENET1K_V2'
        
        if "num_output_neurons" in param_keys:
            num_output_neurons = parameters["num_output_neurons"]
        else:
            raise ValueError("num_output_neurons must be provided while using mobile net")
            
        if "num_layers_to_train" in param_keys:
            num_layers_to_train = parameters["num_layers_to_train"]
        else:
            num_layers_to_train = 0
        
        mobile_net = models.mobilenet_v2(pretrained=True)
        
        if num_layers_to_train >=0:
            self.set_parameter_requires_grad(mobile_net, False)
            
            if num_layers_to_train > 0:
                train_features_index = len(mobile_net.features) - num_layers_to_train
                self.set_parameter_requires_grad(mobile_net.features[train_features_index:], True)
                

        in_features = mobile_net.classifier[-1].in_features
        mobile_net.classifier[-1] = nn.Linear(in_features, num_output_neurons)
                
        return mobile_net
        
        
    def add_resnet18(self, parameters):
        """
        Returns resnet pretrained model. Currently supports ResNet18 and supports IMAGENET1K_V1
        
        Parameters:
            weights: default = 'IMAGENET1K_V1'
            num_output_neurons = number of neurons in the last layer
            num_layers_to_train = number of layers to train. it starts counting from last layer. (default=0 -> only last layer is trained)
                                pretrained weights will be used for layers to be trained.
                                By passing -1, all layers will be trained. this is equal to finetuning.
                                
        """
        param_keys= parameters.keys()
        
        if "weights" in param_keys:
            weights = parameters["weights"]
        else:
            weights = 'IMAGENET1K_V1'
        
        if "num_output_neurons" in param_keys:
            num_output_neurons = parameters["num_output_neurons"]
        else:
            raise ValueError("num_output_neurons must be provided while using mobile net")
            
        if "num_layers_to_train" in param_keys:
            num_layers_to_train = parameters["num_layers_to_train"]
        else:
            num_layers_to_train = 0
            
        resnet = models.resnet18(pretrained=True)
        
        if num_layers_to_train >=0:
            self.set_parameter_requires_grad(resnet, False)
            
            if num_layers_to_train > 0:
                print("Training previous layers in ResNet18 is not supported yet")
                
                #train_features_index = len(resnet.features) - num_layers_to_train
                #self.set_parameter_requires_grad(resnet.features[train_features_index:], True)
                

        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_output_neurons)
                
        return resnet
        
    def add_resnet50(self, parameters):
        """
        Returns resnet pretrained model. Currently supports ResNet18 and supports IMAGENET1K_V1
        
        Parameters:
            weights: default = 'IMAGENET1K_V2'
            num_output_neurons = number of neurons in the last layer
            num_layers_to_train = number of layers to train. it starts counting from last layer. (default=0 -> only last layer is trained)
                                pretrained weights will be used for layers to be trained.
                                By passing -1, all layers will be trained. this is equal to finetuning.
                                
        """
        param_keys= parameters.keys()
        
        if "weights" in param_keys:
            weights = parameters["weights"]
        else:
            weights = 'IMAGENET1K_V2'
        
        if "num_output_neurons" in param_keys:
            num_output_neurons = parameters["num_output_neurons"]
        else:
            raise ValueError("num_output_neurons must be provided while using mobile net")
            
        if "num_layers_to_train" in param_keys:
            num_layers_to_train = parameters["num_layers_to_train"]
        else:
            num_layers_to_train = 0
            
        resnet = models.resnet50(pretrained=True)
        
        if num_layers_to_train >=0:
            self.set_parameter_requires_grad(resnet, False)
            
            if num_layers_to_train > 0:
                print("Training previous layers in ResNet50 is not supported yet")
                
                #train_features_index = len(resnet.features) - num_layers_to_train
                #self.set_parameter_requires_grad(resnet.features[train_features_index:], True)
                

        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_output_neurons)
                
        return resnet



def data_preprocessing(transformations):
    """
    Returns compose of transforms
    
    Parameters:
        transformations: Ordered dict with structure
        transformations["transform_1"] = {}
        transformations["transform_1"]["name"] = value
        transformations["transform_1"]["parameter_1"] = value
        transformations["transform_1"]["parameter_2"] = value
        
        transformations["transform_2"] = {}
        transformations["transform_2"]["name"] = value
        transformations["transform_2"]["parameter_1"] = value
        transformations["transform_2"]["parameter_2"] = value
        
    Currently supports transformation: 
        "normalize": 
                    keys: mean (default=0) 
                           std (default=255), 
        "resize": 
                    keys: output_shape, 
        
        "random_resized_crop"
                    keys: output_shape
        
        "random_horizontal_flip"
                    keys: p = 0.5
        
        "random_vectical_flip"
                    keys: p = 0.5           
        
        "center_crop"
                    keys: output_shape
                    
        "random_rotation"
                    keys: degrees (defines the range to select from (-deg, +deg)) # in degrees
                        expand(default=False, whether to increase the size of image so that it can fit whole roatated image)

        "color_jitter":
                    keys: brightness (between 0 and 1)
                          contrast (between  0 and 1)
        
        "random_affine"
                    keys: "translate" (between 0 and 1)
                          "scale" (between 0 and 1)
                          "degree"

        "random_adjust_sharpness":
                    keys: sharpness_factor
                          

        "random_auto_contrast":
                    keys:
                        
    
    # train transform will be different from valid transform (CHECK)
    """
    
    preprocess = []
    
    for tran in transformations.keys():
        tran_name = transformations[tran]["name"]
        
        # normalization
        if tran_name == "normalize":
            norm = transformations[tran].keys()
            if "mean" in norm:
                m = transformations[tran]["mean"]
                if type(m) == list:
                    mean = m
                else:
                    mean = [m]
            else:
                mean = [0]
            
            if "std" in norm:
                s =transformations[tran]["std"]
                if type(s) == list:
                    std = s
                else:
                    std = [s]
            else:
                std = [255]
                
            preprocess.append(transforms.Normalize(mean=mean,
                             std=std))
        # resize
        elif tran_name == "resize":
            resize = transformations[tran].keys()
            
            if "output_shape" in resize:
                output_shape = transformations[tran]["output_shape"]
            else:
                raise ValueError("Output shape must be provided while resizing")
            
            preprocess.append(transforms.Resize(output_shape))
            
        # random resize crop
        elif tran_name == "random_resized_crop":
            rrc = transformations[tran].keys()
            
            if "output_shape" in rrc:
                output_shape = transformations[tran]["output_shape"]
            else:
                raise ValueError("Output shape must be provided while random_resized_crop") 
            
            preprocess.append(transforms.RandomResizedCrop(output_shape))
        
        # random horizontal flip
        elif tran_name == "random_horizontal_flip":
            rhf = transformations[tran].keys()
            
            if "p" in rhf:
                p = transformations[tran]["p"]
            else:
                p = 0.5
            
            preprocess.append(transforms.RandomHorizontalFlip(p))
         
        # random vertical flip
        elif tran_name == "random_vertical_flip":
            rvf = transformations[tran].keys()
            
            if "p" in rvf:
                p = transformations[tran]["p"]
            else:
                p = 0.5
            
            preprocess.append(transforms.RandomVerticalFlip(p))
            
        # center crop
        elif tran_name == "center_crop":
            cc = transformations[tran].keys()
            
            if "output_shape" in cc:
                output_shape = transformations[tran]["output_shape"]
            else:
                raise ValueError("Output shape must be provided while center_crop") 
            
            preprocess.append(transforms.CenterCrop(output_shape))
        
        # random_rotation
        elif tran_name == "random_rotation":
            rr = transformations[tran].keys()
            
            if "degrees" in rr:
                degrees = transformations[tran]["degrees"]
            else:
                raise ValueError("degrees must be provided while random rotation")
            
            if "expand" in rr:
                expand = transformations[tran]["expand"]
            else:
                expand = False
            
            preprocess.append(transforms.RandomRotation(degrees=degrees, expand=expand))

        elif tran_name == "color_jitter":
            cj = transformations[tran].keys()
            
            if "brightness" in cj:
                brightness = transformations[tran]["brightness"]
            else:
                raise ValueError("brightness must be provided while color jittering")
            
            if "contrast" in cj:
                contrast = transformations[tran]["contrast"]
            else:
                raise ValueError("brightness must be provided while color jittering")
            
            preprocess.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

        elif tran_name == "random_affine":
            ra = transformations[tran].keys()

            if "degrees" in ra:
                degrees = transformations[tran]["degrees"]
            else:
                raise ValueError("degrees must be provided while random affine")
            
            if "translate" in ra:
                translate = transformations[tran]["translate"]
            else:
                raise ValueError("translate must be provided while random affine")
            
            if "scale" in ra:
                scale = transformations[tran]["scale"]
            else:
                raise ValueError("scale must be provided while random affine")
            
            preprocess.append(transforms.RandomAffine(degrees=degrees, scale=scale, translate=translate))

        
        elif tran_name == "random_auto_contrast":
            rac = transformations[tran].keys()
            preprocess.append(transforms.RandomAutocontrast())
        
        elif tran_name == "random_adjust_sharpness":
            ras = transformations[tran].keys()

            if "sharpness_factor" in ras:
                sharpness_factor = transformations[tran]["sharpness_factor"]
            else:
                raise ValueError("sharpness_factor must be provided while random_adjust_sharpness")

            preprocess.append(transforms.RandomAdjustSharpness(sharpness_factor))

        elif tran_name == "random_apply_affine":
            
            preprocess.append(transforms.RandomApply([transforms.RandomAffine(degrees=(-15,15), scale=(0.1,0.3), translate=(0.1,0.3))]))
          
        elif tran_name == "random_apply_rotation":
            
            preprocess.append(transforms.RandomApply([transforms.RandomRotation(degrees=90, expand=True)]))
        
          
        
        else:
            raise ValueError("Unknown transformation passed")
    
    return transforms.Compose(preprocess)

def load_from_json(path_to_json):
    """
    Load a json file

    Parameters:
        path_to_json: path to json file
    
    Returns:
        dictionary with contents of the json
    """
    with open(path_to_json, 'r') as f:
        dic = json.load(f)
    
    return dic#


def save_to_json(dic, path_to_results):
    """
    Save a dicitonary to a location

    Parameters:
        dic: dictionary to be saved
        path_to_results: path where to save the dictionary

    """
    with open(path_to_results+".json", "w") as fp:
        json.dump(dic, fp, indent=4)


def generate_txt_file(y, path_to_results, name_of_file, y_probs=None):
    """
    Generates a text file with structure as follows:
    file_name_1 label
    file_name_2 label


    Parameters:
        y: numpy array of shape(folds, num_images, 2)
        path_to_results: path to the folder where to store results
        name_of_file (without .txt)
        y_probs: numpy array of shape (folds, num_images, 1)

    """
    

    num_folds = y.shape[0]
    num_images = y.shape[1]


    for fold_no in range(num_folds):

        print(f"Processing Fold No: {fold_no}")

        path_to_fold = os.path.join(path_to_results, str(fold_no))
        if not os.path.exists(path_to_fold):
                os.mkdir(path_to_fold)
        
        # generate file
        path_to_file = path_to_fold + "/" + name_of_file + ".txt"
        open(path_to_file, "w").close()
        with open(path_to_file, "a") as file:
            for img_no in range(num_images):
                file.write(f"{y[fold_no, img_no, 0]} {y[fold_no, img_no, 1]}\n")
        
        # generate files with prob
        if y_probs is not None:
            path_to_prob_file = path_to_fold + "/" + name_of_file + "_with_probs.txt"
            open(path_to_prob_file, "w").close()
            with open(path_to_prob_file, "a") as file:
                for img_no in range(num_images):
                    file.write(f"{y[fold_no, img_no, 0]} {y[fold_no, img_no, 1]} {np.round(y_probs[fold_no, img_no], 4)*100}\n")


def get_predictions(path_to_results, path_to_images, y, data_transforms, batch_size, device, fold=-2, color="gray"):
    """
    generates predictions
    
    path_to_results: where models are saved
    path_to_images: images dir
    y: (num_images, 1)
    data_transforms: transformation to apply
    fold: prediction for specific fold
    
    
    Returns:
        y_preds_folds: (num_folds, num_images)
        y_preds_probs_folds: (num_folds, num_images, num_classes) 
        y_preds_en: (num_images,1)
        y_preds_probs_ensemble: (num_images, num_classes)
        
    thresholding could be added for binary (>th)/multi(>th for each class) classsifcation
    """
    y_preds_probs_folds=[]
    y_preds_folds = []
    
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)       
    
    flag = False
    single_neuron = False
    
    for fold_no in sorted(os.listdir(path_to_results)):
        
        path_to_model = os.path.join(path_to_results, fold_no)
        
        if os.path.isdir(path_to_model):
            
            if fold == -2:
                flag = True
            else:
                if fold_no == str(fold):
                    flag = True
                else:
                    flag =False
            
            if flag:
                print(f"Generating predicitons for fold no: {fold_no}")

                # get data
                data = ImageDataset(y, path_to_images, transform=data_transforms, color=color)
                dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

                # load model
                model_path = os.path.join(path_to_model, "model.pt")
                print (f"Loading model from {model_path}")
                model = torch.load(model_path, map_location=device)
                model.to(device)
                model.eval()

                with torch.no_grad():

                    y_pred_prob=[]
                    y_pred_labels =[]

                    for images, _ in dataloader:
                        images = images.to(device)

                        y_pred = model(images)

                        if y_pred.shape[-1] == 1:
                            # get probs
                            prob = sigmoid(y_pred)

                            # get hard labels
                            labels = np.round(prob.detach().cpu().clone().numpy())

                            single_neuron = True
                        else:
                            #get probs
                            prob = softmax(y_pred)

                            # get hard labels
                            labels = np.argmax(prob.detach().cpu().clone().numpy(), axis=1) 

                        #combine iter results
                        y_pred_prob.extend(prob.detach().cpu().clone().numpy())
                        y_pred_labels.extend(labels)


                    # combine fold results
                    y_preds_folds.append(y_pred_labels)
                    y_preds_probs_folds.append(y_pred_prob)

        
    # get ensemble results
    y_preds_probs_ensemble = np.mean(y_preds_probs_folds, axis=0)
    

    if single_neuron:
        y_preds_en = np.round(y_preds_probs_ensemble).reshape(-1,1)
    else:
        y_preds_en = np.argmax(y_preds_probs_ensemble, axis=1).reshape(-1,1)
        
    return np.array(y_preds_folds), np.array(y_preds_probs_folds), np.array(y_preds_en), np.array(y_preds_probs_ensemble)

def CNN_prediction(path_to_images, path_to_json, save_path):
    """
    Generates prediction for the given data set

    Parameters:
        path_to_images: path to the folder containing images on which prediction will be generated
        path_to_results: path to the folder where models are saved
        save_path: path to the folder where the results will be stored
        data_transform: transformation to apply
        batch_size: batch size

    """

    if not os.path.isdir(path_to_images):
        raise ValueError("'path_to_images' must be a directory")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created directory {save_path}")
    else:
        print(f"Warning: {save_path} already exists. Content may get overwritten")
    
    # loading training pipeline
    pipeline = load_from_json(path_to_json)

    pipeline["path_to_images"] = path_to_images
    path_to_models = pipeline["path_to_results"]
    
    pipeline["path_to_models"] = path_to_models
    pipeline["path_to_results"] = save_path
    

    del pipeline['path_to_labels']
    
    device = pipeline["device"]
    batch_size= pipeline["batch_size"]
    classes = pipeline["classes"]
    read_img_color = pipeline["read_img_color"]
    
    
    # get transforms
    data_transform = pipeline["data_preprocessing"]["valid"]
    data_transforms = data_preprocessing(data_transform)
    
    # generate y
    y=np.array(sorted(os.listdir(path_to_images))).reshape(-1,1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
  
    print("\nGenerating labels for the data . . . ")
    y_pred, y_pred_probs, y_pred_en, y_pred_probs_en = get_predictions(path_to_results=path_to_models, path_to_images=path_to_images, y=y, data_transforms=data_transforms, batch_size=batch_size, device=device, color=read_img_color)
    
   
    # transform labels and add image ids
    y_pred_tr = np.empty((y_pred.shape[0], y.shape[0], 2), dtype=y.dtype) 
    for nf in range(y_pred.shape[0]):
        y_pred_tr[nf, :, 0] = y[:,0]
        y_pred_tr[nf, :, 1] = label_encoder(y=y_pred[nf,:], classes=classes, to_numbers=False)
    
    
    y_pred_probs_en = np.expand_dims(y_pred_probs_en, axis=0)
    y_pred_en = np.expand_dims (np.concatenate((y, y_pred_en), axis=1), axis=0)
    y_pred_en[0, :, 1] = label_encoder(y=y_pred_en[0, :, 1], classes=classes, to_numbers=False)
    
    print(f"Generated labels for the data successfully. Shape: y_pred: {y_pred_tr.shape} y_pred_probs: {y_pred_probs.shape} y_pred_en: {y_pred_en.shape} y_pred_probs_en: {y_pred_probs_en.shape}")
    
    

    # save pipeline
    print("\nsaving pipeline dictionary to json")
    save_to_json(
        pipeline, 
        save_path +"/pipeline"
        )
    
    # save labels
    print("\nGenerating text file for the data")
    generate_txt_file(
        y=y_pred_tr, 
        path_to_results=save_path, 
        name_of_file="labels",
        y_probs = np.max(y_pred_probs, axis=-1)
        )
    
     # save labels with probs
    print("\nGenerating text file for the data ensemble")
    generate_txt_file(
        y=y_pred_en, 
        path_to_results=save_path, 
        name_of_file="labels_en",
        y_probs = np.max(y_pred_probs_en, axis=-1)
        )
    
    
    pipeline["path_to_models"] = path_to_models
    pipeline["path_to_results"] = save_path
    
    print("\nGenerated Predictions Successfully")


if __name__ == "__main__":

    path_to_json = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/documents/presentation/phase2/ResNet50/offlien aug/ResNet50_offline_21-01/binary/train/training_pipeline.json"
    save_path = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/documents/presentation/phase2/ResNet50/offlien aug/ResNet50_offline_21-01/binary/noisy_data-21-01"
    path_to_noisy_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/aug_dataset-21-01/combined"

    # change json

    start = time.time()
    CNN_prediction(path_to_noisy_data, path_to_json, save_path)
    end = time.time()
    print("Time taken to generate predictions on test data : ", datetime.timedelta(seconds=end-start))

    #read y

    #read y_pred, y_pred_prob

    # setup for metrics

    # setup plots