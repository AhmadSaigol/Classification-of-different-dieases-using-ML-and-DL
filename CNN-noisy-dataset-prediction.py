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
from utils.misc import change_txt_for_binary


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


def precision(y_true, y_pred, parameters):
    """
    Calculates precision
    
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
        precision = TP/ TP+FP

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

        score = precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    
    else:
        
        if "average" in parameters.keys():
            average = parameters["average"]
        else:
            average = "weighted"

        config["average"] = average

        score = precision_score(y_true=y_true, y_pred=y_pred, average=average)

    return score, config
def F1_score(y_true, y_pred, parameters):
    """
    Calculates f1 score
    
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
        f1_score = 2 * (Precision * Recall)/ (Precision + Recall)

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

        score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    
    else:
        
        if "average" in parameters.keys():
            average = parameters["average"]
        else:
            average = "weighted"

        config["average"] = average

        score = f1_score(y_true=y_true, y_pred=y_pred, average=average)

    return score, config
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
def mcc(y_true, y_pred, parameters):
    """
    Calculates mcc
    
    Parameters:
        y_true: numpy array of shape (num_images,)
        y_pred: numpy array of shape (num_images,)
        parameters: dictionary with the following keys:
            
    Returns:
        score: float
        config:

    Additional Notes:
        Binary:
        
            +1 -> prefect, 0-> random, -1 -> inverse 

            mcc = (tp*tn) - (fp*fn) / sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  )

        Multiclass:
            
            +1 -> perfect, between -1 and 0 -> min

            for more info, see
                "https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef"
    """

    config = {}

    score = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    return score, config
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
def evaluate_metrics(y_true, y_pred, metrics, y_pred_probs=None):
    """
    Applies each metric and generates evaluation score

    Parameters:
        y_true: numpy array of shape (folds, num_images, 2)
        y_pred: numpy array of shape (folds, num_images, 2)
        metrics: dictionary with following structure:
            metrics["metrics_1"]["name"] = name of metric
            metrics["metrics_1"]["parameter_1"] = value
            metrics["metrics_1"]["parameter_2"] = value

            metrics["metrics_2"]["name"] = name of metric
            metrics["metrics_2"]["parameter_1"] = value
            metrics["metrics_2"]["parameter_2"] = value
            
    
    currently, supports "accuracy", "mcc", "precision", "sensitivity", "F1_score"
  

    Returns:
        scores:numpy array of shape (folds, metrics)
        output_config:
       
    Additional Notes:

    """
    # check whether correct shapes of y_* are provided or not

    if len(y_true.shape) !=3:
        raise ValueError("Shape of y_true is not correct.")

    if len(y_pred.shape) == 3:
        num_folds = y_pred.shape[0]
        met_keys = list(metrics.keys())
        num_metrics = len(met_keys)
    else:
        raise ValueError("Shape of y_pred is not correct.")

    config={}

    list_of_metrics = []

    scores = np.full((num_folds, num_metrics), 1000, dtype=np.float32)
    
    flag = True


    for fold_no in range(num_folds):

        for metric_no in range(num_metrics):

            print(f"Processing: Fold No: {fold_no} Metric: {met_keys[metric_no]}")

            metric_name = metrics[met_keys[metric_no]]["name"]
            
            if metric_name == "accuracy":
                fnt_pointer = accuracy
            elif metric_name == "mcc":
                fnt_pointer = mcc
            elif metric_name == "precision":
                fnt_pointer = precision
            elif metric_name =="sensitivity":
                fnt_pointer = sensitivity
            elif metric_name =="F1_score":
                fnt_pointer = F1_score
            else:
                raise ValueError("Unknown metric found")

            
            metric_score, fnt_config = fnt_pointer(y_true=y_true[fold_no, :, 1], 
                                                    y_pred=y_pred[fold_no, :, 1], 
                                                    parameters=metrics[met_keys[metric_no]])


            scores[fold_no, metric_no] = metric_score

            # setup output config
            if flag:
                fnt_config["name"] = metric_name
                config[met_keys[metric_no]] = {} 
                config[met_keys[metric_no]] = fnt_config

                list_of_metrics.append(met_keys[metric_no])

        flag=False
    
    return scores, config, list_of_metrics

def plot_MS(y_true, y_pred, path_to_results, path_to_images):
    """
    Plots and save missclassified samples
    
    y_true: numpy array of shape (num_of_images,2)
    y_pred: numpy array of shape (num_of_images,2)
    path_to_results: path where plot will be saved
    path_to_images: folder containing images


    Only works with (num_samples)^2 = whole number

    """
    
    num_samples_to_plot = 4

    classes = np.unique(y_true[:,1])

    if np.sqrt(num_samples_to_plot) %1 !=0:
        raise ValueError("Currently, this function only supports those number of samples whose sqaure is a whole number")

    
    
    cm = create_dict(classes)

    for true_label in classes:
        
        # get all images for a class in y_true
        pos = y_true[np.where(y_true[:,1] == true_label)]
        
        pred_classes = classes.tolist()
      
        for sample in pos:
            
            # find given image in y_pred
            img_id = y_pred[np.where(y_pred[:,0] == sample[0])]
            
            # store img_ids in their respective col
            for pred_label in pred_classes:
                if img_id[0,1] == pred_label:
                    cm[true_label][pred_label].append(img_id[0,0])

            # check whether there are requried number of samples in each col
            for l in cm[true_label].keys():
                
                if l in pred_classes and len(cm[true_label][l]) == num_samples_to_plot:
                    pred_classes.remove(l)
                
            if not len(pred_classes):
                break
    
    plot_CM_images(cm, num_samples_to_plot, path_to_images, path_to_results)
def plot_CM_images(cm, num_samples, path_to_images, path_to_results):
    """
    Plots and saves images
    
    """
    num_classes = len(cm.keys())
    
    num_imgs_axis = int(np.sqrt(num_samples)) 
    
    num_rows= num_imgs_axis * num_classes
    num_cols = num_imgs_axis * num_classes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10,10))
    
    fig.suptitle("Confusion Matrix of misclassified images")
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.text(0.5, 0.04, 'Predicted Labels', ha='center', va='center')
    fig.text(0.06, 0.5, 'True Labels', ha='center', va='center', rotation='vertical')

    for i, true_label in enumerate(cm.keys()):
        
        pred_labels = cm[true_label]
        
        if i==0:
            row_i = i
        

        for j, pred_label in enumerate(pred_labels.keys()):

            img_ids = pred_labels[pred_label]

            empty_img_ids = num_samples - len(img_ids)

            if j==0:
                col_j = j
            
            row = 0
            col = 0

            for id in img_ids:

                img = mpimg.imread(os.path.join(path_to_images, id))
                img_shape = img.shape
                
                axes[row_i+row, col_j+col].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
                axes[row_i+row, col_j+col].set_xticks([])
                axes[row_i+row, col_j+col].set_yticks([])

                if row_i + row == num_rows-1:
                    axes[row_i+row, col_j+col].set_xlabel(pred_label)

            
                if col_j + col == 0:
                    axes[row_i+row, col_j+col].set_ylabel(true_label)




                if col ==num_imgs_axis-1:
                    col =0
                    row +=1
                else:
                    col +=1

            if len(img_ids) ==0:
                img_shape = (299,299)        
            
            if empty_img_ids > 0:
                temp = np.full(img_shape, 255)

                for _ in range(empty_img_ids):
                    
                    axes[row_i+row, col_j+col].imshow(temp, cmap='gray', vmin=0, vmax=255, aspect='auto')
                    axes[row_i+row, col_j+col].set_xticks([])
                    axes[row_i+row, col_j+col].set_yticks([])

                    if row_i + row == num_rows-1:
                        axes[row_i+row, col_j+col].set_xlabel(pred_label)

                
                    if col_j + col == 0:
                        axes[row_i+row, col_j+col].set_ylabel(true_label)
                        
                    
                    if col ==num_imgs_axis-1:
                        col =0
                        row+=1
                    else:
                        col +=1
                   
            if empty_img_ids<0:
                raise ValueError("There are more image ids in the dict than number of samples to be plotted")
            
            col_j = col_j + num_imgs_axis
        
        row_i += num_imgs_axis
    
    plt.savefig(path_to_results + "_MS")
    plt.close()  
def create_dict(classes):

    """
    Setups a dictionary for storing image ids in confusion matrix
    
    """

    output_dict = dict()
    for i in classes:
        output_dict[i] = {}
        for j in classes:
            output_dict[i][j]= []

    return output_dict 

def plot_CM(y_true, y_pred, path_to_results, path_to_images):
    """
    Plots and save Confusion Matrix
    
    y_true: numpy array of shape (num_of_images,2)
    y_pred: numpy array of shape (num_of_images,2)
    path_to_results: path where plot will be saved

    """

    ConfusionMatrixDisplay.from_predictions(y_pred=y_pred[:,1], y_true =y_true[:,1])

    plt.savefig(path_to_results + "_CM")
    plt.close()

def create_plots(y_true, y_pred, path_to_results, path_to_images, plots, name_of_file, training_metric_scores=None, y_pred_probs = None):
    """
    Generates differenet plots
    Parameters:
        y_true: numpy array of shape (folds, num_images, 2)
        y_pred: numoy array of shape (folds, num_images, 2)
        path_to_results: path to the folder where the results will be stored
        plots: dictionary with following structure:
                plots["plots_1"]["name"] = name of plot
                plots["plots_1"]["parameter_1"] = value
                plots["plots_1"]["parameter_2"] = value

                plots["plots_2"]["name"] = name of plot
                plots["plots_2"]["parameter_1"] = value
                plots["plots_2"]["parameter_2"] = value
        
        name_of_file:
        training_metric_scores: (default=None) list of numpy array with each having shape of (folds, metrics, 2, epochs).Each item in list should represent results of a network.

    Returns:
        output_config:

    Additional Notes:
    currently supports CM, LC and MS


    """

    # check whether correct shapes of y_* are provided or not

    if len(y_true.shape) !=3:
        raise ValueError("Shape of y_true is not correct.")

    if len(y_pred.shape) == 3:
        num_folds = y_pred.shape[0]
    else:
        raise ValueError("Shape of y_pred is not correct.")

    # determine whether to plot learning curves or not
    if training_metric_scores is not None:
        lc = True
    else:
        lc = False
   


    for fold_no in range(num_folds):

        # create directory for fold
        path_to_fold = os.path.join(path_to_results, str(fold_no))
        if not os.path.exists(path_to_fold):
            os.mkdir(path_to_fold)

        # create directory for plots
        path_to_plots = os.path.join(path_to_fold, "plots")
        if not os.path.exists(path_to_plots):
            os.mkdir(path_to_plots)

        for pl in plots:

            # if dict has key "learning_curves" but no metric score is provided,
            if pl == "LC" and not lc:
                print("Warning: key'learning_curves' provided in dict 'plots' but metric score not found. Skipping ploting learning curves" )
                continue 


            print(f"Creating Plot: {pl} Fold No: {fold_no}")
            
            plot_name = pl
            if plot_name == "CM":
                fnt_pointer = plot_CM
            elif plot_name == "MS":
                fnt_pointer = plot_MS
            elif plot_name == "LC":
                fnt_pointer = plot_LC
            else:
                raise ValueError("Unknown plot found")
                
            path_to_figs = path_to_plots+f"/{name_of_file}"

            if plot_name == "LC":
                fnt_pointer(metric_score= training_metric_scores[fold_no], path_to_results=path_to_figs, path_to_images=path_to_images)
            else:
                fnt_pointer(y_true=y_true[fold_no], y_pred=y_pred[fold_no], path_to_results=path_to_figs, path_to_images=path_to_images)


    return plots

def save_results(results, metrics, path_to_results, name_of_file):
    """
    Save results to csv file

    Parameters:
        results: numpy array of shape (num_folds, metrics)
        metrics: numpy array of name of the metrics
        path_to_results: path to the folder where the results will be stored
        name_of_file: file name
       
    """
    df = pd.DataFrame([], columns=["Fold No", "Metric", "Score"])
    num_folds = results.shape[0]
    num_metrics = results.shape[1]

    for fold_no in range(num_folds):
        for metric_no in range(num_metrics):
            temp = pd.DataFrame([[fold_no, metrics[metric_no], results[fold_no, metric_no]]], columns=["Fold No", "Metric", "Score"])
            df = pd.concat([df, temp])

    df.reset_index(inplace=True, drop=True ) 
       
    df.to_csv(path_to_results + "/" + name_of_file +".csv")

if __name__ == "__main__":
    
    # noisy dataset
    path_to_noisy_data = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/final"
    path_to_noisy_data_labels_multi = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/data_augmented.txt"
    path_to_noisy_data_labels_binary = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/noisy_dataset-21-01/data_augmented_binary.txt"

    #change_txt_for_binary(path_to_noisy_data_labels_multi, path_to_noisy_data_labels_binary)


    y_noisy_binary = np.loadtxt(path_to_noisy_data_labels_binary, dtype=str, delimiter=" ")
    y_noisy_binary = np.expand_dims(y_noisy_binary, axis=0)

    y_noisy_multi = np.loadtxt(path_to_noisy_data_labels_multi, dtype=str, delimiter=" ")
    y_noisy_multi = np.expand_dims(y_noisy_multi, axis=0)

    # results
    path_to_results = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/documents/presentation"

    phase = "phase2"
    model_name = "ResNet50"
    type_aug= "simple"#"simple"#"online aug" #"offlien aug"
    folder_name ="ResNet50_simple_21-01"

    json_name = "train/training_pipeline.json"


    # binary
    print("Processing Binary Classification")

    path_to_json = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "binary", json_name)
    save_path = os.path.join(path_to_results, phase,  model_name, type_aug, folder_name, "binary", "noisy_data-21-01")

    # change json
    print("Processing json . . .")
    pb = load_from_json(path_to_json)
    pb["device"] = "cpu"
    pb["path_to_results"] = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "binary", "train")
    pb["batch_size"] = 32
    pb['read_img_color'] = "rgb"
    save_to_json(pb, path_to_json[:path_to_json.rindex(".json")])

    print("Generating predictions . . . ")
    start = time.time()
    CNN_prediction(path_to_noisy_data, path_to_json, save_path)
    end = time.time()
    print("Time taken to generate predictions on test data : ", datetime.timedelta(seconds=end-start))

    y_pred_binary = np.loadtxt(os.path.join(save_path, "0", "labels.txt"), dtype=str, delimiter=" ")
    y_pred_binary = np.expand_dims(y_pred_binary, axis=0)
     
    # setup for metrics
    metrics = pb["metrics"]
    print("\nEvaluating Metrics on data . . . ")
    eval_score, _, metrics_list = evaluate_metrics(
        y_true=y_noisy_binary, 
        y_pred=y_pred_binary, 
        metrics=metrics,
        y_pred_probs = None
        )
    print("\nResults")


    for met_no, met in enumerate(metrics_list):
        print(f"Metric: {met}  Score: {np.around(eval_score[0, met_no], 4)} ")

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
        )
    print("Created Plots for the data")

    print("\nSaving  results ")
    save_results(
        results=eval_score,
        metrics=metrics_list,
        path_to_results=save_path,
        name_of_file="noisy_dataset"
    )

    
    
    # multi
    print("Processing Multi Classification")

    path_to_json = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "multi", json_name)
    save_path = os.path.join(path_to_results, phase,  model_name, type_aug, folder_name, "multi", "noisy_data-21-01")

    # change json
    print("Processing json . . .")
    pb = load_from_json(path_to_json)
    pb["device"] = "cpu"
    pb["path_to_results"] = os.path.join(path_to_results,phase, model_name, type_aug, folder_name, "multi", "train")
    pb["batch_size"] = 32
    pb['read_img_color'] = "rgb"
    save_to_json(pb, path_to_json[:path_to_json.rindex(".json")])

    print("Generating predictions . . . ")
    start = time.time()
    CNN_prediction(path_to_noisy_data, path_to_json, save_path)
    end = time.time()
    print("Time taken to generate predictions on test data : ", datetime.timedelta(seconds=end-start))

    y_pred_multi = np.loadtxt(os.path.join(save_path,"0", "labels.txt"), dtype=str, delimiter=" ")
    y_pred_multi = np.expand_dims(y_pred_multi, axis=0)
     
    # setup for metrics
    metrics = pb["metrics"]
    print("\nEvaluating Metrics on data . . . ")
    eval_score_multi, _, metrics_list = evaluate_metrics(
        y_true=y_noisy_multi, 
        y_pred=y_pred_multi, 
        metrics=metrics,
        y_pred_probs = None
        )
    print("\nResults")


    for met_no, met in enumerate(metrics_list):
        print(f"Metric: {met}  Score: {np.around(eval_score_multi[0, met_no], 4)} ")

    print(f"Evaluated Metrics on data successfully. Shape:{eval_score_multi.shape}")

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
        )
    print("Created Plots for the data")

    print("\nSaving  results ")
    save_results(
        results=eval_score_multi,
        metrics=metrics_list,
        path_to_results=save_path,
        name_of_file="noisy_dataset"
    )

    print("Processing Completed")

