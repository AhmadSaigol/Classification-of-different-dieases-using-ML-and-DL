"""
normalize the features

"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler 
import numpy as np

def normalize_features(features, parameters):
    """
    nomalizes the features 

    Parameters:
        features: numpy array of shape(folds, num_images, num_features)
        parameters: dictionary containing the keys:
            norm_type: how to normalize the features currently it supports
                        ["StandardScaler", "MinMaxScaler", "MaxAbsScaler"]
           
    
    Returns:
        norm_features: numpy array of shape (folds, num_images, num_features)
        config: dictionary with parameters of the function (including default parameters)
    

    Note:
        For method "StandardScaler",
                formula: z = (x - mean) / std
                following keys will be returned in the 'config'
                    mean: mean values of features of shape (folds, num_features)
                    var: var values of features of shape (folds, num_features)
                    scale: scale values of features of shape (folds, num_features)
                if these keys are given in 'parameters', then code will make use of these values for calculation
                otherwise they will be calculated from the data
                if number of folds in these keys do not match with 'features', then mean values of these keys will be used
                for calculation in the testing phase

        For method "MinMaxScaler",
                following keys will be returned in the 'config'
                    min: min values of features of shape (folds, num_features)
                    scale: scale values of features of shape (folds, num_features)
                if these keys are given in 'parameters', then code will make use of these values for calculation
                otherwise they will be calculated from the data
                if number of folds in these keys do not match with 'features', then mean values of these keys will be used
                for calculation in the testing phase
        
        For method "MaxAbsScaler",
                following keys will be returned in the 'config'
                    max_abs: max_abs values of features of shape (folds, num_features)
                    scale: scale values of features of shape (folds, num_features)
                if these keys are given in 'parameters', then code will make use of these values for calculation
                otherwise they will be calculated from the data
                if number of folds in these keys do not match with 'features', then mean values of these keys will be used
                for calculation in the testing phase



    """
    config = {}

    param_keys= parameters.keys()

    if "norm_type" in param_keys:
        norm_type = parameters["norm_type"]
        config["norm_type"] = norm_type
    else:
        raise ValueError("'norm_type' must be provided in the parameters when normalizating the features.") 
    


    if norm_type == "StandardScaler":
        scaler = StandardScaler()
        
        
        # Testing Phase
        if "mean" in param_keys and "var" in param_keys and "scale" in param_keys:
            print("Normalizing Features in Testing Phase")
            mean_features = parameters["mean"]        
            var_features = parameters["var"]
            scale_features = parameters["scale"]

            # Testing features
            if mean_features.shape[0] != features.shape[0]:
                print("using mean values of keys 'mean', 'var' and 'scale' for calculation")
                mean_features = np.mean(mean_features, axis=0)
                var_features = np.mean(var_features, axis=0)
                scale_features = np.mean(scale_features, axis=0)
                
                scaler.mean_ = mean_features
                scaler.var_ = var_features
                scaler.scale_ = scale_features

                results = np.expand_dims(scaler.transform(features[0]), axis=0)

            # validation features
            else:
                results = np.zeros(features.shape)

                for fold_no in range(features.shape[0]):
                    
                    scaler.mean_ = mean_features[fold_no]
                    scaler.var_ = var_features[fold_no]
                    scaler.scale_ = scale_features[fold_no]
                    
                    results[fold_no] = scaler.transform(features[fold_no])

        # Training Phase
        else:   
            print("Normalizing Features in Training Phase")
            
            mean_features = np.zeros((features.shape[0], features.shape[2]))
            var_features = np.zeros((features.shape[0], features.shape[2]))
            scale_features = np.zeros((features.shape[0], features.shape[2]))

            results = np.zeros(features.shape)

            for fold_no in range(features.shape[0]):

                # calculate mean and var of each feature
                scaler.fit(features[fold_no])

                mean_features[fold_no] = scaler.mean_
                var_features[fold_no] = scaler.var_
                scale_features[fold_no] = scaler.scale_
                
                # normalize data
                results[fold_no] = scaler.transform(features[fold_no])
            
        config["mean"] = mean_features
        config["var"] = var_features
        config["scale"] = scale_features
    
    elif norm_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        
        
        # Testing Phase
        if "min" in param_keys and "scale" in param_keys:
            print("Normalizing Features in Testing Phase")
            min_features = parameters["min"]        
            scale_features = parameters["scale"]

            # Testing features
            if min_features.shape[0] != features.shape[0]:
                print("using mean values of keys 'min' and 'scale' for calculation")
                
                min_features = np.mean(min_features, axis=0)
                scale_features = np.mean(scale_features, axis=0)
                
                scaler.min_ = min_features
                scaler.scale_ = scale_features

                results = np.expand_dims(scaler.transform(features[0]), axis=0)

            # validation features
            else:
                results = np.zeros(features.shape)

                for fold_no in range(features.shape[0]):
                    
                    scaler.min_ = min_features[fold_no]
                    scaler.scale_ = scale_features[fold_no]
                    
                    results[fold_no] = scaler.transform(features[fold_no])

        # Training Phase
        else:   
            print("Normalizing Features in Training Phase")
            
            min_features = np.zeros((features.shape[0], features.shape[2]))
            scale_features = np.zeros((features.shape[0], features.shape[2]))

            results = np.zeros(features.shape)

            for fold_no in range(features.shape[0]):

                # calculate mean and max of each feature
                scaler.fit(features[fold_no])

                min_features[fold_no] = scaler.min_
                scale_features[fold_no] = scaler.scale_
                
                # normalize data
                results[fold_no] = scaler.transform(features[fold_no])
            
        config["min"] = min_features
        config["scale"] = scale_features
    
    elif norm_type == "MaxAbsScaler":
        scaler = MaxAbsScaler()
        
        
        # Testing Phase
        if "max_abs" in param_keys and "scale" in param_keys:
            print("Normalizing Features in Testing Phase")
            max_abs_features = parameters["max_abs"]        
            scale_features = parameters["scale"]

            # Testing features
            if max_abs_features.shape[0] != features.shape[0]:
                print("using mean values of keys 'max_abs' and 'scale' for calculation")
                
                max_abs_features = np.mean(max_abs_features, axis=0)
                scale_features = np.mean(scale_features, axis=0)
                
                scaler.max_abs_ = max_abs_features
                scaler.scale_ = scale_features

                results = np.expand_dims(scaler.transform(features[0]), axis=0)

            # validation features
            else:
                results = np.zeros(features.shape)

                for fold_no in range(features.shape[0]):
                    
                    scaler.max_abs_ = max_abs_features[fold_no]
                    scaler.scale_ = scale_features[fold_no]
                    
                    results[fold_no] = scaler.transform(features[fold_no])

        # Training Phase
        else:   
            print("Normalizing Features in Training Phase")
            
            max_abs_features = np.zeros((features.shape[0], features.shape[2]))
            scale_features = np.zeros((features.shape[0], features.shape[2]))

            results = np.zeros(features.shape)

            for fold_no in range(features.shape[0]):

                # calculate mean and max of each feature
                scaler.fit(features[fold_no])

                max_abs_features[fold_no] = scaler.max_abs_
                scale_features[fold_no] = scaler.scale_
                
                # normalize data
                results[fold_no] = scaler.transform(features[fold_no])
            
        config["max_abs"] = max_abs_features
        config["scale"] = scale_features

    else:
        raise ValueError("Unknown Value encountered for the parameter'norm_type' while normalizing the features.")

    return results, config


if __name__ == "__main__":

    features = np.arange(1*3*2).reshape(1,3,2)
    print(features)

    pipeline={}
    pipeline["normalize_features"] = {}
    pipeline["normalize_features"]["norm_type"] = "StandardScaler"
    #pipeline["normalize_features"]['min'] = np.array([[ 0.  , -0.25],[-1.5 , -1.75]])
    #pipeline["normalize_features"]['scale'] = np.array([[0.25, 0.25],[0.25, 0.25]])
    
    #pipeline["normalize_features"]['max_abs'] = np.array([[ 4.,  5.],[10., 11.]])
    #pipeline["normalize_features"]['scale'] = np.array([[ 4.,  5.],[10., 11.]])
    
    pipeline["normalize_features"]['mean'] = np.array([[2., 3.],[8., 9.]])
    pipeline["normalize_features"]['var'] = np.array([[2.66666667, 2.66666667],[2.66666667, 2.66666667]])
    pipeline["normalize_features"]['scale'] = np.array([[1.63299316, 1.63299316],[1.63299316, 1.63299316]])
    
    new, config = normalize_features(features, parameters=pipeline["normalize_features"])
    
    print(new)
    print(config)