"""
Generates a feature vector

"""
import numpy as np
from feature_extractors.contrast import calculate_contrast
from feature_extractors.kurtosis import calculate_kurtosis
from feature_extractors.skewness import calculate_skew

def generate_feature_vector(X, extractors):
    """
    Generates a feature vector 

    Parameters:
        X: numpy array of shape (folds, num_images, height, width, num_channels)
        extractors: dictionary with following structure:
            extractors["feature_extractor_1"]["function"] = pointer to the function
            extractors["feature_extractor_1"]["parameter_1"] = value
            extractors["feature_extractor_1"]["parameter_2"] = value

            extractors["feature_extractor_2"]["function"] = pointer to the function
            extractors["feature_extractor_2"]["parameter_1"] = value
            extractors["feature_extractor_2"]["parameter_2"] = value

    Returns:
        feature_vector: numpy array of shape(folds, num_images, num_features)
        config: dictionary with parameters of the function (including default parameters)

    Additional Notes:
        Each feature extractor can return a numpy array of shape (num_images, 1) or (num_images, num_features).
        In the latter case, all elements in the second dim will be considered a separate feature
    
    """
    config = {}

    num_folds = X.shape[0]

    for fold_no in range(num_folds):
        
        flag = True
        
        # apply all extractors
        for extractor in extractors.keys():

            print(f"Fold No: {fold_no} Feature Extractor: {extractor}")
            fnt_pointer = extractors[extractor]["function"]

            feature, fnt_config = fnt_pointer(X[fold_no], extractors[extractor])
            
            
            # concatenate features
            if flag:
                feature_fold = feature
                flag = False
            else:
                feature_fold = np.concatenate((feature_fold, feature), axis=-1)
            

            # setup output config
            if fold_no==0:
                fnt_config["function"] = fnt_pointer
                config[extractor] = {} 
                config[extractor] = fnt_config

        # store feature vector for each fold
        feature_fold = np.expand_dims(feature_fold, axis=0)
        if fold_no == 0:
            feature_vector = feature_fold
        else:
            feature_vector = np.concatenate((feature_vector, feature_fold))

    return feature_vector, config


if __name__ == "__main__":

    X = np.random.rand(5,3,4,5,1)
    print(X.shape)
    
    pipeline = {}
    pipeline["feature_extractors"] ={}

    # contrast
    pipeline["feature_extractors"]["contrast"] = {}
    pipeline["feature_extractors"]["contrast"]["function"] = calculate_contrast
    pipeline["feature_extractors"]["contrast"]["method"] = "michelson" 

    # skewness
    pipeline["feature_extractors"]["skewness"] = {}
    pipeline["feature_extractors"]["skewness"]["function"] =calculate_skew
    pipeline["feature_extractors"]["skewness"]["bias"] = True

    # kurtosis
    pipeline["feature_extractors"]["kurtosis"] = {}
    pipeline["feature_extractors"]["kurtosis"]["function"] = calculate_kurtosis
    pipeline["feature_extractors"]["kurtosis"]["method"] = "pearson"
    pipeline["feature_extractors"]["kurtosis"]["bias"] = True

    # RMS
    pipeline["feature_extractors"]["RMS"] = {}
    pipeline["feature_extractors"]["RMS"]["function"] = calculate_contrast
    pipeline["feature_extractors"]["RMS"]["method"] = "rms"

    features, config = generate_feature_vector(X, pipeline["feature_extractors"])
    print (features)
    print(features.shape)
    print(config)