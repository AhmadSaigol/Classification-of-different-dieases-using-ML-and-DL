"""
Template for adding new feature extractor 
"""

def new_feature_extractor(X, parameters):
    """
    Describe the feature extractor 

    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: a dictionary consisting of all key:value pair which your expects
                add description of all (optional and non-optional) keys of your function
                with default values (if any)
    Returns:
        features: numpy array of shape (num_images, num_features)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:

        add here if there are any additional notes of your feature extractor

    """

    # read all keys from parameters
    # do error checking and setup default values for parameters that are not provided
    # e.g.
    # if "par1" in parameters.keys():
    # p1 = parameters["par1"]
    # else raise Error/ p1 = some_value


        
    # if your routine calculates some value during training phase and that value is required in testing phase,
    # this is something your function will have to handle and it will have to determine whether it is in training phase
    # or testing phase
    # one possible sol is:
    # e.g. function calculates mean value image during training phase,
    # you can add it in parameters["mean_image"] = mean_value_image
    # and have a check in your function whetther "mean_image" exists in parameters or not
    # this way you can tell whether you are in training phase or testing phase



    #Implement your feature extractor routine



    # set up output config
    # which will be a dictionary containing all keys of "parameters" 
    # plus all keys that use default values  
    # plus keys whose values may be calculated during training phase and are required in testing phase

    return feature, config

if __name__ == "__main__":

    # test your code properly before integrating in the pipeline
    pass