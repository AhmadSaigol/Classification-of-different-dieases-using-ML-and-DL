"""
Template for adding new data preprocessing utility
"""

def preprocess_data_util(images, parameters):

    """
    Describe what the preprocessing does

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: a dictionary consisting of all key:value pair which your expects
                add description of all (optional and non-optional) keys of your function
                with default values (if any)
    
    Must Return
        results: numpy array of shape (num_images, height, width, channel)
        config: dictionary with parameters of the function (including default parameters)
    
    Additional Notes:

        add here if there are any additional notes of your preprocessing

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


    #Implement your preprocessing routine



    # set up output config
    # which will be a dictionary containing all keys of "parameters" 
    # plus all keys that use default values  
    # plus keys whose values may be calculated during training phase and are required in testing phase

    return results, output_config
