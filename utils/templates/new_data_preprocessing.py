"""
Template for adding new data preprocessing utility to the pipeline
"""

def new_data_preprocessing(images, parameters):
    """
    Describe what the preprocessing does

    Parameters: 
        images: numpy array of shape(num_images, height, width, channel)
        parameters: a dictionary consisting of all key:value pair which your function expects
                add description of all (optional and non-optional) keys of your function
                with default values
    
    Returns:
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


    #Implement your preprocessing routine



    # set up output config
    # which will be a dictionary containing all keys of "parameters" 
    # plus all keys that use default values  
    # plus keys whose values may be calculated during training phase and are required in testing phase

    return results, output_config


if __name__ == "__main__":

    # test your code properly before integrating in the pipeline
    pass