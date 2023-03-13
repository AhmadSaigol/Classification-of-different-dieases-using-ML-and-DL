"""
Template for adding new metric to the pipeline 
"""


def new_metric(y_true, y_pred, parameters):   
    """
    Describe the metric 

    Parameters:
        y_true: numpy array of shape (num_images,)
        y_pred: numpy array of shape (num_images,)
        parameters: a dictionary consisting of all key:value pair which your expects
                add description of all (optional and non-optional) keys of your function
                with default values (if any)
    Returns:
        score: float
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

    #Implement your metric calculation routine

    # set up output config
    # which will be a dictionary containing all keys of "parameters" 
    # plus all keys that use default values  
    # plus keys whose values may be calculated during training phase and are required in testing phase

    return score, config