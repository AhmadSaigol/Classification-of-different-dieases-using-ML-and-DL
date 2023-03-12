"""
Template for adding new classifier to the pipeline

Note: for working with apply_classifiers.py (not to be used when using neural netowrk)
"""

def new_classifier(X, parameters, classes, y, save_model_path=None, path_to_model=None):
    """
    Describe the classifier

      Parameters:
            X: numpy array of shape (num_images, num_features)
            parameters: a dictionary consisting of all key:value pair which your function expects
                add description of all (optional and non-optional) keys of your function
                with default values
               
            classes: numpy array with names of classes
            y: labels/ids assoicated with the data. numpy array. 
                During the training phase it will have a shape of (num_of_images, 2) 
                        where the first axis contains image ids and second axis contains image labels 
                During the testing phase, it will have a shape of (num_of_images, 1) containing only image ids.
            
            
            save_model_path: (optional) where to save the trained model
            path_to_model: (optional) path to the foler where trained model is saved
        
        Returns:
            y_pred : numpy array of shape(num_images, 2)
            config: dictionary with parameters of the function (including default parameters)

        Additional Notes:
            if save_model_path is provided, then program shoud train the model (training phase) and
            if path_to_model is provided, then program should load the model and generate the predictions (testing phase)
            in case both are provided, it must raise an error.

            add here if there are any additional notes of your classfiiers (and ideally link for more info)
         
    """

    # determine the phase in which the function is being called

    # Training phase:

        # read all keys from parameters
        # do error checking and setup default values for parameters that are not provided
        # e.g.
        # if "par1" in parameters.keys():
        # p1 = parameters["par1"]
        # else raise Error/ p1 = some_value

        #setup, train and save the model 

    #Testing Phase

        # load the model

    # In both phases:
        # generate predictions
        # combine predicted labels with image ids (provided by 'y') -> y_pred

        # set up output config
        # which will be a dictionary containing all keys of "parameters" 
        # plus all keys that use default values  
        # plus keys whose values may be calculated during training phase and are required in testing phase

    return y_pred, config


if __name__ == "__main__":

    # test your code properly before integrating in the pipeline
    pass