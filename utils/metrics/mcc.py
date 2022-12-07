"""
Calculates mcc

"""
from sklearn.metrics import matthews_corrcoef
import numpy as np

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


if __name__ == "__main__":

    #binary
    #y_true = np.array(["aa", "a", "ab", "b", "ac", "a"]).reshape(-1,2)
    #y_pred = np.array(["aa", "a", "ab", "b", "ac", "b"]).reshape(-1,2)
    
    
     
    # multi
    y_true = np.array(["aa", "a", "ab", "b", "ac", "c"]).reshape(-1,2)
    y_pred = np.array(["aa", "b", "ab", "c", "ac", "a"]).reshape(-1,2)
    
    
    pre={}
    pre["mcc"] = {}
    pre["mcc"]["function"] =0 #name of functions to be used for ensemblers
    

    print("true")
    print(y_true[:,1])

    print("pred")
    print(y_pred[:,1])

    score, config = mcc(y_true=y_true[:,1], y_pred=y_pred[:,1], parameters=pre["mcc"])

    print(score)
    print(config)
