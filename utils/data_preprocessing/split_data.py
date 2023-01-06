"""
Splits the dataset into simple, "simpleStratified", kfold or kfoldStratified

"""
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold
import numpy as np

def split_data(y, split_type, test_size=0.3, n_folds=5):
    """
    Split the datasets

    Parameters:
        y: array of labels and image ids with shape (num_images, 2)
        split_type: type of splitting ("simple", "simpleStratified", "kfold", "kfoldStratified")
        test_size: fraction of data for testing (default=0.3)
        n_folds: number of folds (default=5)
    Returns:
        train_labels: numpy array of shape(folds, num_images, 2)
        valid_labels: numpy array of shape(folds, num_images, 2)
    
    """

    if split_type=="simple":
       
        # shuffle data before split
        ss = ShuffleSplit(n_splits=1, test_size=test_size)

        for train_index, valid_index in ss.split(X=np.ones(len(y))):
            y_train = np.expand_dims(y[train_index], axis=0)
            y_valid = np.expand_dims(y[valid_index], axis=0)
    
    elif split_type=="simpleStratified":

        # shuffle data before split
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

        for train_index, valid_index in ss.split(X=np.ones(len(y)), y=y[:,1]):
            y_train = np.expand_dims(y[train_index], axis=0)
            y_valid = np.expand_dims(y[valid_index], axis=0)

    
    elif split_type=="kfold":

        # shuffle data before creating folds
        kf = KFold(n_splits=n_folds, shuffle=True)
        
        y_train = []
        y_valid = []
        for train_index, valid_index in kf.split(X=np.ones(len(y))):
            y_train.append(y[train_index])
            y_valid.append(y[valid_index])
        
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)

    
    elif split_type == "kfoldStratified":

        # shuffle data before creating folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        y_train = []
        y_valid = []
        
        for train_index, valid_index in skf.split(X=np.ones(len(y)), y=y[:,1]):

            y_train.append(y[train_index])
            y_valid.append(y[valid_index])

        y_train = np.array(y_train)
        y_valid = np.array(y_valid)

    else:
        raise ValueError(f"Unknown value encountered for the parameter 'split_type' during splitting of the data. received {split_type}")

    return y_train, y_valid

def get_batch(y, bs):
    """
    a python generator that yields batch of data

    Parameters:
        y: np.array of shape (num_images, 2) or (num_images, 1)
        bs: batch size

    Returns:
        batch of y of shape (num_images, 2) or (num_images, 1)

    """
    num_images= y.shape[0]

    for index in range(0, num_images, bs):
        yield y[index:min(index+bs, num_images)]
    

if __name__ == "__main__":

    import os

    path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train_multi.txt"
    #images = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/test"
    y=np.loadtxt(path_to_labels, dtype=str, delimiter=" ")
    
    #y = os.listdir(images)
   #y = np.array(y)
    #y = np.expand_dims(y, axis=-1)
    #y= np.sort(y, axis=0)
    y =np.concatenate((y, y))
    print(y)
    print(y.shape)

    #split_type= "simpleStratified" #, "simple", "simpleStratified", "kfold", "kfoldStratified"

    #train, valid = split_data(y, split_type)

    #print(train.shape)
    #print(valid.shape)

    #for i in range(train.shape[0]):
    #    print("Fold NO ", i)
    #    print(train[i])
    #    print(valid[i])
    t = 0
    for x in get_batch(y, 5):
        print("call ", t)
        print(x)
        print(x.shape)
        t = t+1