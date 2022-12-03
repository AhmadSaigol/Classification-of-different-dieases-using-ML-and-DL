"""
Splits the dataset into simple, "simpleStratified", kfold or kfoldStratified

"""
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold
import numpy as np

def split_data(y, split_type):
    """
    Split the datasets

    Parameters:
        y: array of labels and image ids with shape (num_images, 2)
        split_type: type of splitting ("simple", "simpleStratified", "kfold", "kfoldStratified")
        
    Returns:
        train_labels: numpy array of shape(folds, num_images, 2)
        valid_labels: numpy array of shape(folds, num_images, 2)
    
    """

    flag = True

    if split_type=="simple":
        
        test_size= 0.3

        # shuffle data before split
        ss = ShuffleSplit(n_splits=1, test_size=test_size)

        for train_index, valid_index in ss.split(X=np.ones(len(y))):
            y_train = np.expand_dims(y[train_index], axis=0)
            y_valid = np.expand_dims(y[valid_index], axis=0)
    
    elif split_type=="simpleStratified":
        
        test_size= 0.3

        # shuffle data before split
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

        for train_index, valid_index in ss.split(X=np.ones(len(y)), y=y[:,1]):
            y_train = np.expand_dims(y[train_index], axis=0)
            y_valid = np.expand_dims(y[valid_index], axis=0)

    
    elif split_type=="kfold":

        n_folds = 5

        # shuffle data before creating folds
        kf = KFold(n_splits=n_folds, shuffle=True)

        for train_index, valid_index in kf.split(X=np.ones(len(y))):
            train_labels = np.expand_dims(y[train_index], axis=0)
            valid_labels = np.expand_dims(y[valid_index], axis=0)
            
            if flag:
                y_train = train_labels
                y_valid = valid_labels
                flag=False
            else:
                y_train = np.concatenate((y_train, train_labels))
                y_valid = np.concatenate((y_valid, valid_labels))


    
    elif split_type == "kfoldStratified":
        
        n_folds = 5

        # shuffle data before creating folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for train_index, valid_index in skf.split(X=np.ones(len(y)), y=y[:,1]):

            train_labels = np.expand_dims(y[train_index], axis=0)
            valid_labels = np.expand_dims(y[valid_index], axis=0)
            
            if flag:
                y_train = train_labels
                y_valid = valid_labels
                flag=False
            else:
                y_train = np.concatenate((y_train, train_labels))
                y_valid = np.concatenate((y_valid, valid_labels))

    else:
        raise ValueError(f"Unknown value encountered for the parameter 'split_type' during splitting of the data. received {split_type}")

    return y_train, y_valid

    

if __name__ == "__main__":

    path_to_labels = "/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train.txt"
    y=np.loadtxt(path_to_labels, dtype=str, delimiter=" ")
    y= np.sort(y, axis=0)
    y = np.concatenate((y, y))
    print(y)


    split_type= "simpleStratified" #, "simple", "simpleStratified", "kfold", "kfoldStratified"

    train, valid = split_data(y, split_type)

    print(train.shape)
    print(valid.shape)

    for i in range(train.shape[0]):
        print("Fold NO ", i)
        print(train[i])
        print(valid[i])