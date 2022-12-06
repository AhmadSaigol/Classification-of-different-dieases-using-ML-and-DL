"""
For processing ylabels

"""
import numpy as np
from sklearn import preprocessing

def label_encoder(y, classes, to_numbers):
    """
    encodes target labels with value between 0 and n_classes-1 and vice versa


    Parameter:
        y: labels of numpy array (num_images,)
        classes: numpy array of names of classes
        to_numbers True/False: whether to transfer from string to numbers or vice versa

    Returns:
        result: encoded labels of numpy array (num_images, 1)
                or labels of numpy array (num_images, 1)
    """

    le = preprocessing.LabelEncoder()
    le.classes_ = classes
    if to_numbers:
        return le.transform(y.ravel())
    else:
        return le.inverse_transform(y.ravel().astype(int)) 



if __name__ == "__main__":
    a = np.array(["a", "a", "a", "a"])
    print(a)
    print(a.shape)
    classes = np.array(["a", "b"])

    y = label_encoder(a, classes, True)
    print(y.shape)
    print(y)
    print(y.dtype)

    y1 = label_encoder(y, classes, False)
    print(y1.shape)
    print(y1)