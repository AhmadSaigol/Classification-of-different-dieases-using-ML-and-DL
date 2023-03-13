"""
study texture properties after applying wavelet transformation

"""
import numpy as np
import cv2
import pywt
from skimage.feature import graycomatrix, graycoprops
from skimage import util, exposure

def feature_GLCM(X, parameters):
    """
    This function performs three steps to an image:
        - Apply 2D Discrete wavelet transformation
        - calculate gray level co-occurence matrix
        - calculate texture properties of GLCM which includes:
                (energy,correlation,dissimilarity, homogeneity, contrast, ASM)
    
    Parameters:
        X: numpy array of shape (num_images, H, W, C)
        parameters: dictionary with following keys:
                'wavelet_type': default = 'haar'
           
    
    Returns:
        features: numpy array of shape (num_images, num_features)
        config: dictionary with parameters of the function (including default parameters)

    Additional Notes:
        - currently supports grayscale images only

    """
    config = {}

    # make sure grayscale image is given
    if X.shape[-1] != 1:
        raise ValueError("Currently, calculating GCLM is only supported for grayscale images")
    else:
        X=np.squeeze(X, axis=-1)

    num_images = X.shape[0]

    # get wavelet type
    if 'wavelet_type' in parameters.keys():
        wt = parameters['wavelet_type']
    else:
        wt = 'haar'
    config['wavelet_type'] = wt
    
    features = []
    for img in range(num_images):
        
        # 2D Discete wavelet transform
        LL, (LH, HL, HH) = pywt.dwt2(X[img], wt)

        feature = []   
        for n ,img in enumerate([LL,LH,HL,HH]):
            
            img = exposure.rescale_intensity(img, out_range=(0, 1))
            
            bin_width = 32
            im = util.img_as_ubyte(img)
            img = im//bin_width
            
            #calculate gray level co-occurence matrix
            GLCM = graycomatrix(img, [1], [0])

            #calculate texture properties of GLCM       
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            GLCM_ASM = graycoprops(GLCM, 'ASM')[0]
            
            #print([GLCM_Energy, GLCM_corr, GLCM_diss, GLCM_hom, GLCM_contr, GLCM_ASM])

            feature.extend([GLCM_Energy, GLCM_corr, GLCM_diss, GLCM_hom, GLCM_contr, GLCM_ASM])

        features.append(feature)


    features = np.squeeze(np.array(features), axis=-1)

    return features, config


if __name__ == "__main__":

    path_to_image = "/home/ahmad/Pictures/Screenshot from 2022-11-22 12-50-59.png"
    image_c =cv2.imread(path_to_image, 0)

    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    image_c = np.concatenate((image_c, image_c*0.4))
    
    print(image_c.shape) 
    pipeline={}
    pipeline["GLCM"] = {}
    pipeline["GLCM"]["function"] =0 #some function pointer
    #pipeline["GLCM"]["wavelet_type"] = 'bior1.3'
    
    new, config = feature_GLCM(image_c, parameters=pipeline["GLCM"])
    print(new)
    print(new.shape, config)

"""
    image_c = np.expand_dims(image_c, axis=0)
    image_c = np.expand_dims(image_c, axis=-1)
    image_c = np.concatenate((image_c, image_c*0.4))
    
    pipeline={}
    pipeline["skewness"] = {}
    pipeline["skewness"]["function"] =0 #some function pointer
    pipeline["skewness"]["bias"] = False


    print(image_c.shape)
    cv2.imshow("original", image_c[0])
    cv2.waitKey(0)    

    new, config = calculate_skew(image_c, parameters=pipeline["skewness"])
    
    print(new.shape)
    print(new)
    cv2.imshow("a", new[0])
    cv2.waitKey(0)
    print(config)"""