import pandas as pd
import numpy as np
from nilearn.image import load_img, resample_to_img
import warnings

#suppress Deprecation Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#dataframe containing path and name to t1-images per subject and their ages
q = pd.read_csv("mathis_t1_FOR2107_HC_MDD.csv")

#make sure to only have cases with HC-diagnosis
q = q.loc[q["research_diagnosis"] == "HC", :]

#t1-images paths and names: append full image paths as new column
scans = list(q.loc[: ,"NeuroRaw_T1default__image_path"] + "/" + q.loc[: ,"NeuroRaw_T1default__image_name"])
q['scans'] = scans

#subset with relevant variables
q_selection = q.loc[:, ["MS_ID", "research_diagnosis", "scans", "age"]]

#Before shuffling, get path to an image that has target dimensions, to use as reference when loading the immages
ref_img = q_selection.loc[q_selection.index[0] , 'scans']

#Shuffle df
q_selection = q_selection.sample(frac=1)

def load_and_resample(ref_img, imgs):
    """
    Load mri-images iteratively and resampling them to the dimensions of a reference image.
    :param ref_img: (String) Path to reference image having target dimensions
    :param imgs: (List) Paths to all images (including the reference image)
    :return: Numpy array with all images that could be loaded of shape (target dimensions plus list dimension)
    """
    #load reference image to which dimensions all other images will be resampled
    ref_img = load_img(ref_img)
    #extract target shape
    ref_shape = ref_img.shape

    #Find first image that can be loaded
    go_on = True
    i = 0
    while(go_on):
        try:
            first_img = load_img(imgs[i])
            go_on = False
        except:
            i += 1

    #Right now, only able to load one image after another (see read_multiple_niftis.py)
    #Load first one
    first_img = load_img(imgs[i])
    #if necessary, resample to reference dimensions
    if first_img.shape != ref_shape:
        first_img = resample_to_img(source_img=first_img, target_img=ref_img, interpolation='nearest')
    #get image data as np array
    img_data = first_img.get_data()
    #add fourth dimension for concatenation
    img_data = np.expand_dims(img_data, axis=3)
    #Starting the loop from the image coming after the first one that could be loaded
    fails = i
    succ = 1
    for img in imgs[i+1:]:
        try:
            cur_img = load_img(img)
            if cur_img.shape != ref_shape:
                cur_img = resample_to_img(source_img=cur_img, target_img=ref_img, interpolation='nearest')
            cur_img = cur_img.get_data()
            # add fourth dimension for concatenation
            cur_img = np.expand_dims(cur_img, axis=3)
            #concatenate to previous image(s)
            img_data = np.concatenate((img_data, cur_img), axis=3)
            succ += 1
        except:
            fails += 1
    #How many successful loads, how many fails?
    print("Successfully loaded {}/{} images, failed to load {}/{} images".format(succ, len(imgs), fails, len(imgs)))
    return(img_data)
