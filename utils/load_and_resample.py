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
    :return: - Numpy array with all images that could be loaded of shape (target dimensions plus list dimension)
             - List of paths for which loading failed
    """
    #load reference image to which dimensions all other images will be resampled
    ref_img = load_img(ref_img)
    #extract target shape
    ref_shape = ref_img.shape

    #Find first image that can be loaded
    go_on = True
    i = 0
    fails = list()
    while(go_on):
        try:
            first_img = load_img(imgs[i])
            go_on = False
        except:
            fails.append(imgs[i])
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
            fails.append(img)
    #How many successful loads, how many fails?
    print("Successfully loaded {}/{} images, failed to load {}/{} images".format(succ, len(imgs), len(fails), len(imgs)))
    return(img_data, fails)

imgs, fails = load_and_resample(ref_img, list(q_selection.loc[q_selection.index[0:10], "scans"]))

#To get horizontal slices, we need to take slices from the 2nd dimension
#--> Swap axes, so that we have (x, y, slice, img number)
imgs = np.swapaxes(imgs,1,2)

def slicing(n_slices=5, imgs=imgs):
    """
    Takes a 4D-np-array of mri-images (x,y,slice_dimension,subjects) and takes n_slices from the third dimension.
    Returns a 3D-array of (x,y,n_slices*subjects), so n_slices*subjects 2D-images that are ordered like this:
    The first image is the first slice of the first subject. Then the first slice of the second subject and so on.
    Which image belongs to which subject is documented in a 1D-array, which is also returned.
    :param n_slices: How many slices we want. Taken from the middle.
    :param imgs: 4D-array of images with (x,y,slice_dimension,subjects)
    :return: - 3D-array of sliced images with (x,y,n_slices*subjects)
             - 1D-array of subject indices assigning a subject index to each 2D-image slice.
    """
    #ToDo: Parametrize the dimension which should be sliced. Idea: Give slice dimension as parameter. Swap axes so that
    #the slicing dimension is in the third place
    #take slices from the middle
    start_index = int(imgs.shape[2]/2) - int(n_slices/2)
    end_index = start_index + n_slices
    #slice
    imgs_sliced = imgs[:, :, start_index:end_index, :]
    #reshape into 3D-array so that all slices of all subjects are inside the 3rd dimension
    imgs_sliced = imgs_sliced.reshape(imgs_sliced.shape[0], imgs_sliced.shape[1], -1)
    #order: first slice of all subjects, then second slice of all subjects and so on
    #To understand why, see reshape_sliced_image_array.py
    #create array holding slice-person-mapping: each subject is assigned an index. The list of subjects is repeated
    #as many times as we have slices
    n_subjects = imgs.shape[-1]
    subject_idx = np.array([i for i in range(n_subjects)] * n_slices)
    return imgs_sliced, subject_idx




