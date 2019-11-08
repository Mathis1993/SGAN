import pandas as pd
import numpy as np
import warnings
from utils.load_and_resample import load_and_resample
from utils.slicing import slicing

#suppress Deprecation Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    ######################
    ###LOAD IMAGE PATHS###
    ######################

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

    #################
    ###LOAD IMAGES###
    #################

    #load 100 images
    imgs, fails, idx_loaded = load_and_resample(ref_img, list(q_selection.loc[q_selection.index[0:100], "scans"]))

    ###################################
    ###GET TARGETS FOR LOADED IMAGES###
    ###################################

    targets = list(q_selection.loc[q_selection.index[idx_loaded], "age"])

    ##################
    ###SLICE IMAGES###
    ##################

    #To get horizontal slices, we need to take slices from the 2nd dimension
    #--> Swap axes, so that we have (x, y, slice, img number), i.e. the dimension to be sliced is the third one
    #(slicing function expects this at the moment)
    imgs = np.swapaxes(imgs,1,2)

    n_slices = 5
    imgs_sliced, subject_idx = slicing(imgs=imgs, n_slices=n_slices)

    #update targets: Each slice has a corresponding age
    targets = np.array([target for target in targets] * n_slices)

    ###############
    ###SAVE DATA###
    ###############

    np.save("img_data.npy", imgs_sliced)
    np.save("subject_idx.npy", subject_idx)
    np.save("targets.npy", targets)
