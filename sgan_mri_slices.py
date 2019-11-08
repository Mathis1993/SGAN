import pandas as pd
import numpy as np
import warnings
from utils.load_and_resample import load_and_resample
from utils.slicing import slicing

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

#load 10 images
imgs, fails = load_and_resample(ref_img, list(q_selection.loc[q_selection.index[0:10], "scans"]))

#To get horizontal slices, we need to take slices from the 2nd dimension
#--> Swap axes, so that we have (x, y, slice, img number)
imgs = np.swapaxes(imgs,1,2)

imgs_sliced = slicing(imgs=imgs, n_slices=5)
