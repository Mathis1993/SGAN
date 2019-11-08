import pandas as pd
from nilearn.image import load_img, smooth_img

#dataframe containing path and name to t1-images per subject and their ages
q = pd.read_csv("mathis_t1_FOR2107_HC_MDD.csv")

#make sure to only have cases with HC-diagnosis
q = q.loc[q["research_diagnosis"] == "HC"]

#t1-image path and name
q.loc[q.index[0],"NeuroRaw_T1default__image_path"] + "/" + q.loc[q.index[0],"NeuroRaw_T1default__image_name"]

#most images are 256x256x172
im = load_img(q.loc[q.index[0],"NeuroRaw_T1default__image_path"] + "/" + q.loc[q.index[0],"NeuroRaw_T1default__image_name"])
print(im.shape)

scans = list(q.loc[q.index[:],"NeuroRaw_T1default__image_path"] + "/" + q.loc[q.index[:],"NeuroRaw_T1default__image_name"])
print(len(scans))

#how many different sizes (in the third dimension) are there
errors = list()
sizes = list()
#sizes.append(176)
for i in range(len(scans)):
    #some images i got the path to don't seem to exist
    try:
        img = load_img(scans[i])
        #if(img.shape[2]) != 176:
        sizes.append(img.shape[2])
    except:
        errors.append(i)

#try to load one of the problematic images
#load_img(scans[errors[0]])

sizes_unique = set(sizes)
print(sizes_unique)
import numpy as np
for size in sizes_unique:
    print(size)
    print(np.array(sizes)[np.array(sizes)==size].shape)


#example of 176
load_img(scans[0]).shape
#example of 192
load_img(scans[450]).shape
#example of 180
load_img(scans[879]).shape