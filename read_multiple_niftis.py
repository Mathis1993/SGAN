import pandas as pd
from photonai.base.PhotonBase import PipelineElement
import numpy as np
from nilearn.image import load_img
from nilearn._utils import check_niimg

#dataframe containing path and name to t1-images per subject and their ages
q = pd.read_csv("mathis_t1_FOR2107_HC_MDD.csv")

#make sure to only have cases with HC-diagnosis
q = q.loc[q["research_diagnosis"] == "HC", :]

#t1-images paths and names: append full image paths as new column
scans = list(q.loc[: ,"NeuroRaw_T1default__image_path"] + "/" + q.loc[: ,"NeuroRaw_T1default__image_name"])
q['scans'] = scans

#subset with relevant variables
q_selection = q.loc[:, ["MS_ID", "research_diagnosis", "scans", "age"]]

#PHOTON
#Path to two images
#X = np.array(scans[0:2])
#mask = PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter',
#                          extract_mode='vec', batch_size=2)
#X, _, _ = mask.transform(X)

#NILEARN
for path in scans[0:2]:
    print(path)
img = check_niimg(scans[0:2], atleast_4d=True)
#img = load_img(scans[0:2])
#img2 = load_img(scans[1])
print(img.shape)
#print(img2.shape)