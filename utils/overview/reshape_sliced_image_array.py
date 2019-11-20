import pandas as pd
import numpy as np
import warnings
from utils.load_and_resample import load_and_resample
from matplotlib import pyplot as plt

#suppress Deprecation Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#READ DATA
#dataframe containing path and name to t1-images per subject and their ages
q = pd.read_csv("mathis_t1_FOR2107_HC_MDD.csv")
#make sure to only have cases with HC-diagnosis
q = q.loc[q["research_diagnosis"] == "HC", :]
#t1-images paths and names: append full image paths as new column
scans = list(q.loc[: ,"NeuroRaw_T1default__image_path"] + "/" + q.loc[: ,"NeuroRaw_T1default__image_name"])
ref_img = scans[0]

#load images
imgs, fails = load_and_resample(ref_img, scans[0:10])

#Get slices
#To get horizontal slices, we need to take slices from the 2nd dimension
#--> Swap axes, so that we have (x, y, slice, img number)
imgs = np.swapaxes(imgs,1,2)
#amount of slices (taken from the middle)
n_slices = 5
start_index = int(imgs.shape[2]/2) - int(n_slices/2)
end_index = start_index + n_slices
imgs_sliced = imgs[:, :, start_index:end_index, :]
#get all slices over all subjects into one dimension
imgs_sliced = imgs_sliced.reshape(256,176,-1)

#Now we have 5x10=50 slices, but in what oder? Where are slices 1 to 5 from subject 1, and where are 1 to 5 from
#subject 2 and so on?

#From reshape/ravel documentation:
#"In row-major, C-style order, in two dimensions, the row index varies the slowest, and the column index the quickest.
#This can be generalized to multiple dimensions, where row-major order implies that the index along the first axis
#varies slowest, and the index along the last quickest."
# --> When reshaping (256, 176, 5, 10) (5 slices of 256x176 over 10 subjects),  to (256, 176, 50), the subject-index
#will vary quickest, so that the first ten entries of the third dimension in (256, 176, 50) will be the first slice
#for each of the ten subjects, the second ten entries will be the second slice for each of the ten subjects and so on

#Confirmation:
#slice 126 from the first subject (after swapping dimensions from (256,256,176) to (256,176,256)
plt.clf()
plt.imshow(imgs[:,:,126,0])
plt.savefig("after_swapping.png")
#slice 126 from the first subject after resizing the array so that all slices over all subjects are in one dimension
plt.clf()
plt.imshow(imgs_sliced.reshape(256,176,-1)[:,:,0])
plt.savefig("after_reslicing.png")

#slice 126 from the second subject (after swapping dimensions from (256,256,176) to (256,176,256)
plt.clf()
plt.imshow(imgs[:,:,126,1])
plt.savefig("after_swapping2.png")
#slice 126 from the second subject after resizing the array so that all slices over all subjects are in one dimension
plt.clf()
plt.imshow(imgs_sliced.reshape(256,176,-1)[:,:,1])
plt.savefig("after_reslicing2.png")

#after_swapping and after_reslicing as well as after_swapping2 and after_reslicing2 should be totally identical!

#If we would swap the subject and slice axes, would we get all 5 slices of subject 1 for the first five entries, then
#all 5 slices of subject two for the second 5 entries and so on?