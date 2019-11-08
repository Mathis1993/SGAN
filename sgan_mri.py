import numpy as np
from skimage.transform import resize
from sklearn.model_selection import GroupKFold

###############
###LOAD DATA###
###############

imgs = np.load("img_data.npy")
subject_idx = np.load("subject_idx.npy")
targets = np.load("targets.npy")

################################
###RESIZE IMAGES TO BE SQUARE###
################################

imgs_resized = resize(imgs, (128,128))

##################
###SHUFFLE DATA###
##################

#get indices in shuffled form
shuffled_idx = np.random.randint(0, imgs_resized.shape[-1], imgs_resized.shape[-1])
#shuffle order of sliced images and subject indices in the same way
imgs_resized = imgs_resized[:,:,shuffled_idx]
subject_idx = subject_idx[shuffled_idx]
targets = targets[shuffled_idx]

############################################
###FOR NOW, ONLY USE TRAIN AND TEST SPLIT###
############################################
#ToDo: Add Cross-Validation with more groups later

group_kfold = GroupKFold(n_splits=10)
folds = group_kfold.split(np.swapaxes(imgs_resized, 0, 2), targets, subject_idx)

for j, (train_idx, val_idx) in enumerate(folds):
    print("TRAIN:", train_idx, "TEST:", val_idx)