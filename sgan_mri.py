#Select GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
#only use gpu with index 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from skimage.transform import resize
from utils.manipulation import split_data, shuffle_data
from model.train_model import run_cv
from utils.models import mean_model, best_model


###############
###LOAD DATA###
###############

imgs = np.load("data_mri/img_data.npy")
subject_idx = np.load("data_mri/subject_idx.npy")
targets = np.load("data_mri/targets.npy")

################################
###RESIZE IMAGES TO BE SQUARE###
################################

imgs_resized = resize(imgs, (64,64))


###########################
###ASSERT CORRECT SHAPES###
###########################

# img dimensions should be (samples, pixels, pixels), but are (pixels, pixels, samples) so swap axes accordingly
dataset = np.swapaxes(imgs_resized, 0, 2)
#also, reshape to (samples, pixels, pixels, 1) (expected by models)
dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)
targets = targets.reshape(targets.shape[0], 1)


##################
###SHUFFLE DATA###
##################

dataset, targets, subject_idx = shuffle_data(dataset, targets, subject_idx)


############################
###NOMRALIZE PIXEL VALUES###
############################

#gan will generate images with values from -1 to 1, so adjust mri images accordingly
range_mean = (np.max(dataset) - np.min(dataset)) / 2
dataset = (dataset - range_mean) / range_mean

#ToDo: Delete later, only for testing
#select only small amount of samples
#dataset = dataset[:200]
#targets = targets[:200]
#subject_idx = subject_idx[:200]


###############
###TEST DATA###
###############

print(dataset.shape)
print(targets.shape)
#Amount of data held back
test = 0.1
dataset_cv, targets_cv, subject_idx_cv, dataset_test, targets_test = split_data(test, dataset, targets, subject_idx)
print(dataset_cv.shape)
print(targets_cv.shape)
print(dataset_test.shape)
print(targets_test.shape)

################
###PARAMETERS###
################
# size of the latent space
latent_dim = 100
#name of the run
name = "Test1"
#number of folds
n_folds = 10
#number of epochs
n_epochs = 100
#learning rate
lr = 0.0002
#batch size: This also determines the amount of labeled data
n_batch = 100


######################
###CROSS-VALIDATION###
######################

#train the model using cross validation
c_model_trained, d_model_trained, g_model_trained, res_dir = run_cv(dataset_cv, targets_cv, subject_idx_cv, n_folds, range_mean, lr=lr, n_batch=n_batch, n_epochs=n_epochs, name=name, latent_dim=latent_dim)


############################
###FINAL MODEL EVALUATION###
############################

#Evaluate all models and return mean metric on test data
mean_model(res_dir, dataset_test, targets_test)

#Evaluate best model from folds on test data
best_model(res_dir, dataset_test, targets_test)

#ToDo: Test against other models (c_model without gan-structure and best brain age model (svr?))