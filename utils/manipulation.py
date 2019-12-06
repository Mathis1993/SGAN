import numpy as np
import pandas as pd

def shuffle_data(imgs, targets, subject_idx):
    #get indices in shuffled form
    random_idx = np.array([x for x in range(imgs.shape[0])])
    np.random.shuffle(random_idx)
    #shuffle order of sliced images and subject indices in the same way
    imgs = imgs[random_idx, :, :]
    subject_idx = subject_idx[random_idx]
    targets = targets[random_idx]
    return(imgs, targets, subject_idx)


def split_data(test_amount, dataset, targets, subject_idx):
    # Hold back some data to use as evaluation for the best cross-validated model
    # We have to select all slices per person

    #amount of subjects
    number_subjects = np.unique(subject_idx).shape[0]
    test_subjects = int(test_amount*number_subjects)
    #take the first n==test_subjects unique indices
    #pandas' unique is faster than nps' unique and doesn't sort the output
    unique_idx = pd.unique(subject_idx)[:test_subjects]

    #take slices from the first n unique subjects, corresponding to an amount of data == test (eg 0.1)
    subject_idx_test = list()
    for i in range(test_subjects):
        idx = np.argwhere(subject_idx == unique_idx[i])
        subject_idx_test.append(idx)
    #collapse into one array
    subject_idx_test = np.concatenate(subject_idx_test, axis=0)
    #squeeze into 1D
    subject_idx_test = np.squeeze(subject_idx_test)

    #select test data
    dataset_test = dataset[subject_idx_test]
    targets_test = targets[subject_idx_test]

    #shuffle test data
    dataset_test, targets_test, subject_idx_test = shuffle_data(dataset_test, targets_test, subject_idx_test)

    #omit test indices from the rest of the data used for cross validation
    dataset_cv = np.delete(dataset, subject_idx_test, axis=0)
    targets_cv = np.delete(targets, subject_idx_test, axis=0)
    subject_idx_cv = np.delete(subject_idx, subject_idx_test, axis=0)

    return(dataset_cv, targets_cv, subject_idx_cv, dataset_test, targets_test)