import numpy as np
import warnings

#suppress Deprecation Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    def slicing(imgs, n_slices=5):
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