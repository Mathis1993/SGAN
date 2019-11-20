from nilearn.image import load_img, resample_to_img
import numpy as np
import warnings

#suppress Deprecation Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    def load_and_resample(ref_img, imgs):
        """
        Load mri-images iteratively and resampling them to the dimensions of a reference image.
        :param ref_img: (String) Path to reference image having target dimensions
        :param imgs: (List) Paths to all images (including the reference image)
        :return: - Numpy array with all images that could be loaded of shape (target dimensions plus list dimension)
                 - List of paths for which loading failed
                 - Indices of imgs-list (first param) that could be loaded to be able to match targets to them
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
        imgs_loaded = list()
        idx_loaded = list()
        first_img = load_img(imgs[i])
        idx_loaded.append(i)
        #if necessary, resample to reference dimensions
        if first_img.shape != ref_shape:
            first_img = resample_to_img(source_img=first_img, target_img=ref_img, interpolation='nearest')
        #get image data as np array
        img_data = first_img.get_data()
        #add fourth dimension for concatenation
        img_data = np.expand_dims(img_data, axis=3)
        img_data = np.swapaxes(img_data, 0, 3)
        imgs_loaded.append(img_data)
        #Starting the loop from the image coming after the first one that could be loaded
        succ = 1
        for j in range(i+1,len(imgs)):
            try:
                cur_img = load_img(imgs[j])
                if cur_img.shape != ref_shape:
                    cur_img = resample_to_img(source_img=cur_img, target_img=ref_img, interpolation='nearest')
                cur_img = cur_img.get_data()
                # add fourth dimension for concatenation
                cur_img = np.expand_dims(cur_img, axis=3)
                #for stacking later
                cur_img = np.swapaxes(cur_img, 0, 3)
                #concatenate to previous image(s)
                #img_data = np.concatenate((img_data, cur_img), axis=3)
                imgs_loaded.append(cur_img)
                idx_loaded.append(j+i)
                succ += 1
                print("Loaded image {} of {}".format(j, len(imgs)))
            except:
                fails.append(imgs[j])

        #stack from list into one array (along first axis)
        img_data = np.vstack(imgs_loaded)
        #swap axes again so that everything else works as expected
        img_data = np.swapaxes(img_data, 0, 3)

        #How many successful loads, how many fails?
        print("Successfully loaded {}/{} images, failed to load {}/{} images".format(succ, len(imgs), len(fails), len(imgs)))
        return(img_data, fails, idx_loaded)


    def slice(imgs, n_slices=5):
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