#ToDo: Uncomment later
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
# #only use gpu with index 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from skimage.transform import resize
from sklearn.model_selection import GroupKFold
from utils.manipulation import split_data, shuffle_data

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

##################
###SHUFFLE DATA###
##################

imgs_resized, targets, subject_idx = shuffle_data(imgs_resized, targets, subject_idx)


###########################
###ASSERT CORRECT SHAPES###
###########################

# img dimensions should be (samples, pixels, pixels), but are (pixels, pixels, samples) so swap axes accordingly
dataset = np.swapaxes(imgs_resized, 0, 2)
#also, reshape to (samples, pixels, pixels, 1) (expected by models)
dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)
targets = targets.reshape(targets.shape[0], 1)


############################
###NOMRALIZE PIXEL VALUES###
############################

#gan will generate images with values from -1 to 1, so adjust mri images accordingly
dataset = (dataset - ((np.min(dataset) + np.max(dataset)) / 2)) / ((np.min(dataset) + np.max(dataset)) / 2)

#ToDo: Delete later, only for testing
#select only small amount of samples
dataset = dataset[:200]
targets = targets[:200]
subject_idx = subject_idx[:200]


###########
###MODEL###
###########

#TF-GPU IMPORTS
# from numpy import expand_dims, zeros, ones, asarray
# from numpy.random import randn, randint
# from tensorflow import keras
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
# from matplotlib import pyplot
# from utils.mk_result_dir import mk_result_dir
# from utils.save_model import save_model
# import csv
# from utils.save_csv import save_csv
# from utils.plotting import plot_val_train_loss, plot_acc

#TF-CPU IMPORTS
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
#from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from utils.results import mk_result_dir, save_csv
from utils.models import save_model, mean_model, best_model
from utils.plotting import plot_val_train_loss, plot_acc

# define the separate supervised and unsupervised discriminator models with shared weights:
# The models share all feature extraction layers, but one outputs one probability (so classification in real/fake image)
# and one outputs age predictions (1 output node and mse loss)
def define_discriminator(in_shape=(64,64,1)):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # unsupervised output
    d_out_layer = Dense(1, activation='sigmoid')(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    # supervised output
    c_out_layer = Dense(1, activation='linear')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['mae'])
    return d_model, c_model

# define the standalone generator model: A vector of points from latent space (eg 100 numbers drawn from the normal
# distribution) are upsampled to a 7x7 image with 128 feature channels(dense layer and reshaping),
# then upsampled again to 14x14 (Transpose Reverse Convolution),
# then upsampled againg to 28x28 (Transpose Reverse Convolution)
# then feature channels are collapsed to 1 channel (we want to generate a black and white image) with a tanh-activation
# so that we get pixel values between -1 and 1
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 8x8 image
    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8, 8, 128))(gen)
    # upsample to 16x16
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 32x32
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 64*64
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 128*128
    #gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    #gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator:
# We logically combine the generator and the fake/real-discriminator, so that we can update the generator
# based on the fakre/real-discriminator's output. To achieve that the generator is getting better at generating
# images that seem real, we give all generated fake images a label of 1 (true, real image) when feeding them to the
# fake/real-discriminator. The discriminator will then output a low probability of the image being real, but the label
# says it is real, so there is a big parameter update.
# The parameter update only applies to the weights/biases of the generator, as we marked the weights of
# the fake/real-generator as not trainable inside this function (does not affect training the fake/real-discriminator
# when training it from it's standalone model).
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#use labels when needed, otherwise ignore them
def select_samples(data, targets, n_samples=100):
    X_list, y_list = list(), list()
    #choose random instances
    ix = np.array([x for x in range(data.shape[0])])
    np.random.shuffle(ix)
    ix = ix[0:n_samples]
    # add to list
    [X_list.append(data[j,:,:]) for j in ix]
    [y_list.append(targets[j]) for j in ix]
    #also return idx used, so that we now which ones shouldn't appear as unlabeled data right now
    return np.asarray(X_list), np.asarray(y_list), ix

# generate points in latent space as input for the generator
# Eg 100 random values as often as we have samples in a batch
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

# use the generator to generate n fake examples, with class labels
# Using the generation of random input for the generator, get the generators output with the y-label (images real or not)
# being 0, as the images are fake (Although when feeding those to the gan_model used to train the generator, we will make the
# labels to 1. But when training the standalone fake/real-discriminator with the fake images, the labels will be left as they are
# --> The discriminator is trained with real and fake images, knowing which are real and which are fake to get good at discriminating these
# two types of images. The gan-model (to train the generator) will be trained by giving fake images (generated by the generator, but labeled as real!) to the
# fake/real-discriminator and being updated according to the discriminator's performance --> The generator is updated strongly when the discriminator reveals his
# generated images as fake and weakly when the discriminator believes his generated images to be real)
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y


def evaluate_performance(fold, path, metric, prev_metric, epoch_list, c_losses_train , c_losses_val, metrics, c_model, d_model, g_model):
    path_res_dir = path
    path_sub_dir = path + "/" + "fold_{}".format(fold)
    #save to csv (overwrite every epoch, so that if the script breaks down during training, the results up to that
    #point will be saved)
    save_csv(path=path_sub_dir, name="metric_fold_{}".format(fold), mode="w", epoch_list=epoch_list, c_losses_train=c_losses_train, c_losses_val=c_losses_val, metrics=metrics)
    # decide if to save the models because validation accuracy improved
    if (metric < prev_metric) | (prev_metric == 0.0):
        print("Validation metric improved from [{:.3f}] to [{:.3f}]. Saving models...".format(prev_metric, metric))
        save_model(c_model, path_sub_dir, 'c_model_fold_{}.h5'.format(fold))
        save_model(d_model, path_sub_dir, 'd_model_fold_{}.h5'.format(fold))
        save_model(g_model, path_sub_dir, 'g_model_fold_{}.h5'.format(fold))
        #Save current metric to know how the saved model performed
        save_csv(path_res_dir, "best_val_metrics", mode="a", fold=[fold], best_val=[metric])
        #np.save(path + "/" + "best_val_metric_fold_{}".format(fold), np.array((fold,metric)))
        prev_metric = metric
    plot_val_train_loss(c_losses_train, c_losses_val, fold, path_sub_dir)
    plot_acc(metrics, fold, path_sub_dir)
    return(prev_metric)

    #ToDo: Create fake images and save them for viusal inspection
	# prepare fake examples
	# X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# # ToDo: rescale from [-1,1] to former range
	# X = (X + 1) / 2.0
	# # plot images
	# for i in range(100):
	# 	# define subplot
	# 	pyplot.subplot(10, 10, 1 + i)
	# 	# turn off axis
	# 	pyplot.axis('off')
	# 	# plot raw pixel data
	# 	pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# # save plot to file
	# filename1 = 'generated_plot_%04d.png' % (step+1)
	# pyplot.savefig(filename1)
	# pyplot.close()

def train(fold, res_dir, g_model, d_model, c_model, gan_model, train_dataset, train_targets, val_dataset, val_targets, latent_dim, n_epochs=20, n_batch=100):
    # select supervised dataset
    X_sup, y_sup, ix_sup = select_samples(train_dataset, train_targets, n_samples=n_batch)
    print("Supvervised Samples' Shape: {}, Supervised Targets' Shape: {}".format(X_sup.shape, y_sup.shape))
    # calculate the number of batches per training epoch
    bat_per_epo = int(train_dataset.shape[0] / n_batch) #round up?
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    #dataset to draw real samples (unlabeled) from, excluding the indices of the supervised sample
    dataset_real = np.delete(train_dataset, (ix_sup), axis=0)
    print('fold=%d, n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (fold + 1, n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    prev_metric = 0.0
    c_losses_train = list()
    c_losses_val = list()
    metrics = list()
    epoch_list = list()
    for j in range(n_epochs):
        epoch_list.append(j)
        for i in range(bat_per_epo):
            #randomly select half of the supervised samples (just reuse select_supervised_samples with n_samples=50, sampling from the 100 supervised ones)
            X_sup_real, y_sup_real, _ = select_samples(X_sup, y_sup, n_samples=half_batch)
            #update supervised discriminator (c)
            c_loss, c_metric = c_model.train_on_batch(X_sup_real, y_sup_real)
            #randomly select real (unsupervised) samples
            #Here, we need a 1 as a label
            X_real, _, _ = select_samples(dataset_real, train_targets, n_samples=half_batch)
            y_real = np.ones((half_batch, 1))
            # generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #update unsupervised discriminator (d)
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # update generator (g)
            # Here, fake images are labeled as real!
            X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('-->fold %d, epoch %d, batch %d/%d, c[%.3f, %.3f], d[%.3f, %.3f], g[%.3f]' % (fold + 1, j + 1, i + 1, bat_per_epo, c_loss, c_metric, d_loss1, d_loss2, g_loss))
        #after each epoch: save current losses and val metric
        c_losses_train.append(c_loss)
        c_loss_val, metric = c_model.evaluate(val_dataset, val_targets, verbose=0)
        c_losses_val.append(c_loss_val)
        metrics.append(metric)
        #evaluate performance
        path = res_dir
        prev_metric = evaluate_performance(fold, path, metric, prev_metric, epoch_list, c_losses_train , c_losses_val, metrics, c_model, d_model, g_model)
    return c_model, d_model, g_model

def run_cv(dataset, targets, subject_idx, n_folds, name):
    #folds
    group_kfold = GroupKFold(n_splits=n_folds)
    folds = group_kfold.split(dataset, targets, subject_idx)
    #create results dir
    dir_name = mk_result_dir(name, n_folds)
    fold_accuracy = list()
    fold = 0
    for j, (train_idx, val_idx) in enumerate(folds):
        #print("TRAIN:", train_idx, "VAL:", val_idx)
        #select current data
        train_dataset = dataset_cv[train_idx]
        train_targets = targets_cv[train_idx]
        val_dataset = dataset_cv[val_idx]
        val_targets = targets_cv[val_idx]
        #model instantiation
        # create the discriminator models
        d_model, c_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        n_gen_trainable = len(g_model.trainable_weights)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # train models
        c_model_trained, d_model_trained, g_model_trained = train(fold, dir_name, g_model, d_model, c_model, gan_model, train_dataset, train_targets, val_dataset, val_targets, latent_dim, n_epochs=10, n_batch=100)
        fold+=1
    return(c_model_trained, d_model_trained, g_model_trained, dir_name)


###############
###TEST DATA###
###############
#ToDo: Try out whole script with a small number of examples: Does the syntax work without throwing errors?
#ToDo: Code for starting tmux session?

#Amount of data held back
test = 0.1
dataset_cv, targets_cv, subject_idx_cv, dataset_test, targets_test = split_data(test, dataset, targets, subject_idx)


################
###PARAMETERS###
################
# size of the latent space
latent_dim = 100
#name of the run
name = "Run1"
#number of folds
n_folds = 10


######################
###CROSS-VALIDATION###
######################

c_model_trained, d_model_trained, g_model_trained, res_dir = run_cv(dataset_cv, targets_cv, subject_idx_cv, n_folds, name="Run1")


############################
###FINAL MODEL EVALUATION###
############################

#Evaluate all models and return mean metric on test data
mean_model(res_dir, dataset_test, targets_test)

#Evaluate best model from folds on test data
best_model(res_dir, dataset_test, targets_test)

#ToDo: Test against other models (c_model without gan-structure and best brain age model (svr?))