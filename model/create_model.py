#TF-GPU IMPORTS
from numpy import zeros
from numpy.random import randn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
import numpy as np

# define the separate supervised and unsupervised discriminator models with shared weights:
# The models share all feature extraction layers, but one outputs one probability (so classification in real/fake image)
# and one outputs age predictions (1 output node and mse loss)
def define_discriminator(lr, in_shape=(64,64,1)):
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
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5))
    # supervised output
    c_out_layer = Dense(1, activation='linear')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['mae'])
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
def define_gan(g_model, d_model, lr):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=lr, beta_1=0.5)
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

#baseline model only training on supervised subsample
def define_baseline(lr, in_shape=(64,64,1)):
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
    b_out_layer = Dense(1, activation='linear')(fe)
    # define and compile supervised discriminator model
    b_model = Model(in_image, b_out_layer)
    b_model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['mae'])
    return b_model