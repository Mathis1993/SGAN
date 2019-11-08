#https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
# example of semi-supervised gan for mnist
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from matplotlib import pyplot

# define the separate supervised and unsupervised discriminator models with shared weights:
# The models share all feature extraction layers, but one outputs one probability (so classification in real/fake image)
# and one outputs class probabilities (10 classes for digits 0 to 9)
def define_discriminator(in_shape=(28,28,1), n_classes=10):
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
	c_out_layer = Dense(n_classes, activation='softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
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
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
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

# load the images and normalize their pixel-values to be inside [-1,1] to fit to
# what the generator is outputting.
def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	print(X.shape, trainy.shape)
	return [X, trainy]

# select a supervised subset of the dataset, ensures classes are balanced
# This selects the small subsample of labeled examples (If we really had a dataset where a small
# par is labeled and the rest is not, we wouldn't need to artificially only select the labels of some samples
# and ignore the rest of the labels treating the rest of the examples as unlabeled as we do here).
def select_supervised_samples(dataset, n_samples=100, n_classes=10):
	X, y = dataset
	X_list, y_list = list(), list()
	n_per_class = int(n_samples / n_classes)
	for i in range(n_classes):
		# get all images for this class
		X_with_class = X[y == i]
		# choose random instances
		ix = randint(0, len(X_with_class), n_per_class)
		# add to list
		[X_list.append(X_with_class[j]) for j in ix]
		[y_list.append(i) for j in ix]
	return asarray(X_list), asarray(y_list)

# select real samples
# This chooses the data per batch we treat as unlabeled. The label is extracted here, but when calling the function
# inside the training loop, we will just ignore the class labels and only use the data and the y-label (images real or not)
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

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

# generate samples and save as a plot and save the model
# Evaluation of generator: Visually inspect how real the fake images look
# Evaluation of classifier model (class-discriminator): Performance over whole dataset (Although it saw a small fraction
# of this during training?)
# fake/real-discriminator is not evaluated alone?
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	# save the generator model
	filename2 = 'g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
	# save the classifier model
	filename3 = 'c_model_%04d.h5' % (step+1)
	c_model.save(filename3)
	print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

#There are three models:
# - Fake/real-discriminator (the discriminator)
# - Classes-discriminator (the classifier) sharing everything but the outputlayer with the discriminator
# - Generator
# - As a logical fourth model (linking generator and discriminator) the gan_model to train the generator that is only
# logical and not really a foruth physical model
# Train the generator and discriminator(s)
# 1. Determine what part of the dataset will be used as labeld samples (supervised dataset)
# 2. How many batches make up one epoch?
# 3. Number of batche per epoch times number of epochs --> total number of steps
# 4. how many samples are inside half a batch?
# 5. Take a half batch of the supervised samples (So half of the supervised samples, as we use as many supervised samples
# as there are examples in one batch; here: 100) and train the classifier on them
# 6. Take half a batch of unsupervised (unlabeled) samples and train the discriminator on them
# 7. Take half a batch of fake samples and train the discriminator on them
# 8. Take another half batch of fake samples and train the generator on them (so use the gan_model)
# 9. Print loss for the classifier, for the discriminator for real images and fake images seperately and the loss for
# the generator (which really is the loss of the discriminator on fake images he was told are real)
# 10. Call the evaluation function when one epoch is finished
# --> We are training the classifier with real, labeled images (labels 0-9 for the digits). Then, we are training the
# discriminator with real images (labeled as real) and fake images (labeled as fake), which updates the same feature
# extraction layers used by the classifier. So we incorporate the information the discriminator is learning about how to
# tell if an example is real or not (so is a real example of the domain we are interested in) into the classifier model we
# really want to be good. In order to make this additional information valuable, we train the generator the generate
# better (more real-seeming) fake images, because otherwise the discriminator would only have to differentiate between
# real images and fake ones that are completely random, which wouldn't add much insight.
# By generating better and better (more real-seeming) fake images, the generator learns to give meaning to the latent
# feature space that was random at first until it represents a compressed representation of the output space, the mnist
# images, that the generator can turn into plausible images from this domain
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=100):
	# select supervised dataset
	X_sup, y_sup = select_supervised_samples(dataset)
	print(X_sup.shape, y_sup.shape)
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
	# manually enumerate epochs
	for i in range(n_steps):
		# update supervised discriminator (c)
		[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
		c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
		# update unsupervised discriminator (d)
		[X_real, _], y_real = generate_real_samples(dataset, half_batch)
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update generator (g)
		X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
		print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model, c_model, latent_dim, dataset)

# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, c_model, gan_model, dataset, latent_dim)