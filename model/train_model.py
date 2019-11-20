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
from numpy import ones
from utils.results import mk_result_dir
from model.create_model import select_samples, generate_fake_samples, generate_latent_points, define_discriminator, define_gan, define_generator
from model.evaluate_model import evaluate_performance
from sklearn.model_selection import GroupKFold
import numpy as np

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

def run_cv(dataset, targets, subject_idx, n_folds, name, latent_dim):
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
        train_dataset = dataset[train_idx]
        train_targets = targets[train_idx]
        val_dataset = dataset[val_idx]
        val_targets = targets[val_idx]
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