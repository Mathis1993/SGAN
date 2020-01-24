from numpy import ones
from utils.results import mk_result_dir
from utils.manipulation import split_data
from model.create_model import select_samples, generate_fake_samples, generate_latent_points, define_discriminator, define_gan, define_generator, define_baseline
from model.evaluate_model import evaluate_performance
from sklearn.model_selection import GroupKFold
import numpy as np



def train_model(model, X, y, metrics=True):
    if metrics:
        loss, metric = model.train_on_batch(X, y)
        return loss, metric
    else:
        loss = model.train_on_batch(X, y)
        return loss

def train(fold, res_dir, g_model, d_model, c_model, gan_model, b_model, train_dataset, train_targets, train_subject_idx, val_dataset, val_targets, latent_dim, n_epochs=20, n_batch=100):
    # select supervised dataset
    # use whole persons for the supervised data set --> just utilize the split function!
    #sup_amount = 0.1 #ca. 77 persons (when 10% are already gone for being the final test set and another 10% for the validation data in this fold)
    sup_amount = 100/train_dataset.shape[0]
    print(sup_amount)
    dataset_real, targets_real, idx_real, dataset_sup, targets_sup = split_data(sup_amount, train_dataset, train_targets, train_subject_idx)

    #dataset_sup = train_dataset[0:100]
    #targets_sup = train_targets[0:100]

    #dataset_real = train_dataset[100:]
    #targets_real = train_targets[100:]

    from matplotlib import pyplot as plt
    for i in range(100):
        # define subplot
        plt.subplot(10, 10, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(dataset_sup[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'test_images.png'
    plt.savefig(filename1)
    plt.close()

    print(np.unique(targets_sup))
    for number in np.unique(targets_sup):
        print(sum(targets_sup[targets_sup == float(number)]) / float(number))
    print(targets_sup[:100])

    print("Supvervised Samples' Shape: {}, Supervised Targets' Shape: {}".format(dataset_sup.shape, targets_sup.shape))
    print("Unupvervised Samples' Shape: {}, Unsupervised Targets' Shape: {}".format(dataset_real.shape, targets_real.shape))

    ##Batching and epochs
    #calculate the number of batches per training epoch
    bat_per_epo = int(train_dataset.shape[0] / n_batch) #round up?
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('fold=%d, n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (fold + 1, n_epochs, n_batch, half_batch, bat_per_epo, n_steps))

    ##Prepare storage for logging
    epochs_list = list()
    c_losses_train = list()
    c_metrics_train = list()
    b_losses_train = list()
    b_metrics_train = list()
    c_losses_val = list()
    c_metrics_val = list()
    b_losses_val = list()
    b_metrics_val = list()
    best_metric_val = float("-inf")
    path_sub_dir = res_dir + "/" + "fold_{}".format(fold)

    #Manually enumerate epochs
    for j in range(n_epochs):
        epochs_list.append(j)
        for i in range(bat_per_epo):

            ##Create inputs
            #real supervised samples
            X_sup, y_sup, _ = select_samples(dataset_sup, targets_sup, n_samples=half_batch)
            #real unsupervised samples (ignore age-target and use vector of ones to indicate the images are real)
            X_real, _, _ = select_samples(dataset_real, targets_real, n_samples=half_batch)
            y_real = np.ones((half_batch, 1))
            #fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #input from latent space (labeled as REAL!)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))

            ##Train models
            #train supervised discriminator (c)
            c_loss_train, c_metric_train = train_model(c_model, X_sup, y_sup)
            #train unsupervised discriminator (d) on real images
            d_loss_real = train_model(d_model, X_real, y_real, metrics=False)
            #train unsupervised discriminator (d) on fake images
            d_loss_fake = train_model(d_model, X_fake, y_fake, metrics=False)
            #train generator (so train gan-model)
            g_loss = train_model(gan_model, X_gan, y_gan, metrics=False)
            #train baseline model (b)
            b_loss_train, b_metric_train = train_model(b_model, X_sup, y_sup)

            ##Summarize loss on this batch
            print('-->Training: fold %d, epoch %d, batch %d/%d, c[%.3f, %.3f], b[%.3f, %.3f], d[%.3f, %.3f], g[%.3f]' % (
            fold + 1, j + 1, i + 1, bat_per_epo, c_loss_train, c_metric_train, b_loss_train, b_metric_train,
            d_loss_real, d_loss_fake, g_loss))

            if i % 300 == 0:
                _, c_test_acc = c_model.evaluate(dataset_real, targets_real)
                _, b_test_acc = b_model.evaluate(dataset_real, targets_real)
                print("acc C: {:.3f} / acc B: {:.3f}".format(c_test_acc, b_test_acc))

        ##After each epoch

        ##Validation
        c_loss_val, c_metric_val = c_model.evaluate(val_dataset, val_targets, verbose=0)
        b_loss_val, b_metric_val = b_model.evaluate(val_dataset, val_targets, verbose=0)
        print('-->Validation: fold %d, epoch %d, mae c[%.3f], mae b[%.3f]' % (fold + 1, j + 1, c_metric_val, b_metric_val))

        ##Save losses and metrics
        c_losses_train.append(c_loss_train)
        c_metrics_train.append(c_metric_train)
        b_losses_train.append(b_loss_train)
        b_metrics_train.append(b_metric_train)
        c_losses_val.append(c_loss_val)
        c_metrics_val.append(c_metric_val)
        b_losses_val.append(b_loss_val)
        b_metrics_val.append(b_metric_val)

        ##Evaluate performance
        best_metric_val = evaluate_performance(epochs_list, fold, res_dir, c_losses_train, c_losses_val,
                                               c_metrics_val, b_losses_train, b_losses_val, b_metrics_val,
                                               best_metric_val, c_model, d_model, g_model, b_model, latent_dim)

    return c_model, d_model, g_model, b_model


def run_cv(dataset, targets, subject_idx, n_folds,  lr=0.0002, n_batch=100, n_epochs=100, name="Run1", latent_dim=100):
    ##Make folds
    group_kfold = GroupKFold(n_splits=n_folds)
    folds = group_kfold.split(dataset, targets, subject_idx)

    ##Create results directory
    dir_name = mk_result_dir(name, n_folds)

    ##Enumerate epochs
    fold = 0
    for j, (train_idx, val_idx) in enumerate(folds):

        ##Select current data
        train_dataset = dataset[train_idx]
        train_targets = targets[train_idx]
        train_subject_idx = subject_idx[train_idx]
        val_dataset = dataset[val_idx]
        val_targets = targets[val_idx]
        val_subject_idx = subject_idx[val_idx]

        ##Model instantiation
        # create the discriminator models
        d_model, c_model = define_discriminator(lr=lr)
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model, lr=lr)
        # create basline model
        b_model = define_baseline(lr=lr)

        ##Train models
        c_model_trained, d_model_trained, g_model_trained, b_model_trained = train(fold, dir_name, g_model, d_model,
                                                            c_model, gan_model, b_model, train_dataset, train_targets,
                                                            train_subject_idx, val_dataset, val_targets, latent_dim,
                                                            n_epochs=n_epochs, n_batch=n_batch)
        #increment fold index
        fold+=1

    return(c_model_trained, d_model_trained, g_model_trained, b_model_trained, dir_name)