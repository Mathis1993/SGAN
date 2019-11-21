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
from utils.results import save_csv
from utils.models import save_model
from utils.plotting import plot_val_train_loss, plot_acc


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