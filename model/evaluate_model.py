#TF-GPU IMPORTS
from utils.results import save_csv
from utils.models import save_model
from utils.plotting import plot_val_train_loss, plot_acc, generate_images


def evaluate_performance(fold, path, metric, prev_metric, epoch_list, c_losses_train , c_losses_val, metrics, c_model, d_model, g_model, dataset, latent_dim):
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
    generate_images(g_model, path_sub_dir, fold, dataset, latent_dim)
    return(prev_metric)