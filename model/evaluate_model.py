#TF-GPU IMPORTS
from utils.results import save_csv
from utils.models import save_model
from utils.plotting import plot_val_train_loss, plot_acc, generate_images


def evaluate_performance(epochs_list, fold, res_dir, c_losses_train, c_losses_val, c_metrics_val, b_losses_train, b_losses_val,
                                 b_metrics_val, best_metric_val, c_model, d_model, g_model, b_model, latent_dim):
    #sub directory for current fold
    sub_dir = res_dir + "/" + "fold_{}".format(fold)

    #log information
    save_csv(path=sub_dir, name="results_fold_{}".format(fold), mode="w", epoch_list=epochs_list,
             c_losses_train=c_losses_train, c_losses_val=c_losses_val, b_losses_train=b_losses_train,
             b_losses_val=b_losses_val, c_metrics_val=c_metrics_val, b_metrics_val=b_metrics_val)

    #decide if validation metric approved and if to save models
    c_metric_val = c_metrics_val[-1]
    b_metric_val = b_metrics_val[-1]
    if c_metric_val < best_metric_val:
        print("C Validation metric improved from [{:.3f}] to [{:.3f}]. Saving models...".format(best_metric_val,
                                                                                                c_metric_val))
        save_csv(res_dir, "best_val_metrics", mode="a", fold=[fold], c_best_val=[c_metric_val], b_val=[b_metric_val])
        best_metric_val = c_metric_val
        save_model(c_model, sub_dir, "c", fold)
        save_model(d_model, sub_dir, "d", fold)
        save_model(g_model, sub_dir, "g", fold)
        save_model(b_model, sub_dir, "b", fold)

    #plot training and validation performance
    plot_val_train_loss(c_losses_train, c_losses_val, b_losses_train, b_losses_val, fold, sub_dir)
    plot_acc(c_metrics_val, b_metrics_val, fold, sub_dir)

    #generate images
    generate_images(g_model, sub_dir, fold, latent_dim)

    return best_metric_val


