from matplotlib import pyplot as plt
import numpy as np
from model.create_model import generate_fake_samples
from utils.manipulation import normalize


def plot_val_train_loss(c_losses_train, c_losses_val, b_losses_train, b_losses_val, fold, path):

    #turn interactive mode off, because plot cannot be displayed in console
    plt.ioff()

    #clear figure
    plt.clf()

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(c_losses_train)+1),c_losses_train, label='C Training Loss')
    plt.plot(range(1,len(c_losses_val)+1),c_losses_val, label='C Validation Loss')
    plt.plot(range(1, len(b_losses_train) + 1), b_losses_train, label='B Training Loss')
    plt.plot(range(1, len(b_losses_val) + 1), b_losses_val, label='B Validation Loss')

    # find position of lowest validation loss
    minposs = c_losses_val.index(min(c_losses_val))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Lowest C Validation Loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(c_losses_train)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Loss per Epoch", fontsize=20)
    plt.tight_layout()
    #plt.show() #no showing, only saving
    name = "train_val_loss_fold_{}.png".format(fold)
    fig.savefig(path + "/" + name, bbox_inches='tight')
    plt.close()


def plot_acc(c_metrics_val, b_metrics_val, fold, path):
    # turn interactive mode off, because plot cannot be displayed in console
    plt.ioff()

    #clear figure
    plt.clf()

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(c_metrics_val) + 1), c_metrics_val, label='C Validation Metric')
    plt.plot(range(1, len(b_metrics_val) + 1), b_metrics_val, label='B Validation Metric')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.xlim(0, len(c_metrics_val) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title("Validation Metric per Epoch", fontsize=20)
    plt.tight_layout()
    # plt.show() #no showing, only saving
    name = "val_metric_fold_{}.png".format(fold)
    fig.savefig(path + "/" + name, bbox_inches='tight')
    plt.close()


def generate_images(g_model, path, fold, latent_dim):
    # prepare fake examples (if changing n_samples, also change specifications of subplot)
    n_samples = 9
    X, _ = generate_fake_samples(g_model, latent_dim=latent_dim, n_samples=n_samples)
    #normalize from [-1,1] to [0,1] for plotting
    X = normalize(X, feature_range=(0,1))
    #plot images
    for i in range(n_samples):
        plt.subplot(3, 3, 1 + i)
        plt.axis("off")
        plt.imshow(X[i, :, :, 0])
    filename1 = path + "/" + "generated_images_fold_{}".format(fold)
    plt.savefig(filename1)
    plt.close()

