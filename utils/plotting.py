from matplotlib import pyplot as plt


def plot_val_train_loss(train_losses, val_losses, fold, path):

    #turn interactive mode off, because plot cannot be displayed in console
    plt.ioff()

    #clear figure
    plt.clf()

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
    plt.plot(range(1,len(val_losses)+1),val_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = val_losses.index(min(val_losses))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Lowest Validation Loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Loss per Epoch", fontsize=20)
    plt.tight_layout()
    #plt.show() #no showing, only saving
    name = "train_val_loss_fold_{}.png".format(fold)
    fig.savefig(path + "/" + name, bbox_inches='tight')
    plt.close()

#ToDo: Why does this throw an error?!
def plot_acc(metric, fold, path):
    # turn interactive mode off, because plot cannot be displayed in console
    plt.ioff()

    #clear figure
    plt.clf()

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(metric) + 1), metric, label='Validation Metric')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.xlim(0, len(metric) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title("Validation Metric per Epoch", fontsize=20)
    plt.tight_layout()
    # plt.show() #no showing, only saving
    name = "val_metric_fold_{}.png".format(fold)
    fig.savefig(path + "/" + name, bbox_inches='tight')
    plt.close()