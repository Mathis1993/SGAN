import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


def search_files(dir, term="c_model"):
    files = list()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dir):
        for file in f:
            if term in file:
                files.append(os.path.join(r, file))
    return(files)


def mean_model(res_dir, dataset_test, targets_test):
    files = search_files(res_dir, "c_model")
    metrics = list()
    for file in files:
        cur_model = load_model(file)
        _, metric = cur_model.evaluate(dataset_test, targets_test, verbose=0)
        metrics.append(metric)
    print("Mean metric of best models over folds on test set: {:.3f}".format(np.mean(metrics)))


def best_model(res_dir, dataset_test, targets_test):
    #Find best model and return metric on test data
    best_val = pd.read_csv(res_dir + "/" + "best_val_metrics.csv", header=None)
    min_fold = best_val.loc[best_val.iloc[:,1]==np.min(best_val.iloc[:,1]) , 0].iloc[0]
    best_model = load_model(res_dir + "/" + "fold_" + str(min_fold) + "/" + "c_model_fold_" + str(min_fold) + ".h5")
    _, metric = best_model.evaluate(dataset_test, targets_test, verbose=0)
    print("Metric of best model on test set (fold {}): {:.3f}".format(min_fold, metric))
    return best_model


def save_model(model, path, filename):
   model.save(path + "/" +  filename)