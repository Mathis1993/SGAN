import csv
import os
from datetime import date, datetime


def save_csv(path, name, mode="w", **variables):
#def save(**variables):
    res_file = path + "/" + name + ".csv"
    #get the dict keys as a list: names of the columns --> name row to be written to the csv
    row_names = [key for key in variables.keys()]
    if mode == "w":
        with open(res_file, mode) as f:
            writer = csv.writer(f)
            writer.writerow(row_names)
            #get a list of the dict values: So a list of lists
            value_lists = [value_list for value_list in variables.values()]
            #Make new lists: The first one containing every first element of the sublists, the second one containing
            #every second element of the sublists, etc. --> rows to be written to the csv
            for i in range(len(value_lists[0])):
                row = list()
                for j in range(len(value_lists)):
                    row.append(value_lists[j][i])
                writer.writerow(row)
    if mode == "a":
        with open(res_file, mode) as f:
            writer = csv.writer(f)
            # get a list of the dict values: So a list of lists
            value_lists = [value_list for value_list in variables.values()]
            # Make new lists: The first one containing every first element of the sublists, the second one containing
            # every second element of the sublists, etc. --> rows to be written to the csv
            for i in range(len(value_lists[0])):
                row = list()
                for j in range(len(value_lists)):
                    row.append(value_lists[j][i])
                writer.writerow(row)


def mk_result_dir(name, n_folds):
    #get date and time
    today = date.today().strftime("%d-%m-%Y")
    time = datetime.now().time().strftime("%H-%M-%S")
    now = today + "_" + time

    #full directory name
    dir_name = name + "_" + now

    #create directory, if it doesn't already exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Created directory {}".format(dir_name))

    #create sub-directories for single folds
    for i in range(n_folds):
        fold_name = "fold_" + str(i)
        if not os.path.exists(dir_name + "/" + fold_name):
            os.mkdir(dir_name + "/" + fold_name)

    return(dir_name)