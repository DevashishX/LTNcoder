print("in create_training_test_datasets.py")
import numpy as np
from sklearn.model_selection import StratifiedKFold

datasets = ["pioneer", "Unix", "reuters", "auslan2", "aslbu", "context"]
no_folds = 10
np.random.seed(42)

for dataset in datasets:
    print("Handling dataset:", dataset)

    trace_file = open("./datasets/" + dataset + ".dat", "r")
    label_file = open("./datasets/" + dataset + ".lab", "r")

    traces = []
    label_list = []
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace("\n", ""))

    label_set = set(label_list)
    no_labels = len(label_set)
    print("#labels:", no_labels)

    trace_file.close()
    label_file.close()

    skf = StratifiedKFold(no_folds)

    for fold, (train_index, test_index) in enumerate(skf.split(traces, label_list)):
        trace_file_write = open(
            "./datasets/training-test-data/"
            + dataset
            + "_training_fold_"
            + str(fold)
            + ".dat",
            "w",
        )
        label_file_write = open(
            "./datasets/training-test-data/"
            + dataset
            + "_training_fold_"
            + str(fold)
            + ".lab",
            "w",
        )
        for i in train_index:
            trace_file_write.write(traces[i])
            label_file_write.write(label_list[i] + "\n")
        trace_file_write.close()
        label_file_write.close()

        trace_file_write = open(
            "./datasets/training-test-data/"
            + dataset
            + "_test_fold_"
            + str(fold)
            + ".dat",
            "w",
        )
        label_file_write = open(
            "./datasets/training-test-data/"
            + dataset
            + "_test_fold_"
            + str(fold)
            + ".lab",
            "w",
        )
        for i in test_index:
            trace_file_write.write(traces[i])
            label_file_write.write(label_list[i] + "\n")
        trace_file_write.close()
        label_file_write.close()

    #############################################
    ### MiSeRe employs a different file structure
    trace_file = open("./datasets/" + dataset + ".dat", "r")
    label_file = open("./datasets/" + dataset + ".lab", "r")

    traces = []
    label_list = []
    label_dict = {}
    label_no = -3
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_str = label.replace("\n", "")
        if label_str not in label_dict.keys():
            label_dict[label_str] = str(label_no)
            label_no -= 1
        label_list.append(label_dict[label_str])

    label_set = set(label_list)
    no_labels = len(label_set)
    print("#labels:", no_labels)

    skf = StratifiedKFold(no_folds)

    for fold, (train_index, test_index) in enumerate(skf.split(traces, label_list)):
        trace_file_write = open(
            "./datasets/training-test-data/MiSeRe_data/"
            + dataset
            + "_training_fold_"
            + str(fold)
            + ".text",
            "w",
        )
        for i in train_index:
            trace_file_write.write(label_list[i] + " -1 " + traces[i])
        trace_file_write.close()

        trace_file_write = open(
            "./datasets/training-test-data/MiSeRe_data/"
            + dataset
            + "_test_fold_"
            + str(fold)
            + ".text",
            "w",
        )
        for i in test_index:
            trace_file_write.write(label_list[i] + " -1 " + traces[i])
        trace_file_write.close()
