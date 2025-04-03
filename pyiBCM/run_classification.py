print("in run_classification.py")
import os
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.tree import DecisionTreeClassifier as DT


def classify(technique, classifiers, parameters):
    for name, classifier in classifiers:
        print(f"Classifier ", name)

        acc_sum = 0
        feat_sum = 0
        auc_sum = 0
        for i in range(0, no_folds):
            if technique == "MiSeRe":
                second = parameters[0]
                training = pd.read_csv(
                    "./datasets/training-test-data/MiSeRe_data2/"
                    + dataset
                    + "_training_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + "_"
                    + str(second)
                    + ".csv"
                )
                test = pd.read_csv(
                    "./datasets/training-test-data/MiSeRe_data2/"
                    + dataset
                    + "_test_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + "_"
                    + str(second)
                    + ".csv"
                )
            elif technique == "ISM":
                training = pd.read_csv(
                    "./datasets/training-test-data/"
                    + dataset
                    + "_training_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + ".csv"
                )
                test = pd.read_csv(
                    "./datasets/training-test-data/"
                    + dataset
                    + "_test_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + ".csv"
                )
            elif technique in ["iBCM", "BIDE", "SPADE", "PrefixSPAN"]:
                support = parameters[0]
                training = pd.read_csv(
                    "./datasets/training-test-data/"
                    + dataset
                    + "_training_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + "_"
                    + str(support)
                    + ".csv"
                )
                test = pd.read_csv(
                    "./datasets/training-test-data/"
                    + dataset
                    + "_test_fold_"
                    + str(i)
                    + "_"
                    + technique
                    + "_"
                    + str(support)
                    + ".csv"
                )

            y_train = training["label"]
            X_train = training.drop(["label"], axis=1)

            y_test = test["label"]
            X_test = test.drop(["label"], axis=1)

            if len(X_train.columns) < 2:
                print("No features for fold", i)
                continue

            feat_sum += len(X_train.columns)

            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            predictions_prob = classifier.predict_proba(X_test)

            acc = accuracy_score(y_test, predictions)
            if len(y_test.unique()) > 2:
                auc = roc_auc_score(y_test, predictions_prob, multi_class="ovo")
            else:
                auc = roc_auc_score(y_test, predictions_prob[:, 1])
            acc_sum += acc
            auc_sum += auc

        avg_feat = feat_sum / no_folds
        avg_acc = acc_sum / no_folds
        avg_auc = auc_sum / no_folds
        if write_results:
            results = open(name_result_file, "a")
            results.write(
                f"{dataset},{technique},{parameters[0]},{name},{no_folds},{avg_feat},{avg_acc},{avg_auc}\n"
            )
            results.close()
        print("Avg. acc.:", avg_acc)
        print("Avg. AUC.:", avg_auc)
        print("Avg. #features.:", avg_feat)


########################
## Start classification

techniques = ["iBCM", "MiSeRe", "BIDE", "PrefixSPAN", "SPADE"]

datasets = ["auslan2", "aslbu", "pioneer"]

support_levels = [0.2, 0.4, 0.6, 0.8]
seconds = [1, 2, 5]

from sklearn.feature_selection import SelectKBest, mutual_info_classif

no_folds = 10

classifiers = [("Decision tree", DT())]
classifiers = [("Random forest", RF(n_estimators=100))]  # ,('NB',NB())]


write_results = True

name_result_file = "results_java.csv"

if write_results:
    write_header = False

    if not os.path.exists(name_result_file):
        write_header = True
    results = open(name_result_file, "a")

    if write_header:
        results.write(
            "dataset,technique,parameter,classifier,no_folds,no_features,accuracy,auc\n"
        )

for dataset in datasets:
    print("\n")
    print(f"Dataset {dataset}")
    for technique in techniques:
        print(f"\nTechnique {technique}")

        if technique in ["iBCM", "BIDE", "SPADE", "PrefixSPAN"]:
            for support in support_levels:
                print(f"Support {support}")
                parameters = [support]
                classify(technique, classifiers, parameters)
        elif technique == "ISM":
            classify("ISM", classifiers, list())
        elif technique == "MiSeRe":
            for second in seconds:
                print(f"Seconds {second}")
                parameters = [second]
                classify("MiSeRe", classifiers, parameters)
