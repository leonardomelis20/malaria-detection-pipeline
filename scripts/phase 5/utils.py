import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score



RANDOM_STATE = 42

def load_features(path, target="label_ids"):
    with h5py.File(path, "r") as f:
        x = f["features"][:]
        y = f[target][:]
        species = f["labels"][:]
        filenames = f["filenames"][:]

    return x, y, species, filenames

def get_classifiers():
    classifiers = {}
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    knn_5 = KNeighborsClassifier(n_neighbors=5)

    classifiers["RF"] = rf
    classifiers["LR"] = lr
    classifiers["KNN_1"] = knn_1
    classifiers["KNN_3"] = knn_3
    classifiers["KNN_5"] = knn_5

    return classifiers

def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "metrics.csv", index=False)

def run_experiment(train_h5, test_h5, clf, clf_name, model_name, fold_name):
    X_train , y_train, _, _ = load_features(train_h5)
    X_test, y_test, _, _ = load_features(test_h5)

    #addestro il classificatore
    clf.fit(X_train, y_train)
    all_classes = np.unique(y_train)

    #predico sui test
    y_pred = clf.predict(X_test)

    #calcolo delle metriche
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)

    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        y_prob = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    n_classes_val = len(np.unique(y_test))

    results = {
        "model":               model_name,
        "classifier":          clf_name,
        "fold":                fold_name,
        "accuracy":            acc,
        "balanced_accuracy":   balanced_acc,
        "precision_macro":     precision_macro,
        "precision_weighted":  precision_weighted,
        "recall_macro":        recall_macro,
        "recall_weighted":     recall_weighted,
        "f1_macro":            f1_macro,
        "f1_weighted":         f1_weighted,
        "mcc":                 mcc,
        "roc_auc_macro":       roc_auc,
        "n_classes_in_val":    n_classes_val,
    }

    return results, cm

def run_experiment_ood(X_train, y_train, X_test, y_test,
                       clf, clf_name, model_name,
                       source_species, target_species):

    clf.fit(X_train, y_train)
    all_classes = np.unique(y_train)
    y_pred = clf.predict(X_test)

    acc              = accuracy_score(y_test, y_pred)
    f1_macro         = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm               = confusion_matrix(y_test, y_pred, labels=all_classes)
    precision_macro  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_macro     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    recall_weighted  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    balanced_acc     = balanced_accuracy_score(y_test, y_pred)
    mcc              = matthews_corrcoef(y_test, y_pred)

    try:
        y_prob  = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    results = {
        "model":               model_name,
        "classifier":          clf_name,
        "source_species":      source_species,
        "target_species":      target_species,
        "accuracy":            acc,
        "balanced_accuracy":   balanced_acc,
        "precision_macro":     precision_macro,
        "precision_weighted":  precision_weighted,
        "recall_macro":        recall_macro,
        "recall_weighted":     recall_weighted,
        "f1_macro":            f1_macro,
        "f1_weighted":         f1_weighted,
        "mcc":                 mcc,
        "roc_auc_macro":       roc_auc,
        "n_classes_in_test":   len(np.unique(y_test)),
    }

    return results, cm