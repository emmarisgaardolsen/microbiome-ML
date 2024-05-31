import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# Update matplotlib defaults
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 16, 'legend.fontsize': 16, 'font.family': 'serif'})

# Decorator for running a function on multiple dataset splits
def run_on_splits(func):
    def _run_loop(model, splits, **kwargs):
        results, roc_data, prc_data = {}, {}, {}
        test_roc_data, test_prc_data = {}, {}
        model_name = kwargs.get('model_name', 'model')

        for X, y, nsplit in splits:
            result, roc_info, prc_info = func(model, X, y, nsplit, **kwargs)
            results[nsplit], roc_data[nsplit], prc_data[nsplit] = result, roc_info, prc_info
            if nsplit == 'test':
                test_roc_data[model_name], test_prc_data[model_name] = roc_info, prc_info

        return results, roc_data, prc_data, test_roc_data, test_prc_data
    return _run_loop

@run_on_splits
def evaluate_classification(model, X, y, nsplit, model_name, best_params=None):
    preds = model.predict(X)
    pred_probs = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y, preds)
    roc_auc = roc_auc_score(y, pred_probs)
    fpr, tpr, _ = roc_curve(y, pred_probs)
    precision, recall, _ = precision_recall_curve(y, pred_probs)
    prc_auc = auc(recall, precision)
    report = classification_report(y, preds, output_dict=True)
    
    print(f"{model_name} - {nsplit} - Accuracy: {accuracy}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}\n{report}")
    return (accuracy, report), (fpr, tpr, roc_auc), (precision, recall, prc_auc)

def save_model_results(results, model_name, results_dir):
    directory = results_dir
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f'{model_name}_results.txt')
    with open(filepath, 'w') as f:
        for split, (accuracy, report) in results.items():
            f.write(f"{model_name} - {split} - Accuracy: {accuracy}\n")
            f.write("Classification Report:\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def save_roc_auc_scores(roc_data, results_dir, filename='roc_auc_scores.txt'):
    with open(os.path.join(results_dir, filename), 'w') as f:
        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            f.write(f"{model_name}: ROC AUC = {roc_auc:.2f}\n")

def plot_feature_importances(model, model_name, feature_names, results_dir, filename='feature_importances.png'):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(os.path.join(results_dir, f'{model_name}_{filename}'))
    plt.close()

def plot_roc_curves(roc_data, model_name, results_dir, filename='roc_curves.png'):
    plt.figure(figsize=(10, 8))
    for split, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} - {split} (ROC AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f'{model_name}_{filename}'))
    plt.close()

def plot_prc_curves(prc_data, model_name, results_dir, filename='prc_curves.png'):
    plt.figure(figsize=(10, 8))
    for split, (precision, recall, prc_auc) in prc_data.items():
        plt.plot(recall, precision, label=f'{model_name} - {split} (PRC AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, f'{model_name}_{filename}'))
    plt.close()

def plot_combined_roc_curves(all_roc_data, results_dir, filename='all_roc_curves.png'):
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in all_roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (ROC AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def plot_selected_roc_curves(selected_roc_data, results_dir, filename='selected_roc_curves.png'):
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in selected_roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (ROC AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Selected Models ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def plot_combined_prc_curves(all_prc_data, results_dir, filename='all_prc_curves.png'):
    plt.figure(figsize=(10, 8))
    for model_name, (precision, recall, prc_auc) in all_prc_data.items():
        plt.plot(recall, precision, label=f'{model_name} (PRC AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def save_best_params(model_name, best_params, results_dir):
    with open(os.path.join(results_dir, 'best_params.txt'), 'a') as f:
        f.write(f"{model_name}:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

def load_data(data_dir):
    train_data = pd.read_csv(data_dir / "train.csv")
    val_data = pd.read_csv(data_dir / "val.csv")
    test_data = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_val, y_val = val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    feature_names = train_data.columns[:-1]
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def evaluate_deterministic_model(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    def deterministic_predict(X):
        return (1 - np.sum(X, axis=1) >= 0.01).astype(int)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    results, roc_data, prc_data = {}, {}, {}

    for X, y, nsplit in splits:
        preds = deterministic_predict(X)
        pred_probs = preds
        accuracy = accuracy_score(y, preds)
        roc_auc = roc_auc_score(y, pred_probs)
        fpr, tpr, _ = roc_curve(y, pred_probs)
        precision, recall, _ = precision_recall_curve(y, pred_probs)
        prc_auc = auc(recall, precision)
        report = classification_report(y, preds, output_dict=True)

        print(f"Deterministic - {nsplit} - Accuracy: {accuracy}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}\n{report}")
        results[nsplit] = (accuracy, report)
        roc_data[nsplit] = (fpr, tpr, roc_auc)
        prc_data[nsplit] = (precision, recall, prc_auc)

    save_model_results(results, "Deterministic", results_dir)
    plot_roc_curves(roc_data, "Deterministic", results_dir)
    plot_prc_curves(prc_data, "Deterministic", results_dir)

    all_roc_data["Deterministic"] = roc_data['test']
    all_prc_data["Deterministic"] = prc_data['test']

    return results, roc_data, prc_data

def tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, results_dir):
    basic_rfc = RandomForestClassifier(random_state=42)
    basic_rfc.fit(X_train, y_train)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    basic_results, basic_roc_data, basic_prc_data, test_roc_data, test_prc_data = evaluate_classification(basic_rfc, splits, model_name="Random_Forest_Basic")
    save_model_results(basic_results, "Random_Forest_Basic", results_dir)
    plot_roc_curves(basic_roc_data, "Random_Forest_Basic", results_dir)
    plot_prc_curves(basic_prc_data, "Random_Forest_Basic", results_dir)
    plot_feature_importances(basic_rfc, "Random_Forest_Basic", feature_names, results_dir)

    all_roc_data["Random_Forest_Basic"] = test_roc_data["Random_Forest_Basic"]
    all_prc_data["Random_Forest_Basic"] = test_prc_data["Random_Forest_Basic"]

    param_grid = {
        'n_estimators': [10, 50, 80, 100, 120, 200, 300, 400],
        'max_depth': [None, 3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', None]
    }

    cv_rfc = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, scoring='balanced_accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    cv_rfc.fit(X_train, y_train)
    best_params = cv_rfc.best_params_
    save_best_params('Random_Forest_Optimized', best_params, results_dir)
    print("Best parameters:", best_params)

    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(cv_rfc.best_estimator_, splits, model_name="Random_Forest_Optimized")
    save_model_results(results, "Random_Forest_Optimized", results_dir)
    plot_roc_curves(roc_data, "Random_Forest_Optimized", results_dir)
    plot_prc_curves(prc_data, "Random_Forest_Optimized", results_dir)
    plot_feature_importances(cv_rfc.best_estimator_, "Random_Forest_Optimized", feature_names, results_dir)

    all_roc_data["Random_Forest_Optimized"] = test_roc_data["Random_Forest_Optimized"]
    all_prc_data["Random_Forest_Optimized"] = test_prc_data["Random_Forest_Optimized"]

    return results, roc_data, prc_data

def tune_clf_hyperparameters(clf, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    print("Best hyperparameters:\n", clf_grid.best_params_)
    return clf_grid.best_estimator_

def tune_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    basic_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    basic_model.fit(X_train, y_train)
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    basic_results, basic_roc_data, basic_prc_data, test_roc_data, test_prc_data = evaluate_classification(basic_model, splits, model_name="XGBoost_Basic")
    save_model_results(basic_results, "XGBoost_Basic", results_dir)
    plot_roc_curves(basic_roc_data, "XGBoost_Basic", results_dir)
    plot_prc_curves(basic_prc_data, "XGBoost_Basic", results_dir)

    all_roc_data["XGBoost_Basic"] = test_roc_data["XGBoost_Basic"]
    all_prc_data["XGBoost_Basic"] = test_prc_data["XGBoost_Basic"]

    xgb_param_grid = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2),
        'learning_rate': [0.0001, 0.01, 0.1],
        'n_estimators': [50, 200]
    }

    xgb_opt = tune_clf_hyperparameters(xgb.XGBClassifier(random_state=0), xgb_param_grid, X_train, y_train)
    best_params = xgb_opt.get_params()
    save_best_params('XGBoost_Optimized', best_params, results_dir)

    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(xgb_opt, splits, model_name="XGBoost_Optimized")
    save_model_results(results, "XGBoost_Optimized", results_dir)
    plot_roc_curves(roc_data, "XGBoost_Optimized", results_dir)
    plot_prc_curves(prc_data, "XGBoost_Optimized", results_dir)

    all_roc_data["XGBoost_Optimized"] = test_roc_data["XGBoost_Optimized"]
    all_prc_data["XGBoost_Optimized"] = test_prc_data["XGBoost_Optimized"]

    return results, roc_data, prc_data

def tune_and_evaluate_linear_svc(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    grid_search = GridSearchCV(LinearSVC(random_state=42, dual=False), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_linear_svc = grid_search.best_estimator_
    calibrated_svc = CalibratedClassifierCV(best_linear_svc, method='sigmoid', cv=5)
    calibrated_svc.fit(X_train, y_train)

    save_best_params('Linear_SVC_Optimized', grid_search.best_params_, results_dir)
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(calibrated_svc, splits, model_name="Linear_SVC_Optimized")
    save_model_results(results, "Linear_SVC_Optimized", results_dir)
    plot_roc_curves(roc_data, "Linear_SVC_Optimized", results_dir)
    plot_prc_curves(prc_data, "Linear_SVC_Optimized", results_dir)

    all_roc_data["Linear_SVC_Optimized"] = test_roc_data["Linear_SVC_Optimized"]
    all_prc_data["Linear_SVC_Optimized"] = test_prc_data["Linear_SVC_Optimized"]

    return results, roc_data, prc_data

def tune_and_evaluate_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Neural Network Test accuracy:', test_acc)

    test_predictions = (model.predict(X_test) > 0.5).astype("int32")
    test_pred_probs = model.predict(X_test).flatten()
    test_report = classification_report(y_test, test_predictions, output_dict=True)

    fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
    precision, recall, _ = precision_recall_curve(y_test, test_pred_probs)
    roc_auc = roc_auc_score(y_test, test_pred_probs)
    prc_auc = auc(recall, precision)

    results = {'train': ('Not Evaluated', {}), 'val': ('Not Evaluated', {}), 'test': (test_acc, test_report)}
    save_model_results(results, "Neural_Network", results_dir)

    test_roc_data = {"Neural_Network": (fpr, tpr, roc_auc)}
    test_prc_data = {"Neural_Network": (precision, recall, prc_auc)}

    all_roc_data["Neural_Network"] = test_roc_data["Neural_Network"]
    all_prc_data["Neural_Network"] = test_prc_data["Neural_Network"]

    plot_roc_curves(test_roc_data, "Neural_Network", results_dir)
    plot_prc_curves(test_prc_data, "Neural_Network", results_dir)

    return results, test_roc_data, test_prc_data

def evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=10000), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    save_best_params('Logistic_Regression_Best', grid_search.best_params_, results_dir)

    best_lr = grid_search.best_estimator_
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    best_results, best_roc_data, best_prc_data, test_roc_data, test_prc_data = evaluate_classification(best_lr, splits, model_name="Logistic_Regression_Best")
    save_model_results(best_results, "Logistic_Regression_Best", results_dir)
    plot_roc_curves(best_roc_data, "Logistic_Regression_Best", results_dir)
    plot_prc_curves(best_prc_data, "Logistic_Regression_Best", results_dir)

    all_roc_data["Logistic_Regression_Best"] = test_roc_data["Logistic_Regression_Best"]
    all_prc_data["Logistic_Regression_Best"] = test_prc_data["Logistic_Regression_Best"]

    return best_results, best_roc_data, best_prc_data

def evaluate_elastic_net_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],
        'solver': ['saga'],
        'penalty': ['elasticnet']
    }
    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=10000), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)

    best_enet_lr = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    save_best_params('Elastic_Net_Logistic_Regression_Best', grid_search.best_params_, results_dir)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    enet_results, enet_roc_data, enet_prc_data, test_roc_data, test_prc_data = evaluate_classification(best_enet_lr, splits, model_name="Elastic_Net_Logistic_Regression_Best")
    save_model_results(enet_results, "Elastic_Net_Logistic_Regression_Best", results_dir)
    plot_roc_curves(enet_roc_data, "Elastic_Net_Logistic_Regression_Best", results_dir)
    plot_prc_curves(enet_prc_data, "Elastic_Net_Logistic_Regression_Best", results_dir)

    all_roc_data["Elastic_Net_Logistic_Regression_Best"] = test_roc_data["Elastic_Net_Logistic_Regression_Best"]
    all_prc_data["Elastic_Net_Logistic_Regression_Best"] = test_prc_data["Elastic_Net_Logistic_Regression_Best"]

    return enet_results, enet_roc_data, enet_prc_data

def tune_and_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    save_best_params('KNN_Optimized', grid_search.best_params_, results_dir)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(best_knn, splits, model_name="KNN_Optimized")
    save_model_results(results, "KNN_Optimized", results_dir)
    plot_roc_curves(roc_data, "KNN_Optimized", results_dir)
    plot_prc_curves(prc_data, "KNN_Optimized", results_dir)

    all_roc_data["KNN_Optimized"] = test_roc_data["KNN_Optimized"]
    all_prc_data["KNN_Optimized"] = test_prc_data["KNN_Optimized"]

    return results, roc_data, prc_data

def evaluate_majority_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    majority_class_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    majority_class_clf.fit(X_train, y_train)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    majority_results, majority_roc_data, majority_prc_data, test_roc_data, test_prc_data = evaluate_classification(majority_class_clf, splits, model_name="Majority_Class_Classifier")
    save_model_results(majority_results, "Majority_Class_Classifier", results_dir)
    plot_roc_curves(majority_roc_data, "Majority_Class_Classifier", results_dir)
    plot_prc_curves(majority_prc_data, "Majority_Class_Classifier", results_dir)

    all_roc_data["Majority_Class_Classifier"] = test_roc_data["Majority_Class_Classifier"]
    all_prc_data["Majority_Class_Classifier"] = test_prc_data["Majority_Class_Classifier"]

    return majority_results, majority_roc_data, majority_prc_data

def evaluate_chance_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    random_class_clf = DummyClassifier(strategy='uniform', random_state=42)
    random_class_clf.fit(X_train, y_train)

    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    random_results, random_roc_data, random_prc_data, test_roc_data, test_prc_data = evaluate_classification(random_class_clf, splits, model_name="Chance_Class_Classifier")
    save_model_results(random_results, "Chance_Class_Classifier", results_dir)
    plot_roc_curves(random_roc_data, "Chance_Class_Classifier", results_dir)
    plot_prc_curves(random_prc_data, "Chance_Class_Classifier", results_dir)

    all_roc_data["Chance_Class_Classifier"] = test_roc_data["Chance_Class_Classifier"]
    all_prc_data["Chance_Class_Classifier"] = test_prc_data["Chance_Class_Classifier"]

    return random_results, random_roc_data, random_prc_data

def run_all_models(data_dir, results_dir):
    global all_roc_data, all_prc_data
    all_roc_data, all_prc_data = {}, {}

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data(data_dir)
    dataset_results_dir = results_dir / data_dir.stem
    os.makedirs(dataset_results_dir, exist_ok=True)

    # SVC
    tune_and_evaluate_linear_svc(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Random Forest
    tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, dataset_results_dir)

    # XGBoost
    tune_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Logistic Regression
    evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Elastic Net Logistic Regression
    evaluate_elastic_net_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # KNN Classifier
    tune_and_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Majority Class Classifier
    evaluate_majority_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Random Class Classifier
    evaluate_chance_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Neural Network
    tune_and_evaluate_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # plot combined PRC and ROC curves for all models for the current dataset
    plot_combined_prc_curves(all_prc_data, dataset_results_dir, filename='all_prc_curves.png')
    plot_combined_roc_curves(all_roc_data, dataset_results_dir, filename='all_roc_curves.png')
    save_roc_auc_scores(all_roc_data, dataset_results_dir)

    selected_roc_data = {
        "RF": all_roc_data["Random_Forest_Optimized"],
        "XG": all_roc_data["XGBoost_Optimized"],
        "SVC": all_roc_data["Linear_SVC_Optimized"],
        "LogReg": all_roc_data["Elastic_Net_Logistic_Regression_Best"],
        "KNN": all_roc_data["KNN_Optimized"],
        "NN": all_roc_data["Neural_Network"]
    }

    plot_selected_roc_curves(selected_roc_data, dataset_results_dir, filename='selected_roc_curves.png')

def main():
    root = Path.cwd().parents[1]
    data_dir = root / "data"
    results_dir = root / "results" / "model_reports"

    for current_dir in data_dir.iterdir():
        if current_dir.is_dir() and current_dir.name != 'raw':
            print(f"Now processing {current_dir}")
            run_all_models(current_dir, results_dir)


if __name__ == "__main__":
    main()
