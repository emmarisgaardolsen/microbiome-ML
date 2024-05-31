# %% [markdown]
# # Notebook used for analysis
# Obs: on a Macbook Pro 13" M1 2020 with 16 GB memory and Sonoma 14.1, it took approximately 9.5-10 hours to run this code.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# %%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# %%
# set default font size
plt.rcParams.update({'font.size': 16})
# set default legend font size
plt.rcParams.update({'legend.fontsize': 16})

# %%
plt.rcParams['font.family'] = 'serif'  # similar to Times New Roman
#plt.rcParams['font.family'] = 'sans-serif'  # similar to Libertine

# %%
all_roc_data = {}
all_prc_data = {}

# %%
# decorator for running a function on multiple dataset splits
def run_on_splits(func):
    def _run_loop(model, splits, **kwargs):
        results = {}
        roc_data = {}
        prc_data = {}
        test_roc_data = {}
        test_prc_data = {}
        model_name = kwargs.get('model_name', 'model')
        for split in splits:
            X, y, nsplit = split
            result, roc_info, prc_info = func(model, X, y, nsplit, **kwargs)
            results[nsplit] = result
            roc_data[nsplit] = roc_info
            prc_data[nsplit] = prc_info
            if nsplit == 'test':
                test_roc_data[model_name] = roc_info
                test_prc_data[model_name] = prc_info
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
    full_path = os.path.join(results_dir, f'{model_name}_{filename}')
    plt.savefig(full_path)
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
    full_path = os.path.join(results_dir, f'{model_name}_{filename}')
    plt.savefig(full_path)
    plt.close()

def plot_prc_curves(prc_data, model_name, results_dir, filename='prc_curves.png'):
    plt.figure(figsize=(10, 8))
    for split, (precision, recall, prc_auc) in prc_data.items():
        plt.plot(recall, precision, label=f'{model_name} - {split} (PRC AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    full_path = os.path.join(results_dir, f'{model_name}_{filename}')
    plt.savefig(full_path)
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
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path)
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
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path)
    plt.close()

def plot_combined_prc_curves(all_prc_data, results_dir, filename='all_prc_curves.png'):
    plt.figure(figsize=(10, 8))
    for model_name, prc_data in all_prc_data.items():
        precision, recall, prc_auc = prc_data
        plt.plot(recall, precision, label=f'{model_name} (PRC AUC = {prc_auc:.2f})')
    plt.xlabel('Recall',fontsize=12)
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curves')
    plt.legend(loc="lower left")
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path)
    plt.close()
    
    
def save_best_params(model_name, best_params, results_dir):
    filepath = os.path.join(results_dir, 'best_params.txt')
    with open(filepath, 'a') as f:
        f.write(f"{model_name}:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

def load_data(data_dir):
    train_data_path = data_dir / "train.csv"
    val_data_path = data_dir / "val.csv"
    test_data_path = data_dir / "test.csv"
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    feature_names = train_data.columns[:-1]
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


# %%
def evaluate_deterministic_model(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    
    # calculate predictions for the deterministic model
    def deterministic_predict(X):
        sum_non_outcome = np.sum(X, axis=1)
        difference = 1 - sum_non_outcome
        return (difference >= 0.01).astype(int)

    splits = [
        (X_train, y_train, 'train'),
        (X_val, y_val, 'val'),
        (X_test, y_test, 'test')
    ]

    results = {}
    roc_data = {}
    prc_data = {}
    test_roc_data = {}
    test_prc_data = {}

    for X, y, nsplit in splits:
        preds = deterministic_predict(X)
        pred_probs = preds  # since it's deterministic, we use the binary predictions
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
        if nsplit == 'test':
            test_roc_data = {"Deterministic": (fpr, tpr, roc_auc)}
            test_prc_data = {"Deterministic": (precision, recall, prc_auc)}

    save_model_results(results, "Deterministic", results_dir)
    plot_roc_curves(roc_data, "Deterministic", results_dir, filename='roc_curves.png')
    plot_prc_curves(prc_data, "Deterministic", results_dir, filename='prc_curves.png')

    all_roc_data["Deterministic"] = test_roc_data["Deterministic"]
    all_prc_data["Deterministic"] = test_prc_data["Deterministic"]

    return results, roc_data, prc_data


# %%
def tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, results_dir):
    # basic random foorest model
    basic_rfc = RandomForestClassifier(random_state=42)
    basic_rfc.fit(X_train, y_train)
    
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    basic_results, basic_roc_data, basic_prc_data, test_roc_data, test_prc_data = evaluate_classification(basic_rfc, splits, model_name="Random_Forest_Basic")
    save_model_results(basic_results, "Random_Forest_Basic", results_dir)
    
    plot_roc_curves(basic_roc_data, "Random_Forest_Basic", results_dir, filename='roc_curves.png')
    plot_prc_curves(basic_prc_data, "Random_Forest_Basic", results_dir, filename='prc_curves.png')
    plot_feature_importances(basic_rfc, "Random_Forest_Basic", feature_names, results_dir, filename='feature_importances.png')

    all_roc_data["Random_Forest_Basic"] = test_roc_data["Random_Forest_Basic"]
    all_prc_data["Random_Forest_Basic"] = test_prc_data["Random_Forest_Basic"]

    # hyperparameter-tuned random forest model
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 80, 100, 120, 200, 300, 400],
        'max_depth': [None, 3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', None]
    }
    #cv_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, scoring='accuracy', n_iter=20, cv=3, random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='balanced_accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    cv_rfc.fit(X_train, y_train)
    best_params = cv_rfc.best_params_
    save_best_params('Random_Forest_Optimized', best_params, results_dir)
    print("Best parameters:", best_params)

    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(cv_rfc.best_estimator_, splits, model_name="Random_Forest_Optimized")
    save_model_results(results, "Random_Forest_Optimized", results_dir)

    plot_roc_curves(roc_data, "Random_Forest_Optimized", results_dir, filename='roc_curves.png')
    plot_prc_curves(prc_data, "Random_Forest_Optimized", results_dir, filename='prc_curves.png')
    plot_feature_importances(cv_rfc.best_estimator_, "Random_Forest_Optimized", feature_names, results_dir, filename='feature_importances.png')

    all_roc_data["Random_Forest_Optimized"] = test_roc_data["Random_Forest_Optimized"]
    all_prc_data["Random_Forest_Optimized"] = test_prc_data["Random_Forest_Optimized"]

    return results, roc_data, prc_data


# %%
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    print("Best hyperparameters:\n", clf_grid.best_params_)
    return clf_grid.best_estimator_

def tune_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    # basic XGBoost model
    basic_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    basic_model.fit(X_train, y_train)
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    basic_results, basic_roc_data, basic_prc_data, test_roc_data, test_prc_data = evaluate_classification(basic_model, splits, model_name="XGBoost_Basic")
    save_model_results(basic_results, "XGBoost_Basic", results_dir)
    
    plot_roc_curves(basic_roc_data, "XGBoost_Basic", results_dir, filename='roc_curves.png')
    plot_prc_curves(basic_prc_data, "XGBoost_Basic", results_dir, filename='prc_curves.png')
    
    all_roc_data["XGBoost_Basic"] = test_roc_data["XGBoost_Basic"]
    all_prc_data["XGBoost_Basic"] = test_prc_data["XGBoost_Basic"]

    # hyperparameter-tuned XGBoost model
    xgb_param_grid = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2),
        'learning_rate': [0.0001, 0.01, 0.1],
        'n_estimators': [50, 200]
    }
    
    xgb_clf = xgb.XGBClassifier(random_state=0)
    xgb_opt = tune_clf_hyperparameters(xgb_clf, xgb_param_grid, X_train, y_train)
    # identify the best hyperparameters
    best_params = xgb_opt.get_params()
    save_best_params('XGBoost_Optimized', best_params, results_dir)

    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(xgb_opt, splits, model_name="XGBoost_Optimized")
    save_model_results(results, "XGBoost_Optimized", results_dir)

    plot_roc_curves(roc_data, "XGBoost_Optimized", results_dir, filename='roc_curves.png')
    plot_prc_curves(prc_data, "XGBoost_Optimized", results_dir, filename='prc_curves.png')

    all_roc_data["XGBoost_Optimized"] = test_roc_data["XGBoost_Optimized"]
    all_prc_data["XGBoost_Optimized"] = test_prc_data["XGBoost_Optimized"]

    return results, roc_data, prc_data

# %%
from sklearn.calibration import CalibratedClassifierCV

def tune_and_evaluate_linear_svc(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    # define the LinearSVC model
    linear_svc = LinearSVC(random_state=42, dual=False)  # dual=False when n_samples > n_features

    # define the parameter grid for GridSearchCV
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    # set up the GridSearchCV
    grid_search = GridSearchCV(linear_svc, param_grid, cv=5, n_jobs=-1, verbose=1)

    # fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # best estimator
    best_linear_svc = grid_search.best_estimator_

    # wrap the best LinearSVC model with CalibratedClassifierCV
    calibrated_svc = CalibratedClassifierCV(best_linear_svc, method='sigmoid', cv=5)
    calibrated_svc.fit(X_train, y_train)

    # save the best parameters
    save_best_params('Linear_SVC_Optimized', grid_search.best_params_, results_dir)

    # define splits
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]

    # evaluate the calibrated model using the existing evaluate_classification function
    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(calibrated_svc, splits, model_name="Linear_SVC_Optimized")

    # save and plot results
    save_model_results(results, "Linear_SVC_Optimized", results_dir)
    plot_roc_curves(roc_data, "Linear_SVC_Optimized", results_dir)
    plot_prc_curves(prc_data, "Linear_SVC_Optimized", results_dir)

    # store results for combined plotting
    all_roc_data["Linear_SVC_Optimized"] = test_roc_data["Linear_SVC_Optimized"]
    all_prc_data["Linear_SVC_Optimized"] = test_prc_data["Linear_SVC_Optimized"]

    return results, roc_data, prc_data


# %%
def tune_and_evaluate_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    # define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # compile the model, i.e., define the loss function and the optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

    # evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Neural Network Test accuracy:', test_acc)

    # prepare results for consistency, this step is to compare with other models
    test_predictions = (model.predict(X_test) > 0.5).astype("int32")
    test_pred_probs = model.predict(X_test).flatten()
    test_report = classification_report(y_test, test_predictions, output_dict=True)

    # calculate ROC and PRC data
    fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
    precision, recall, _ = precision_recall_curve(y_test, test_pred_probs)
    roc_auc = roc_auc_score(y_test, test_pred_probs)
    prc_auc = auc(recall, precision)

    results = {
        'train': ('Not Evaluated', {}),
        'val': ('Not Evaluated', {}),
        'test': (test_acc, test_report)
    }
    save_model_results(results, "Neural_Network", results_dir)

    # store ROC and PRC data for the test set
    test_roc_data = {"Neural_Network": (fpr, tpr, roc_auc)}
    test_prc_data = {"Neural_Network": (precision, recall, prc_auc)}

    all_roc_data["Neural_Network"] = test_roc_data["Neural_Network"]
    all_prc_data["Neural_Network"] = test_prc_data["Neural_Network"]

    # plot ROC and PRC curves
    plot_roc_curves(test_roc_data, "Neural_Network", results_dir, filename='roc_curves.png')
    plot_prc_curves(test_prc_data, "Neural_Network", results_dir, filename='prc_curves.png')

    return results, test_roc_data, test_prc_data


# %%
def evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    
    # grid search for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }
    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=10000), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)

    # identify best hyperparameters
    print("Best hyperparameters:", grid_search.best_params_)
    # save the best parameters
    save_best_params('Logistic_Regression_Best', grid_search.best_params_, results_dir)        
    best_lr = grid_search.best_estimator_
    
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    best_results, best_roc_data, best_prc_data, test_roc_data, test_prc_data = evaluate_classification(best_lr, splits, model_name="Logistic_Regression_Best")
    save_model_results(best_results, "Logistic_Regression_Best", results_dir)
    
    plot_roc_curves(best_roc_data, "Logistic_Regression_Best", results_dir, filename='roc_curves_best.png')
    plot_prc_curves(best_prc_data, "Logistic_Regression_Best", results_dir, filename='prc_curves_best.png')

    all_roc_data["Logistic_Regression_Best"] = test_roc_data["Logistic_Regression_Best"]
    all_prc_data["Logistic_Regression_Best"] = test_prc_data["Logistic_Regression_Best"]

    return best_results, best_roc_data, best_prc_data


# %%
def evaluate_elastic_net_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    
    # grid search for hyperparameter tuning with Elastic Net penalty
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],
        'solver': ['saga'],
        'penalty': ['elasticnet']
    }
    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=10000), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)
    
    best_enet_lr = grid_search.best_estimator_
    
    # identify best hyperparameters
    print("Best hyperparameters:", grid_search.best_params_)
    
    # save the best hyperparameters
    save_best_params('Elastic_Net_Logistic_Regression_Best', grid_search.best_params_, results_dir)
    
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    enet_results, enet_roc_data, enet_prc_data, test_roc_data, test_prc_data = evaluate_classification(best_enet_lr, splits, model_name="Elastic_Net_Logistic_Regression_Best")
    save_model_results(enet_results, "Elastic_Net_Logistic_Regression_Best", results_dir)
    
    plot_roc_curves(enet_roc_data, "Elastic_Net_Logistic_Regression_Best", results_dir, filename='roc_curves_best.png')
    plot_prc_curves(enet_prc_data, "Elastic_Net_Logistic_Regression_Best", results_dir, filename='prc_curves_best.png')

    all_roc_data["Elastic_Net_Logistic_Regression_Best"] = test_roc_data["Elastic_Net_Logistic_Regression_Best"]
    all_prc_data["Elastic_Net_Logistic_Regression_Best"] = test_prc_data["Elastic_Net_Logistic_Regression_Best"]

    return enet_results, enet_roc_data, enet_prc_data

# %%
def tune_and_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    # define the KNN model
    knn = KNeighborsClassifier()

    # define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # perform Grid Search
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # get the best estimator
    best_knn = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    
    # save the best hyperparameters
    save_best_params('KNN_Optimized', grid_search.best_params_, results_dir)
    
    # define splits
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]

    # evaluate the model
    results, roc_data, prc_data, test_roc_data, test_prc_data = evaluate_classification(best_knn, splits, model_name="KNN_Optimized")
    
    # save results
    save_model_results(results, "KNN_Optimized", results_dir)
    
    # plot ROC and PRC curves
    plot_roc_curves(roc_data, "KNN_Optimized", results_dir, filename='roc_curves.png')
    plot_prc_curves(prc_data, "KNN_Optimized", results_dir, filename='prc_curves.png')

    # store ROC and PRC data for the test set
    all_roc_data["KNN_Optimized"] = test_roc_data["KNN_Optimized"]
    all_prc_data["KNN_Optimized"] = test_prc_data["KNN_Optimized"]

    return results, roc_data, prc_data

# %%
def evaluate_majority_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    
    # train a dummy classifier that predicts the majority class
    majority_class_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    majority_class_clf.fit(X_train, y_train)
    
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    majority_results, majority_roc_data, majority_prc_data, test_roc_data, test_prc_data = evaluate_classification(majority_class_clf, splits, model_name="Majority_Class_Classifier")
    save_model_results(majority_results, "Majority_Class_Classifier", results_dir)
    
    plot_roc_curves(majority_roc_data, "Majority_Class_Classifier", results_dir, filename='roc_curves.png')
    plot_prc_curves(majority_prc_data, "Majority_Class_Classifier", results_dir, filename='prc_curves.png')

    all_roc_data["Majority_Class_Classifier"] = test_roc_data["Majority_Class_Classifier"]
    all_prc_data["Majority_Class_Classifier"] = test_prc_data["Majority_Class_Classifier"]

    return majority_results, majority_roc_data, majority_prc_data

# %%
def evaluate_Chance_Class_Classifier(X_train, y_train, X_val, y_val, X_test, y_test, results_dir):
    
    # train a dummy classifier that predicts a random class
    random_class_clf = DummyClassifier(strategy='uniform', random_state=42)
    random_class_clf.fit(X_train, y_train)
    
    splits = [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]
    random_results, random_roc_data, random_prc_data, test_roc_data, test_prc_data = evaluate_classification(random_class_clf, splits, model_name="Chance_Class_Classifier")
    save_model_results(random_results, "Chance_Class_Classifier", results_dir)
    
    plot_roc_curves(random_roc_data, "Chance_Class_Classifier", results_dir, filename='roc_curves.png')
    plot_prc_curves(random_prc_data, "Chance_Class_Classifier", results_dir, filename='prc_curves.png')

    all_roc_data["Chance_Class_Classifier"] = test_roc_data["Chance_Class_Classifier"]
    all_prc_data["Chance_Class_Classifier"] = test_prc_data["Chance_Class_Classifier"]

    return random_results, random_roc_data, random_prc_data

# %%
def run_all_models(data_dir, results_dir):
    global all_roc_data, all_prc_data
    all_roc_data = {}
    all_prc_data = {}
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data(data_dir)
    
    # create a separate directory for this dataset's results
    dataset_name = data_dir.stem
    dataset_results_dir = results_dir / dataset_name
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # SVC
    SVC_results, SVC_roc_data, SVC_prc_data = tune_and_evaluate_linear_svc(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)

    # Random Forest
    results_rf, roc_data_rf, prc_data_rf = tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, dataset_results_dir)
    
    # XGBoost
    results_xgb, roc_data_xgb, prc_data_xgb = tune_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # Logistic Regression
    basic_results, basic_roc_data, basic_prc_data = evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # Elastic Net Logistic Regression
    enet_results, enet_roc_data, enet_prc_data = evaluate_elastic_net_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # KNN Classifier
    knn_results, knn_roc_data, knn_prc_data = tune_and_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # Majority Class Classifier
    majority_results, majority_roc_data, majority_prc_data = evaluate_majority_class_classifier(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # Random Class Classifier
    random_results, random_roc_data, random_prc_data = evaluate_Chance_Class_Classifier(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    # Neural Network
    results_nn, roc_data_nn, prc_data_nn = tune_and_evaluate_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, dataset_results_dir)
    
    
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



# %% [markdown]
# #root / "data" / "backup"]
# #root / "data" / "CLR",
# #root / "data" / "CLR_nonreduced",
# #root / "data" / "CLR_PCA", 
# #root / "data" / "CLR_SVD", 
# #root / "data" / "reduced_0_1", 
# #root / "data" / "reduced_0_1_PCA", 
# #root / "data" / "reduced_0_1_SVD", 
# #root / "data" / "baseline_demographic",
# #root / "data" / "non_reduced"]

# %%
def main():
    root = Path.cwd().parents[1]
    data_dir = root / "data"
    results_dir = root / "results" / "model_reports"
    
    for current_dir in data_dir.iterdir():
        if current_dir.is_dir():
            # skip the directory if it's named 'raw'
            if current_dir.name == 'raw':
                print(f"Skipping {current_dir}")
                continue
            print(f"Now processing {current_dir}")
            run_all_models(current_dir, results_dir)

main()


# %% [markdown]
# 

# %%


# %%



