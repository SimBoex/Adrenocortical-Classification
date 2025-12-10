from Components import logger
import pandas as pd
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import random 
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold,cross_validate

import joblib
import time


Customlogger = logger.createLogger("Multi-class classification: Buruta and RusBoost pipeline ",False)

class BorutaAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, estimator = RandomForestClassifier(random_state=42), max_iter=100):
        Customlogger.debug(f"Defining the Boruta instance")
        self.estimator = estimator
        self.max_iter = max_iter
        self.feature_selector = None
        self.columns_names = []

    def fit(self, X, y):
        Customlogger.debug(f"the dataset has  {X.shape} samples and {len(y)} labels")
        # to fix the package!
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_
        
        self.feature_selector = BorutaPy(self.estimator, 
                                         max_iter=self.max_iter, random_state=42,verbose=0)
        self.feature_selector.fit(X.values, y.values)
        # taking the names!
        columns_names = []
        for i in range(len(self.feature_selector.support_)):
            if self.feature_selector.support_[i]:
                columns_names.append(X.columns[i])
        self.columns_names = columns_names
        Customlogger.debug(f"The boruta algorithm has selected {len(columns_names)} features")
        return self

    def transform(self, X):
        Customlogger.debug(f"the batch of data to transform is {X.shape} samples")
        filtered_values = self.feature_selector.transform(X.values)
        filtered_values = pd.DataFrame(filtered_values, columns=self.columns_names)
        Customlogger.debug(f"therefore, the dataset has  {filtered_values.shape} samples")
        return filtered_values

''' 
This returns the pipeline 
'''
def Define_Pipeline():
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    Customlogger.info("Defining the Pipeline...")
    model = RUSBoostClassifier(random_state=42,estimator=DecisionTreeClassifier(max_depth=1,random_state = 42))
    Customlogger.info(f"Let's define the pipeline: Boruta Adapter with a random forest classifier and {model}")
    pipeline = Pipeline([
        ('feature_selection',BorutaAdapter()),
        ('rusboostclassifier',model)
    ])
    return pipeline

'''
This returns a Grid-search object with the metrics
'''
def Grid_Search(pipeline,X_train,y_train):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=42)
    Customlogger.info(f"Let's define the cross validation:{cv}")

    param_grid = {
        'feature_selection__estimator' : [RandomForestClassifier(random_state=42)],
        'feature_selection__estimator__n_estimators': [50,100],  # Parameters for the  RandomForest 
        'feature_selection__estimator__max_depth': [3,5],  # Parameters for the RandomForest
        'rusboostclassifier__n_estimators': [300],  # Parameters  for RUSBoost
        'rusboostclassifier__learning_rate': [ 0.05,0.01]
    }
    Customlogger.info(f"Let's define the parameters for the grid search: {param_grid}")

    grid_search = GridSearchCV(pipeline, param_grid, cv= cv, scoring='f1_macro',n_jobs=3)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result


''' 
This is computing a CV with 5 folds 50 repeats and saving its metrics
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TRAINING DATASET, LABELS
OUTPUT: NONE (METRICS SAVED IN THE CORRISPONDING ELEMENTS DIR)
'''
from sklearn.metrics import precision_score, make_scorer

precision_macro_scorer = make_scorer(precision_score, average='macro', zero_division=0)

def Testing_hyperparameters(pipeline, X_train, y_train,Prefix = "Multi_"):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=42)
    
    scoring = {
    'accuracy': 'accuracy',
    'precision_macro': precision_macro_scorer,
    'recall_macro': 'recall_macro',
    "f1_macro": 'f1_macro',
    'AUC': 'roc_auc_ovr'
    }
    
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    fit_time = cv_results["fit_time"]
    print('Mean fit_time  is: {:.3f} and Std is: {:.3f}'.format(np.mean(fit_time ), np.std(fit_time )))
    # Save the set
    with open('Elements/Boruta_RusBoost/{}fit_timeCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(fit_time   , f)
        
    test_precision = cv_results["test_precision_macro"]
    print('Mean precision is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_precision), np.std(test_precision)))
    # Save the set
    with open('Elements/Boruta_RusBoost/{}precisionCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_precision  , f)
    
    test_recall = cv_results["test_recall_macro"]
    print('Mean recall is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_recall), np.std(test_recall)))
    # Save the set
    with open('Elements/Boruta_RusBoost/{}recallCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_recall  , f)
        
    test_f1= cv_results["test_f1_macro"]
    print('Mean f1 is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_f1), np.std(test_f1)))
    # Save the set
    with open('Elements/Boruta_RusBoost/{}F1CV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_f1  , f)
        
    test_auc = cv_results["test_AUC"] 
    print('Mean AUC (OvR) is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_auc), np.std(test_auc)))
    # Save the set
    with open('Elements/Boruta_RusBoost/{}aucCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_auc, f) 
        

    test_accuracy = cv_results['test_accuracy']
    print('Mean accuracy is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_accuracy), np.std(test_accuracy)))

    # Save the set
    with open('Elements/Boruta_RusBoost/{}accuracyCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_accuracy  , f)

    
'''
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TRAINING DATASET, LABELS
OUTPUT: FITTED PIPELINE
'''
def fit(pipeline, X_train, y_train, Prefix = "Multi_"):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time  = round(elapsed_time,5)
    print(f"Time taken to fit the model: {elapsed_time} seconds")
    joblib.dump(pipeline, 'Elements/Boruta_RusBoost/{}Boruta_RusBoostFitted_pipeline.pkl'.format(Prefix))
    
    ReduceDataset = pipeline.named_steps['feature_selection'].transform(X_train)
    ReduceDataset.to_excel("Elements/Boruta_RusBoost/{}Boruta_RusBoost_ReduceDataset.xlsx".format(Prefix), index=False)
    return pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

'''
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TEST DATASET, LABELS
OUTPUT: PREDICTIONS CORRISPONDING
'''
def test(pipeline, X_test, y_test, Curves = False ):
    predictions = pipeline.predict(X_test)

    probs = pipeline.predict_proba(X_test)

    report = classification_report(y_test, predictions)
    print(report)
    
    if Curves:
        MultiClass_CURVES(y_test, probs)
    
    return predictions
    


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score 
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

def MultiClass_CURVES(y_test, probs):
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2,3,4,5])  # Adjust classes accordingly
    n_classes = y_test_binarized.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute Precision-Recall and average precision
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], probs[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], probs[:, i])

    # Plot ROC curves
    plt.figure(figsize=(12, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall curves
    plt.figure(figsize=(12, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'Precision-Recall curve of class {i} (area = {average_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
