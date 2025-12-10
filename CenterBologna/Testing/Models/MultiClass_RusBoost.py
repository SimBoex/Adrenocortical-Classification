import  os
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,precision_recall_curve, average_precision_score
import pandas as pd


from Components import logger
import pickle

Customlogger = logger.createLogger("Multi Class Classification: RusBoost model ",False)


def Define_Pipeline():
    Customlogger.info("Defining the pipeline...")
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    model = RUSBoostClassifier(random_state=42,estimator=DecisionTreeClassifier(max_depth=1,random_state = 42))
    pipeline = Pipeline([
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
    'rusboostclassifier__n_estimators': [300],  # Parameters  for RUSBoost
    'rusboostclassifier__learning_rate': [ 0.05,0.01]
    }
    Customlogger.info(f"Let's define the parameters for the grid search: {param_grid}")

    grid_search = GridSearchCV(pipeline, param_grid, cv= cv, scoring='f1_macro', n_jobs=3)
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


def Testing_hyperparameters(pipeline, X_train, y_train, Prefix = "Multi_"):
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
    with open('Elements/RusBoost/{}fit_timeCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(fit_time   , f)
        
    test_precision = cv_results["test_precision_macro"]
    print('Mean precision is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_precision), np.std(test_precision)))
    # Save the set
    with open('Elements/RusBoost/{}precisionCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_precision  , f)
    
    test_recall = cv_results["test_recall_macro"]
    print('Mean recall is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_recall), np.std(test_recall)))
    # Save the set
    with open('Elements/RusBoost/{}recallCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_recall  , f)
        
    test_f1= cv_results["test_f1_macro"]
    print('Mean f1 is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_f1), np.std(test_f1)))
    # Save the set
    with open('Elements/RusBoost/{}F1CV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_f1  , f)
        
    test_auc = cv_results["test_AUC"] 
    print('Mean AUC (OvR) is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_auc), np.std(test_auc)))
    # Save the set
    with open('Elements/RusBoost/{}aucCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_auc, f) 
        

    test_accuracy = cv_results['test_accuracy']
    print('Mean accuracy is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_accuracy), np.std(test_accuracy)))

    # Save the set
    with open('Elements/RusBoost/{}accuracyCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_accuracy  , f)
        
import time
import joblib

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
    joblib.dump(pipeline, 'Elements/RusBoost/{}RusBoostFitted_pipeline.pkl'.format(Prefix))
    
    ReduceDataset = X_train
    ReduceDataset.to_excel("Elements/RusBoost/{}RusBoost_ReduceDataset.xlsx".format(Prefix), index=False)
    Customlogger.info("saving the resulting dataframe into: {}RusBoost_ReduceDataset.xlsx".format(Prefix))
    return pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

'''
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TEST DATASET, LABELS
OUTPUT: PREDICTIONS CORRISPONDING
'''

def test(pipeline, X_test, y_test,  Curves = False):
    predictions = pipeline.predict(X_test)

    probs = pipeline.predict_proba(X_test)

    # Generate classification report
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
