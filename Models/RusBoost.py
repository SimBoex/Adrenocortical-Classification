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

Customlogger = logger.createLogger("RusBoost model",False)


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
def Grid_Search(pipeline,cv, X_train,y_train):
    Customlogger.info(f"Let's define the cross validation:{cv}")

    param_grid = {
    'rusboostclassifier__n_estimators': [150, 300],  # Parameters  for RUSBoost
    'rusboostclassifier__learning_rate': [ 0.05,0.01]
    }
    Customlogger.info(f"Let's define the parameters for the grid search: {param_grid}")

    grid_search = GridSearchCV(pipeline, param_grid, cv= cv, scoring='f1',n_jobs=3)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result



''' 
This is computing a CV with 5 folds 50 repeats and saving its metrics
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TRAINING DATASET, LABELS
OUTPUT: NONE (METRICS SAVED IN THE CORRISPONDING ELEMENTS DIR)
'''
def Testing_hyperparameters(pipeline, X_train, y_train, Prefix = ""):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=42)
    scoring = ['accuracy', 'precision', 'recall',"f1","roc_auc"]
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    fit_time = cv_results["fit_time"]
    print('Mean fit_time  is: {:.3f} and Std is: {:.3f}'.format(np.mean(fit_time ), np.std(fit_time )))
    # Save the set
    with open('Elements/RusBoost/{}fit_timeCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(fit_time   , f)
        
    test_precision = cv_results["test_precision"]
    print('Mean precision is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_precision), np.std(test_precision)))
    # Save the set
    with open('Elements/RusBoost/{}precisionCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_precision  , f)
    
    test_recall = cv_results["test_recall"]
    print('Mean recall is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_recall), np.std(test_recall)))
    # Save the set
    with open('Elements/RusBoost/{}recallCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_recall  , f)
        
    test_f1= cv_results["test_f1"]
    print('Mean f1 is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_f1), np.std(test_f1)))
    # Save the set
    with open('Elements/RusBoost/{}F1CV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_f1  , f)

    test_roc_auc = cv_results['test_roc_auc']
    print('Mean roc_auc is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_roc_auc), np.std(test_roc_auc)))

    # Save the set
    with open('Elements/RusBoost/{}aucCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_roc_auc , f)
        
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
def fit(pipeline, X_train, y_train, Prefix = ""):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time  = round(elapsed_time,5)
    print(f"Time taken to fit the model: {elapsed_time} seconds")
    joblib.dump(pipeline, 'Elements/RusBoost/{}RusBoostFitted_pipeline.pkl'.format(Prefix))
    
    ReduceDataset = X_train
    ReduceDataset.to_excel("Elements/RusBoost/{}RusBoost_ReduceDataset.xlsx".format(Prefix), index=False)
    Customlogger.info(f"saving the resulting dataframe into: RusBoost_ReduceDataset.xlsx")
    return pipeline

import matplotlib.pyplot as plt
import seaborn as sns
'''
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TEST DATASET, LABELS
OUTPUT: PREDICTIONS CORRISPONDING
'''
def test(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)

    probs = pipeline.predict_proba(X_test)

    Accuracy = round(accuracy_score(y_test, predictions),3)
    precision = round(precision_score(y_test, predictions),3)
    recall = round(recall_score(y_test, predictions),3)
    f1 = round(f1_score(y_test, predictions),3)
    probs = probs[:, 1]

    auc_roc = round(roc_auc_score(y_test,probs),3)
    
    
    conf_matrix = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    specificity = round(specificity,3)
    Customlogger.info(f" the performances are \n Precision: {precision} \n Recall: {recall} \n F1-score: {f1} \n Specificity: {specificity} \n Accuracy {Accuracy} \n AUC_ROC {auc_roc}")
    # Create a heatmap of the confusion matrix
    
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size for readability
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return predictions




