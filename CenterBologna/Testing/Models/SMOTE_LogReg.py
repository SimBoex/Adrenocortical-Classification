import  os
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler 
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImblearnPipeline

from Components import logger
import pickle

Customlogger = logger.createLogger("SMOTE_LogReg model",False)

# this the class used for standardize the dataframe
class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = []  # To store column names after scaling
    
    def fit(self, X, y=None):
        Customlogger.debug(f"Fitting the StandardScaler on {X.shape}")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")        
        self.scaler.fit(X, y)        
        self.feature_names_ = X.columns
        return self
    
    def transform(self, X):
        # Verify that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names_)
## --------------------------------------------------------------------------------------------
# This is the class used for perforing the oversampling using SMOTE
class SMOTETransformer(BaseEstimator):
    def __init__(self, sampling_strategy='auto', random_state=42,k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X, y):
        Customlogger.debug(f"SMOTETransformer is fitting on  {X.shape} samples with {len(y)} labes of which {sum(y)} are positive and {len(y) -  sum(y)} are negative")
        smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state,k_neighbors=self.k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        if isinstance(X, pd.DataFrame):
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            return X_resampled_df, y_resampled
        else:
            return X_resampled, y_resampled


def Define_Pipeline():
    Customlogger.info("Defining the pipeline...")
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pipeline = ImblearnPipeline([
        ('scaler', DataFrameStandardScaler()),  
        ('smote', SMOTETransformer()),
        ('logistic_regression', LogisticRegression(random_state=42))
    ])
    return pipeline


'''
This returns a Grid-search object with the metrics
'''
def Grid_Search(pipeline,X_train,y_train):
    
    # define the cross validation (stratified)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    param_grid = {
    'smote__sampling_strategy': ['auto'],
    'smote__k_neighbors': [2,5,8],
    'logistic_regression__C': [0.1, 1, 10],
    'logistic_regression__penalty': ['l2'],
    'logistic_regression__solver': ['newton-cg','lbfgs','liblinear'],
    'logistic_regression__max_iter': [300],
}

    grid_search = GridSearchCV(pipeline, param_grid, cv= cv, scoring='f1',verbose=1,n_jobs=3,error_score='raise')
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
    with open('Elements/SMOTE_LogReg/{}fit_timeCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(fit_time   , f)
        
    test_precision = cv_results["test_precision"]
    print('Mean precision is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_precision), np.std(test_precision)))
    # Save the set
    with open('Elements/SMOTE_LogReg/{}precisionCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_precision  , f)
    
    test_recall = cv_results["test_recall"]
    print('Mean recall is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_recall), np.std(test_recall)))
    # Save the set
    with open('Elements/SMOTE_LogReg/{}recallCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_recall  , f)
        
    test_f1= cv_results["test_f1"]
    print('Mean f1 is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_f1), np.std(test_f1)))
    # Save the set
    with open('Elements/SMOTE_LogReg/{}F1CV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_f1  , f)

    test_roc_auc = cv_results['test_roc_auc']
    print('Mean roc_auc is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_roc_auc), np.std(test_roc_auc)))

    # Save the set
    with open('Elements/SMOTE_LogReg/{}aucCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_roc_auc , f)
        
    test_accuracy = cv_results['test_accuracy']
    print('Mean accuracy is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_accuracy), np.std(test_accuracy)))

    # Save the set
    with open('Elements/SMOTE_LogReg/{}accuracyCV.pkl'.format(Prefix), 'wb') as f:
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
    joblib.dump(pipeline, 'Elements/SMOTE_LogReg/{}SMOTE_LogRegFitted_pipeline.pkl'.format(Prefix))
    
    ReduceDataset = X_train
    ReduceDataset.to_excel("Elements/SMOTE_LogReg/{}SMOTE_LogReg_ReduceDataset.xlsx".format(Prefix), index=False)
    Customlogger.info(f"saving the resulting dataframe into: SMOTE_LogReg_ReduceDataset.xlsx")
    return pipeline


import seaborn as sns
import matplotlib.pyplot as plt

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



