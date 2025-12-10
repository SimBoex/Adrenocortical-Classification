
from Components import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ttest_ind
import pandas as pd
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold,cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from collections import Counter

import random
import os
import joblib
import time
import pickle


Customlogger = logger.createLogger("MultiClass classification: Multi-level, SMOTE, LogReg pipeline ", False)
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
# This is the class used for filtering according to the variance
class DataFrameVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variance_threshold = VarianceThreshold(threshold)
        self.feature_names_ = []  
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("L'input deve essere un pandas DataFrame.")
        Customlogger.debug(f"DataFrameVarianceThreshold {X.shape} samples with a threshold of {self.threshold}")
        self.variance_threshold.fit(X, y)        
        self.feature_names_ = X.columns[self.variance_threshold.get_support()]
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("L'input deve essere un pandas DataFrame.")        
        X_transformed = self.variance_threshold.transform(X)
        output = pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)
        Customlogger.debug(f"DataFrameVarianceThreshold returns  {output.shape} samples with a threshold of {self.threshold}")
        return output
    
## --------------------------------------------------------------------------------------------
# This is the class used for performing the Welchs t-test 
class GroupedTTestFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.significant_features_ = None

    def fit(self, X, y):
        Customlogger.debug(f"GroupedTTestFeatureSelector is fitting on {X.shape} samples with a threshold of {self.threshold}")
        
        group1_mask = np.isin(y, [0, 1, 5])
        group2_mask = np.isin(y, [2, 3, 4])

        group1 = X[group1_mask]
        group2 = X[group2_mask]

       
        p_values = [ttest_ind(group1.iloc[:, i], group2.iloc[:, i], equal_var=False).pvalue for i in range(X.shape[1])]
        
        # Seleziona le caratteristiche significative
        self.significant_features_ = [i for i, p_value in enumerate(p_values) if p_value < self.threshold]

        return self
    
    def transform(self, X):
        selected_columns = X.iloc[:, self.significant_features_]
        return pd.DataFrame(selected_columns, columns=X.columns[self.significant_features_])

    
## --------------------------------------------------------------------------------------------
# This is the class used for perforing the oversampling using SMOTE
class SMOTETransformer(BaseEstimator):
    def __init__(self, sampling_strategy='auto', random_state=42,k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X, y):

        class_counts = Counter(y)
        class_info = ", ".join([f"{key}: {value}" for key, value in class_counts.items()])
        Customlogger.debug(f"SMOTETransformer is fitting on {X.shape[0]} samples with the following class distribution: {class_info}")

        smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state,k_neighbors=self.k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        if isinstance(X, pd.DataFrame):
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)

            class_counts = Counter(y_resampled)
            class_info = ", ".join([f"{key}: {value}" for key, value in class_counts.items()])
            Customlogger.debug(f"SMOTETransformer creates {X_resampled.shape[0]} samples with the following class distribution: {class_info}")

            smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state,k_neighbors=self.k_neighbors)

            return X_resampled_df, y_resampled
        else:
            return X_resampled, y_resampled
        
## --------------------------------------------------------------------------------------------
# This is the class used for performing the LASSO and to find the best coefficient
class LassoCoeffSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha = None, coe_step=0.001, random_state=42):
        self.alpha = alpha
        self.random_state = random_state
        self.relevant_features_indexes = None
        self.lasso_coefs = None        
        self.coe_step = coe_step
        self.Entire_y = None
        self.intercept = None
        self.best_coef = None
        self.time = True
    def fit(self, Entire_X, Entire_y):
        assert self.alpha != None
        self.Entire_y = Entire_y
        Customlogger.debug(f"LassoCoeffSelector is fitting on  {Entire_X.shape} samples with {len(Entire_y)} labes of which {sum(Entire_y)} are positive and {len(Entire_y) -  sum(Entire_y)} are negative")      
        lasso = Lasso(random_state=42, alpha=self.alpha, max_iter=100000).fit(Entire_X, Entire_y)
        self.intercept = lasso.intercept_
        # saving the obtained coefficients
        self.lasso_coefs  = lasso.coef_
        # saving the indexes of the relevant features
        self.relevant_features_indexes = lasso.coef_[np.where(lasso.coef_ != 0)]     
        if len(self.relevant_features_indexes) == 0:
            raise Exception("Relevant features indexes are empty.")
        Customlogger.debug(f"the shape of the dataframe  after Lasso but before the coefficient threshold is {len(self.relevant_features_indexes)}")
        best_coef, best_mse = self.CoefficientTh(coe_step=self.coe_step,lasso_coefs=self.lasso_coefs,Entire_X=Entire_X, Entire_y=self.Entire_y,intercept=self.intercept)
        self.best_coef = best_coef
        self.best_mse = best_mse
        self.selected_features_indexes = np.where(self.abs_lasso_coefs > best_coef)[0]

        return self


    def transform(self,  Entire_X):
        Customlogger.debug(f"applying the threshold {self.best_coef} found during the training (using the Lasso alg) on the features coefficients to filter them out")

        X_selected = Entire_X.iloc[:, self.selected_features_indexes] 
        X_selected  = pd.DataFrame(X_selected, columns=Entire_X.columns[self.selected_features_indexes])
        Customlogger.debug(f"the resulting dataframe has the following shape {X_selected.shape}")
        return X_selected 
    

    def CoefficientTh(self,coe_step,lasso_coefs,Entire_X,Entire_y,intercept):
        # it loads the coefficients found using Lasso on the entire db
        self.abs_lasso_coefs = np.abs(lasso_coefs)

        # here i compute the min and max coeffs
        hmax = np.max(self.abs_lasso_coefs)
        hmin = np.min(self.abs_lasso_coefs)
        Customlogger.debug(f"the absolute values of the  coefficients of the LASSO fitted on the entire dataset is: {hmin} and {hmax}")
        Customlogger.debug(f"splitting the dataset in half and extracting X_D1")
        # here i split the entire trainig db in 2: one for the training and one for the test
        X_train_toy, X_D1, y_train_toy, y_D1 = train_test_split(Entire_X, Entire_y, test_size=0.5, random_state=self.random_state)
        Customlogger.debug(f"The shape of X_D1 is {X_D1.shape} and the length of y_DI is {len(y_D1)} of which {sum(y_D1)} are positive and {len(y_D1) -  sum(y_D1)} are negative")
        Customlogger.debug(f"finding the best threshold...")
        coe_thr = hmin
        best_mse = np.inf
        best_coe_thr = coe_thr
        
        # here for each candidate th:
        while coe_thr <= hmax :
            Customlogger.debug(f"extracting the features from the original dataset with a coefficient larger than {round(coe_thr,4)}")
            # i take the index corrisponding to the coeff more than the threshold
            selected_features_indexes = np.where(self.abs_lasso_coefs > coe_thr)[0]
            if len(selected_features_indexes) == 0:
                break
            #  I take the corrisponding features  on the training db
            X_selected =  X_train_toy.iloc[:, selected_features_indexes] # extracting from the toy-test dataset the features
            Customlogger.debug(f"computing MSE fitting again")
            X_selected  = pd.DataFrame(X_selected, columns=Entire_X.columns[selected_features_indexes])
            Customlogger.debug(f"the shape of the dataframe corrisponding is {X_selected.shape}")

            # i fit the training 
            lasso_D1 = Lasso(random_state=42, alpha=self.alpha, max_iter=100000).fit(X_selected,y_train_toy)

            # I prepare the test as before using the coefficients more than the th
            X_selected_test =  X_D1.iloc[:, selected_features_indexes] # extracting from the toy-test dataset the features
            X_selected_test = pd.DataFrame(X_selected_test, columns=Entire_X.columns[selected_features_indexes])

            # I predict 
            y_pred = lasso_D1.predict(X_selected_test)
            Customlogger.debug(f"the shape of the prediction  is {y_pred.shape}")

            # i measure the MSE
            mse = mean_squared_error(y_D1, y_pred)
            Customlogger.debug(f"the MSE  is {mse}")

            # take the best th
            if mse < best_mse:
                best_mse = mse
                best_coe_thr = coe_thr
            coe_thr += coe_step

        Customlogger.debug(f"returning the best threshold {round(best_coe_thr,4)} with a MSE of {best_mse}")
        return best_coe_thr, best_mse



''' 
This returns the pipeline 
'''
def Define_Pipeline():
    Customlogger.info("Defining the Pipeline...")
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    pipeline = ImblearnPipeline([
    ('variance_threshold', DataFrameVarianceThreshold(threshold=0)), 
    ('welchs_ttest_selector', GroupedTTestFeatureSelector(threshold=0.05)),
    ('scaler', DataFrameStandardScaler()),  
    ('smote', SMOTETransformer()),
    ('LassoCoeffSelector',LassoCoeffSelector(random_state=42)),
    ('logistic_regression', LogisticRegression(random_state=42))
    ])
    return pipeline



'''
This returns a Grid-search object with the metrics
'''
def Grid_Search(pipeline,X_train,y_train):
    values_alphas = [0.01] + [i/500 for i in range(2, 101)]
    
    # define the cross validation (stratified)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    param_grid = {
        'LassoCoeffSelector__alpha': values_alphas,
        'smote__sampling_strategy': ['auto'],
        'smote__k_neighbors': [2,5],
        'logistic_regression__C': [0.1, 1, 10],
        'logistic_regression__penalty': ['l2'],
        'logistic_regression__solver': ['newton-cg','lbfgs','liblinear'],
        'logistic_regression__max_iter': [500],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv= cv, scoring='f1_macro',verbose=1, n_jobs = 3)
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
    with open('Elements/Multi_SMOTE_LogReg/{}fit_timeCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(fit_time   , f)
        
    test_precision = cv_results["test_precision_macro"]
    print('Mean precision is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_precision), np.std(test_precision)))
    # Save the set
    with open('Elements/Multi_SMOTE_LogReg/{}precisionCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_precision  , f)
    
    test_recall = cv_results["test_recall_macro"]
    print('Mean recall is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_recall), np.std(test_recall)))
    # Save the set
    with open('Elements/Multi_SMOTE_LogReg/{}recallCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_recall  , f)
        
    test_f1= cv_results["test_f1_macro"]
    print('Mean f1 is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_f1), np.std(test_f1)))
    # Save the set
    with open('Elements/Multi_SMOTE_LogReg/{}F1CV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_f1  , f)

    test_roc_auc = cv_results['test_AUC']
    print('Mean roc_auc is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_roc_auc), np.std(test_roc_auc)))

    # Save the set
    with open('Elements/Multi_SMOTE_LogReg/{}aucCV.pkl'.format(Prefix), 'wb') as f:
        pickle.dump(test_roc_auc , f)
        
    test_accuracy = cv_results['test_accuracy']
    print('Mean accuracy is: {:.3f} and Std is: {:.3f}'.format(np.mean(test_accuracy), np.std(test_accuracy)))

    # Save the set
    with open('Elements/Multi_SMOTE_LogReg/{}accuracyCV.pkl'.format(Prefix), 'wb') as f:
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
    joblib.dump(pipeline, 'Elements/Multi_SMOTE_LogReg/{}Multi_SMOTE_LogRegFitted_pipeline.pkl'.format(Prefix))
    
    ReduceDataset= pipeline[:-1].transform(X_train)

    ReduceDataset.to_excel("Elements/Multi_SMOTE_LogReg/{}Multi_SMOTE_LogReg_ReduceDataset.xlsx".format(Prefix), index=False)
    Customlogger.info(f"saving the resulting dataframe into: Multi_SMOTE_LogReg_ReduceDataset.xlsx")
    return pipeline


from sklearn.metrics import  accuracy_score, confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

'''
INPUT: PIPELINE WITH ITS HYPERPARAMETERS ALREADY SET, TEST DATASET, LABELS
OUTPUT: PREDICTIONS CORRISPONDING
'''
from sklearn.metrics import classification_report

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
