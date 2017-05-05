import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from imblearn.under_sampling import RandomUnderSampler

import random
from scipy import optimize
import time
import preprocess


'''
The following functions were modified from the code of rayidghani (Github ID), https://github.com/rayidghani/magicloops/blob/master/magicloops.py:
define_clfs_params, generate_binary_at_k, evaluate_at_k, plot_precision_recall_n, clf_loop
'''

def define_clfs_params(grid_size):
    '''

    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }


    medium_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.001,0.1,1],'kernel':['linear']},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'KNN' :{'n_neighbors': [1,5,25,50],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']}
            }
    

    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'medium'):
        return clfs, medium_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0



def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def evaluate_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f_score = f1_score(y_true, preds_at_k)
    return (precision, recall, f_score)


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()



NOTEBOOK = 1

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    
    results_df =  pd.DataFrame(columns=('Model','Classifier', 'Parameters', 'AUC-ROC', "f1_score_5", "f1_score_10", "f1_score_20", 'Accuracy', 'Precision_5', 
                                        'Precision_10', 'Precision_20', 'Recall_5', 'Recall_10','Recall_20'))

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print (models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                
                accuracy = clf.score(X_test, y_test)
                roc = roc_auc_score(y_test, y_pred_probs)
                p5, r5, f5 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 5.0)
                p10, r10, f10 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 10.0)
                p20, r20, f20 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 20.0)
                
                results_df.loc[len(results_df)] = [models_to_run[index], clf, p, roc, f5, f10, f20, accuracy,                                               
                                                   p5, p10, p20, r5, r10, r20]

                if NOTEBOOK == 1:
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
            except IndexError as e:
                print ('Error:',e)
                continue
    #For a base case when result is always 0

    y_prob = np.zeros(X_test.shape[0])
    accuracy = clf.score(X_test, y_test)
    roc = roc_auc_score(y_test, y_pred_probs)
    p5, r5, f5 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 5.0)
    p10, r10, f10 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 10.0)
    p20, r20, f20 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, 20.0)
                
    results_df.loc[len(results_df)] = ["BASE_ZERO_CASE", "BASE_ZERO_CASE", None, roc, f5, f10, f20, accuracy,                                               
                                                   p5, p10, p20, r5, r10, r20]

        
    return results_df



def find_best_classifier_by_model(result_df, eval_method):
    '''
    Find the best classifier by each model
    Input: 
        - result_df: pandas dataframe of evaluation values
        - eval_method: evaluation method (accuracy, recall, etc.) to check
    Output: pandas dataframe of best classifier by each model
    '''
    
    tracker = {}
    index_list = []
    
    for index, row in result_df.iterrows():
        model = row['Model'][:2] 
        score = row[eval_method]
        if model not in tracker:
            tracker[model] = [(score, index)]
        else:
            if score > tracker[model][0][0]:
                tracker[model] = [(score, index)]
            if score == tracker[model][0][0]:
                tracker[model].append((score, index))
                
    for value in tracker.values():
        for tup in value: 
            index_list.append(tup[1])
    
    result_df = result_df.loc[index_list]
    
    return result_df

'''#doesnt work perfectly
def performance(model, y_pred_scores, y_test, thresholds, k_values):
    
    performance_df = pd.DataFrame(columns=['model_type', 'threshold', 'accuracy', 'auc-roc', "f1_score" "p_at_1","p_at_10","p_at_50","p_at_500","p_at_'All'", "rec_at_1","rec_at_10","rec_at_50","rec_at_500","rec_at_'All'"])
    
    for threshold in thresholds:
        y_pred = [1 if y_pred_score >= threshold else 0 for y_pred_score in y_pred_scores]
        precision_at_k = []
        recall_at_k = []
        for k in k_values:
            precision, recall, threshold = precision_recall_curve(y_pred, y_test, k)
            precision_at_k.append(precision)
            recall_at_k.append(recall)
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_calc(y_test, y_pred)
        f1_score = f1_score_calc(y_test, y_pred)
        performance_df.loc[len(performance_df)] = [model, threshold, accuracy, auc_roc, f1_score] + precision_at_k + recall_at_k
    print(performance_df)        
    return performance_df
    

def precision_recall_score(y_pred, y_test, k = "All"):
    if k == "All":
        k = len(y_pred)
    from sklearn.metrics import precision_score,recall_score
    precision = precision_score(y_test[:k], y_pred[:k], average='binary')
    recall = recall_score(y_test[:k], y_pred[:k], average='binary')
    return precision,recall
    
def accuracy_score_calc(y_test, y_pred):   
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)

def f1_score_calc(y_test, y_pred):
    from sklearn.metrics import f1_score
    return f1_score( y_test, y_pred)

def confusion_matrix_calc(y_test, y_pred, k = "All"):
    if k == "All":
        k = len(y_pred)
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_test, y_pred)

def roc_auc_calc(y_test, y_pred):
    from sklearn.metrics import roc_auc_score
    base_roc_auc = roc_auc_score(y_test, y_pred)'''
