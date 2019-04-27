# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:33:54 2017

@author: Mike Staddon
"""

from sklearn.model_selection import (train_test_split, KFold,
                                     LeaveOneGroupOut, StratifiedKFold)

from sklearn.metrics import get_scorer

import numpy as np

from time import time
   

# Can't pass lambda functions to a process
def CrossValidationScore(estimator, X, y, sample_weight=None, scoring='', cv=5, cv_type=None, best_mean=None, best_std=None):
    # Use best_mean and best_std to implement early stopping
    
    if cv_type is None or cv_type == 'random cv':
        splitter = KFold(cv).split(X, y)
    elif cv_type == 'stratified':
        splitter = StratifiedKFold(cv).split(X, y)
    elif cv_type == 'group CV':
        splitter = LeaveOneGroupOut().split(X, y, X[cv])
        
    scores = []
    run_times = []
    scorer = get_scorer(scoring)

    for train_index, test_index in splitter:
        start = time()
        
        if sample_weight is None:
            w_train = w_test = None
        else:
            w_train, w_test = sample_weight[train_index], sample_weight[test_index]
        
        try:
            estimator.fit(X.iloc[train_index,:], y.iloc[train_index], sample_weight=w_train)
        except:
            estimator.fit(X.iloc[train_index,:], y.iloc[train_index])
            
        scores += [scorer(estimator, X.iloc[test_index,:], y.iloc[test_index], sample_weight=w_test)]
        
        run_times += [time()-start]
        
        # Reject badly performing models early
        if len(scores) > 1 and best_mean is not None and best_std is not None:
            if np.array(scores).mean() < best_mean - 1.6 * best_std / (len(scores) - 1)**0.5 :
                break
        
    scores = np.array(scores)
    run_times = np.array(run_times)
    
    return scores.mean(), scores.std(), run_times.mean()
    

# Can't pass lambda functions to a process
def TestScore(estimator, X, y, sample_weight=None, scoring='', test_size=0.1):

    scorer = get_scorer(scoring)
    
    #Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    
    if sample_weight is None:
        w_train = w_test = None
    else:
         w_train, w_test = train_test_split(sample_weight,
                                            test_size=test_size,
                                            random_state=42)

    start = time()

    
    try:
        estimator.fit(X_train, y_train, sample_weight=w_train)
    except:
        estimator.fit(X_train, y_train)
        
    score = scorer(estimator, X_test, y_test, sample_weight=w_test)
    
    run_time = time()-start
    
    # Return 0 standard deviation - needed for gaussian process
    return score, 0, run_time