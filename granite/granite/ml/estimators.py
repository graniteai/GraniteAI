# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:03:03 2017

@author: Mike Staddon

Estimators and model fitting
"""

import numpy as np
import pandas as pd

from time import time
from multiprocessing import Process, Pipe

from .optimisation import BayesianOptimise
from .preprocessing import GetPreprocessor, DummyTransformer
from .metrics import TestScore, CrossValidationScore

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer


# Benchmark models
from sklearn.dummy import DummyClassifier, DummyRegressor

# Regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet, LassoLars, SGDRegressor,
                                  PassiveAggressiveRegressor)

from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import RBFSampler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              AdaBoostRegressor, GradientBoostingRegressor)

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

# Classification
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                  PassiveAggressiveClassifier)

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)


# Process fitting hyper parameters, allows cancellation by user
class FitProcess(Process):
    """
    Uses gaussian processes to optimise the given loss function eg cross val
    scores, by testing the points with the highest expected improvement in 
    score per second
    """
    
    def __init__(self, conn, estimator, bounds, scorer, score_type, score_option, X, y, sample_weight=None, n_random_search=10, max_train_time=3600,
                 params=None,
                 raw_params=None,
                 loss_results=None,
                 loss_results_std=None,
                 run_times=None):
        
        """
        See BayesianOptEstimator for details - this just fits in a process so it can quit early
        """
        
        super(FitProcess, self).__init__()
        
        self.conn=conn
        
        self.estimator = estimator
        self.bounds = bounds
        
        self.scorer = scorer
        self.score_type=score_type
        self.score_option=score_option

        self.X = X
        self.y = y
        self.sample_weight=sample_weight
            
        self.n_random_search=n_random_search
        self.max_train_time=max_train_time

        # Results
        self.params=[] if params is None else params
        self.raw_params=[] if raw_params is None else raw_params
        self.loss_results=[] if loss_results is None else loss_results
        self.loss_results_std=[] if loss_results_std is None else loss_results_std
        self.run_times=[] if run_times is None else run_times
        self.total_time=[]
        self.prob_improvement=[]
        
        self.param_names = [bounds[0] for bounds in self.bounds]
        self.param_types = [bounds[1] for bounds in self.bounds]
        self.param_bounds = [bounds[2] for bounds in self.bounds]
        
        self.param_categories = {self.bounds[i][0]: self.bounds[i][2] for i in range(len(self.bounds)) if self.bounds[i][1] == 'categoric'}
        
        # Categoric bounds are indicies
        for i in range(len(bounds)):
            if bounds[i][1] == 'categoric':
                self.param_bounds[i] = [0, len(self.bounds[i][2])-1]
                
        self.param_bounds = np.array(self.param_bounds)
        
        # Var types for bayesian optimisation
        self.integers=[i for i in range(len(self.bounds)) if self.bounds[i][1] == 'integer']
        self.categorics=[i for i in range(len(self.bounds)) if self.bounds[i][1] == 'categoric']
        
        # Number of categories
        self.num_categories = [len(bound[2]) if bound[1] == 'categoric' else 0 for bound in self.bounds]
        if len(self.num_categories)==0:
            self.num_categories=None

        # Maximum combinations of parameters
        if 'float' in self.param_types or 'exponential' in self.param_types:
            self.max_combinations = None
        else:
            # Get maximum combos
            self.max_combinations = 1
            for par in self.bounds:
                if par[1] == 'integer':
                    # Any integer in the range
                    self.max_combinations *= (par[2][1] - par[2][0] + 1)
                else:
                    # Any category
                    self.max_combinations *= len(par[2])
                    
        
    def SampleRandomParameters(self):
        values = []
        for i in range(len(self.param_bounds)):
            lb, ub = self.param_bounds[i][0], self.param_bounds[i][1]
            
            if self.param_types[i] == 'integer':
                values += [np.random.randint(lb, ub+1)]
                
            elif self.param_types[i] == 'categoric':
                values += [np.random.randint(self.num_categories[i])]
                
            else:
                values += [np.random.uniform(lb, ub)]
                
        return values
    
    
    def EvaluateParameters(self):
        
        if len(self.loss_results) > 0:
            best_mean = max(self.loss_results)
            best_std = self.loss_results_std[self.loss_results.index(best_mean)]
        else:
            best_mean, best_std = None, None

        if self.score_type == 'test':
            return TestScore(self.estimator, self.X, self.y,
                             sample_weight=self.sample_weight,
                             scoring=self.scorer,
                             test_size=self.score_option)
            
        return CrossValidationScore(self.estimator, self.X, self.y,
                                    sample_weight=self.sample_weight,
                                    scoring=self.scorer,
                                    cv=self.score_option,
                                    cv_type=self.score_type,
                                    best_mean=best_mean, best_std=best_std)
    
    
    # Run the process
    def run(self):
        fit_start = time()
            
        # First search randomly
        while time() - fit_start < self.max_train_time:
            #Sample random parameters
            new_pars={}
            values = []
            
            # Sample randomly to begin with, or random every other time
            if len(self.loss_results) < self.n_random_search:
                values = self.SampleRandomParameters()
                      
            # Then use bayesian optimisation
            else:
                # + 10s for run time because optimisation takes time!
                values, pi = BayesianOptimise(np.array(self.raw_params),
                                              np.array(self.loss_results),
                                              self.param_bounds,
                                              cost=np.array(self.run_times)+10,
                                              integers=self.integers,
                                              categorics=self.categorics,
                                              num_categories=self.num_categories,
                                              return_pi=True,
                                              alphas=self.loss_results_std)

                self.prob_improvement += [pi]
                
            # Resample
            if values in self.raw_params:
                if self.max_combinations is None:
                    values = self.SampleRandomParameters()
                # Keep sampling until we get a new combo
                elif len(self.params) < self.max_combinations:
                    # Keep going until we get a random one
                    while values in self.raw_params:
                        values = self.SampleRandomParameters()
                else:
                    # Else we have explored the entire space
                    break
            
            for par, par_type, value in zip(self.param_names, self.param_types, values):
                if par == 'mod__hidden_layer_sizes':
                    new_pars[par] = (int(np.round(value)),)
                elif par_type == 'float':
                    new_pars[par] = value
                elif par_type == 'integer':
                    new_pars[par] = int(np.round(value))
                elif par_type == 'exponential':
                    new_pars[par] = np.exp(value)
                elif par_type == 'categoric':
                    # Select the item from the "bounds"
                    new_pars[par] = self.param_categories[par][int(np.round(value))]
            
            self.params.append(new_pars)
            self.raw_params.append(values)
            self.estimator.set_params(**new_pars)
            
            mean, std, run_time = self.EvaluateParameters()

            self.loss_results.append(mean)
            self.loss_results_std.append(std)
            
            self.run_times.append(run_time)
            self.total_time.append(time()-fit_start)
            
            # Send results
            self.conn.send(['results', [new_pars, values, mean, std, run_time]])
            
            # Send best esimator to main model
            if self.loss_results[-1] == max(self.loss_results):
                self.conn.send(['estimator', self.estimator])
                
                # Fit on rest of data
                try:
                    self.estimator.fit(self.X, self.y, sample_weight=self.sample_weight)
                except:
                    self.estimator.fit(self.X, self.y)
                    
                self.conn.send(['estimator', self.estimator])
                
                
            if len(self.param_names) == 0:
                break
                
        self.best_score_ = max(self.loss_results)
        best_index = self.loss_results.index(self.best_score_)
        self.best_params_ = self.params[best_index]


class BayesianOptEstimator():
    """
    Uses gaussian processes to optimise the given loss function eg cross val
    scores, by testing the points with the highest expected improvement in 
    score per second
    """
    
    def __init__(self, estimator, params, scoring, score_type, score_option, n_random_search=5, max_train_time=600):
        """
        Estimator is our base model
        
        Params should be in format ['param', 'type', ['lower bound', 'upper bound']]
            For categories ['param', 'categoric', ['cat1', 'cat2', ...]]
            
        scoring: the metric we will use to score models
            
        score_type: in ['random cv', 'group cv', 'test']
        
        score_option: should be number of folds for random, group for group,
                        and test_size for test
        
        n_random_search: number of random searches to do before using GP
        
        max_train_time: return best results after this time
        """
        self.estimator = estimator
        self.bounds = params
        
        self.scoring = scoring
        self.score_type=score_type
        self.score_option=score_option
        
        self.n_random_search=n_random_search
        self.max_train_time=max_train_time
        
        # Results
        self.params=[]
        self.raw_params=[]
        self.loss_results=[]
        self.loss_results_std=[] # Standard deviation over CV Folds
        self.run_times=[]
        self.total_time=[]
        
        self.trained = False
        self.fitting = False
        self.queued = False
        self.fit_start = 0
    
    
    def fit(self, X, y, sample_weight=None, max_train_time=None):
        """
        X: pandas data frame
        y: target data
        sample_weight: array of weights or None
        max_train_time: use this to keep fitting!
        """
        
        # Should we stop early?
        self.cancel_flag = False
        self.fitting = True
        self.fit_start = time()
        
        # Use this to continue fitting
        if max_train_time is not None:
            self.max_train_time=max_train_time

        # Create and run process
        parent_conn, child_conn = Pipe()
        
        fit_process = FitProcess(child_conn, self.estimator, self.bounds,
                                 self.scoring, self.score_type, self.score_option,
                                 X, y, sample_weight=sample_weight,
                                 n_random_search=self.n_random_search,
                                 max_train_time=self.max_train_time,
                                 params=self.params,
                                 raw_params=self.raw_params,
                                 loss_results=self.loss_results,
                                 loss_results_std=self.loss_results_std,
                                 run_times=self.run_times)
        
        fit_process.start()
        
        if len(self.total_time) == 0:
            previous_time = 0
        else:
            previous_time = self.total_time[-1]
            
        while time() - self.fit_start < self.max_train_time and self.cancel_flag is False and fit_process.is_alive():
            
            # Recieve results in form of ['type', results]
            if parent_conn.poll():
                msg = parent_conn.recv()
                
                if msg[0] == 'results':
                    new_pars, values, mean, std, run_time = msg[1]
                    
                    self.params.append(new_pars)
                    self.raw_params.append(values)
                    self.loss_results.append(mean)
                    self.loss_results_std.append(std)
                    self.run_times.append(run_time)
                    self.total_time.append(time()-self.fit_start + previous_time)
                    
                elif msg[0] == 'estimator':
                    self.best_estimator_temp_ = msg[1]
                    self.best_score_ = max(self.loss_results)
                    self.best_params_ = self.params[-1]
                                    
        else:
            if fit_process.is_alive():
                fit_process.terminate()
                
            if len(self.loss_results) > 0:
                self.trained = True
                self.best_estimator_ = self.best_estimator_temp_
                
                if hasattr(self.best_estimator_, 'classes_'):
                    self.classes_ = self.best_estimator_.classes_
            
        self.fitting = False
    
    
    # Stop training early
    def cancel_training(self):
        self.cancel_flag = True
           
            
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
    
    
    def score(self, X, y, sample_weight=None):
        return get_scorer(self.scoring)(self.best_estimator_, X, y, sample_weight=sample_weight)


#Dummy models - used for benchmarking
def GetRegressionDummy(scoring='neg_mean_log_loss', split_option=None, split_type=None, test_size=0.1, max_train_time=3600):
    
    #Fast is just a flag if we only want the "fast" models     
    models = {}
    
    mod = Pipeline([('pre', DummyTransformer()),
                    ('mod', DummyRegressor())])
            
    params=[]
            
    models['Dummy Benchmark'] = BayesianOptEstimator(mod, params, scoring, split_type, split_option)
    
    return models


def GetClassDummy(scoring='neg_mean_squared_error', split_option=None, split_type=None, max_train_time=3600):
    

    #Fast is just a flag if we only want the "fast" models     
    models = {}

    mod = Pipeline([('pre', DummyTransformer()),
                    ('mod', DummyClassifier(strategy='prior'))])
            
    params=[]
    
    if split_type == 'Random CV':
        split_type = 'Stratified'
            
    models['Dummy Benchmark'] = BayesianOptEstimator(mod, params, scoring, split_type, split_option)
    
    return models
   
    
def GetRegModels(nums=None, cats=None, text=None, scoring='neg_mean_squared_error', split_option=None, split_type=None, max_train_time=3600):
    
    if nums is None and cats is None and text is None:
        return
    
    if nums is None:
        nums = []
    if cats is None:
        cats = []
    if text is None:
        text = []
        
    models = []
    
    """ Numeric preprocessing options """
    if len(nums) > 0:
        params_impute = [['pre__nums__imputer', 'categoric', ['mean', 'median']]]
        params_scale = [['pre__nums__scaler', 'categoric', ['MinMax', 'MaxAbs', 'Standard', 'Robust']]]
    else:
        params_impute = []
        params_scale = []
    
    
    """ Linear models """
    
    # Linear regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', LinearRegression())])
    
    params = []
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Linear Regression'
    models += [mod]
    
    
    # Lasso regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', Lasso(copy_X=False))])
    
    params = [['mod__alpha', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Lasso Regression'
    models += [mod]
    
    
    # Ridge regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', Ridge(copy_X=False))])
    
    params = [['mod__alpha', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Ridge Regression'
    models += [mod]
    
            
    # ElasticNet regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', ElasticNet(copy_X=False))])
    
    params = [['mod__l1_ratio', 'float', [1e-3, 1-1e-3]],
              ['mod__alpha', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'ElasticNet Regression'
    models += [mod]
    
    
#    # SGD Regression
#    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
#                    ('mod', SGDRegressor())])
#    
#    params = [['mod__alpha', 'exponential', [-5, 5]],
#              ['mod__penalty', 'categoric', ['l2', 'l2', 'elasticnet']],
#              ['mod__l1_ratio', 'float', [1e-5, 1-1e-5]]]
#    
#    params += params_impute + params_scale
#    
#    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
#    mod.name = 'SGD Regression'
#    models += [mod]
    
    
#    # Lasso LARS
#    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
#                    ('mod', LassoLars(max_iter=500))])
#    
#    params = [['mod__alpha', 'exponential', [-5, 5]]]
#    params += params_impute + params_scale
#    
#    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
#    mod.name = 'LARS Regression'
#    models += [mod]
    

    # Passive Aggressive Regressor
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', PassiveAggressiveRegressor())])
    
    params = [['mod__C', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Passive Aggressive Regression'
    models += [mod]
    
    
    """ Support Vector Machines """
    
    # Linear SVM
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', LinearSVR(dual=False, loss='squared_epsilon_insensitive'))])
    
    params = [['mod__C', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Linear SVM'
    models += [mod]
    
    
    # Kernel SVM
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('krn', RBFSampler()),
                    ('mod', LinearSVR(dual=False, loss='squared_epsilon_insensitive'))])
    
    params = [['krn__gamma', 'exponential', [-10, 10]],
              ['krn__n_components', 'integer', [10, 200]],
              ['mod__C', 'exponential', [-5, 5]]]
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Kernel SVM'
    models += [mod]
    
    
    """ Tree based methods """
    # Decision Tree
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', DecisionTreeRegressor())])
    
    params = [['mod__criterion', 'categoric', ['mse', 'friedman_mse', 'mae']],
              ['mod__max_depth', 'integer', [1, 15]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Decision Tree'
    models += [mod]
    
    
    # Random Forest
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', RandomForestRegressor())])
    
    params = [['mod__criterion', 'categoric', ['mse', 'friedman_mse', 'mae']],
              ['mod__max_depth', 'integer', [1, 15]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Random Forest'
    models += [mod]
    
    
    # Extremely Random Forest
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', ExtraTreesRegressor())])
    
    params = [['mod__criterion', 'categoric', ['mse', 'friedman_mse', 'mae']],
              ['mod__max_depth', 'integer', [1, 15]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Extra Trees'
    models += [mod]
    
    # Boosting needs dense data
    if len(text) == 0:
        # Gradient Boosted Trees
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', GradientBoostingRegressor())])
        
        params = [['mod__criterion', 'categoric', ['mse', 'friedman_mse', 'mae']],
                  ['mod__max_depth', 'integer', [1, 5]],
                  ['mod__min_samples_split', 'integer', [2, 20]],
                  ['mod__min_samples_leaf', 'integer', [1, 20]],
                  ['mod__learning_rate', 'exponential', [-5, 0]],
                  ['mod__n_estimators', 'integer', [10, 50]]]
        
        params += params_impute
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Gradient Boosted Trees'
        models += [mod]
        
        
        # AdaBoost
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', AdaBoostRegressor())])
        
        params = [['mod__loss', 'categoric', ['linear', 'square', 'exponential']],
                  ['mod__learning_rate', 'exponential', [-5, 5]],
                  ['mod__n_estimators', 'integer', [10, 50]]]
        
        params += params_impute
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'AdaBoost'
        models += [mod]
    
    
    # XGBoost!
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', XGBRegressor())])
    
    params = [['mod__max_depth', 'integer', [1, 5]],
              ['mod__learning_rate', 'exponential', [-5, 0]],
              ['mod__n_estimators', 'integer', [10, 50]]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'XGBoost'
    models += [mod]
     
    
    """ KNN """
    
    # KNN
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', KNeighborsRegressor())])
    
    params = [['mod__n_neighbors', 'integer', [1, 20]],
              ['mod__weights', 'categoric', ['uniform', 'distance']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'K Nearest Neighbors'
    models += [mod]
    
    
    """ Neural Network"""
    
    # NN
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', MLPRegressor(learning_rate_init=0.01))])
    
    params = [['mod__alpha', 'exponential', [-10, 10]],
              ['mod__hidden_layer_sizes', 'integer', [5, 50]]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Neural Network'
    models += [mod]
    
    
    return {mod.name: mod for mod in models}
    
    

def GetClassModels(nums=None, cats=None, text=None, scoring='accuracy', split_option=None, split_type=None, max_train_time=3600):     

    if nums is None and cats is None and text is None:
        return
    
    if nums is None:
        nums = []
    if cats is None:
        cats = []
    if text is None:
        text = []
        
    models = []
    
    if split_type == 'Random CV':
        split_type = 'Stratified'
        
    
    """ Numeric preprocessing options """
    if len(nums) > 0:
        params_impute = [['pre__nums__imputer', 'categoric', ['mean', 'median']]]
        params_scale = [['pre__nums__scaler', 'categoric', ['MinMax', 'MaxAbs', 'Standard', 'Robust']]]
    else:
        params_impute = []
        params_scale = []
    
    
    """ Linear models """
    
    # Logistic regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', LogisticRegression(C=1e10))])
    
    params = [['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Logistic Regression'
    models += [mod]
    
    
    # Lasso regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', LogisticRegression(penalty='l1'))])
    
    params = [['mod__C', 'exponential', [-10, 10]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Lasso Logistic Regression'
    models += [mod]
    
    
    # Ridge regression
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', LogisticRegression(penalty='l2'))])
    
    params = [['mod__C', 'exponential', [-10, 10]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Ridge Logistic Regression'
    models += [mod]
    
    
#    # Elastic net - must be done through SGD
#    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
#                    ('mod', SGDClassifier(penalty='elasticnet', loss='log'))])
#    
#    params = [['mod__alpha', 'exponential', [-10, 10]],
#              ['mod__l1_ratio', 'float', [1e-5, 1-1e-5]],
#              ['mod__class_weight', 'categoric', [None, 'balanced']]]
#    
#    params += params_impute + params_scale
#    
#    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
#    mod.name = 'ElasticNet Logistic Regression'
#    models += [mod]
    

#    # Passive Aggressive Regressor
#    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
#                    ('mod', PassiveAggressiveClassifier())])
#    
#    params = [['mod__C', 'exponential', [-10, 10]],
#              ['mod__class_weight', 'categoric', [None, 'balanced']]]
#    
#    params += params_impute + params_scale
#    
#    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
#    mod.name = 'Passive Aggressive Classifier'
#    models += [mod]
    
    
    """ Support Vector Machines """
    

    # Linear SVM
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', SVC(probability=True, kernel='linear'))])
    
    params = [['mod__C', 'exponential', [-10, 10]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Linear SVM'
    models += [mod]
    
    
    # Kernel SVM
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('krn', RBFSampler()),
                    ('mod', SVC(probability=True, kernel='linear'))])
    
    params = [['krn__gamma', 'exponential', [-10, 10]],
              ['krn__n_components', 'integer', [5, 200]],
              ['mod__C', 'exponential', [-10, 10]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Kernel SVM'
    models += [mod]
    
    
    """ Tree based methods """
    # Decision Tree
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', DecisionTreeClassifier())])
    
    params = [['mod__criterion', 'categoric', ['gini', 'entropy']],
              ['mod__max_depth', 'integer', [1, 20]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Decision Tree'
    models += [mod]
    
    
    # Random Forest
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', RandomForestClassifier())])
    
    params = [['mod__criterion', 'categoric', ['gini', 'entropy']],
              ['mod__max_depth', 'integer', [1, 20]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Random Forest'
    models += [mod]
    
    
    # Extremely Random Forest
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', ExtraTreesClassifier())])
    
    params = [['mod__criterion', 'categoric', ['gini', 'entropy']],
              ['mod__max_depth', 'integer', [1, 20]],
              ['mod__min_samples_split', 'integer', [2, 20]],
              ['mod__min_samples_leaf', 'integer', [1, 20]],
              ['mod__class_weight', 'categoric', [None, 'balanced']]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Extra Trees Classifier'
    models += [mod]
    
    # Boosting needs dense data
    if len(text) == 0:
        # Gradient Boosted Trees
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', GradientBoostingClassifier())])
        
        params = [['mod__n_estimators', 'integer', [10, 50]],
                  ['mod__max_depth', 'integer', [1, 5]],
                  ['mod__min_samples_split', 'integer', [2, 20]],
                  ['mod__min_samples_leaf', 'integer', [1, 20]],
                  ['mod__learning_rate', 'exponential', [-5, 0]]]
        
        params += params_impute
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Gradient Boosted Trees'
        models += [mod]
        
        
        # AdaBoost
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', AdaBoostClassifier())])
        
        params = [['mod__n_estimators', 'integer', [10, 50]],
                  ['mod__learning_rate', 'exponential', [-5, 5]]]
        
        params += params_impute
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'AdaBoost'
        models += [mod]
    
    
    # XGBoost!
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', XGBClassifier())])
    
    params = [['mod__n_estimators', 'integer', [10, 50]],
              ['mod__max_depth', 'integer', [1, 5]],
              ['mod__learning_rate', 'exponential', [-5, 0]],
              ['mod__reg_alpha', 'exponential', [-10, 10]],
              ['mod__reg_lambda', 'exponential', [-10, 10]]]
    
    params += params_impute
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'XGBoost'
    models += [mod]
     
    
    """ KNN """
    
    # KNN
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', KNeighborsClassifier())])
    
    params = [['mod__n_neighbors', 'integer', [1, 20]],
              ['mod__weights', 'categoric', ['uniform', 'distance']]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'K Nearest Neighbors'
    models += [mod]
    
    
    """ Neural Network"""
    
    # NN
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', MLPClassifier(learning_rate_init=0.01))])
    
    params = [['mod__alpha', 'exponential', [-10, 10]],
              ['mod__hidden_layer_sizes', 'integer', [5, 50]]]
    
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Neural Network'
    models += [mod]
    
    
    """ Naive Bayes """
    
    # Bernoulli
    mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                    ('mod', BernoulliNB())])
    
    params = []
    params += params_impute + params_scale
    
    mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
    mod.name = 'Bernoulli Naive Bayes'
    models += [mod]
    
    
    # Multinomial - only for nums == 0
    if len(nums) == 0:
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', MultinomialNB())])
        
        params = [['pre__nums__imputer', 'categoric', ['mean', 'median', 'most_frequent']]]
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Multinomial Naive Bayes'
        models += [mod]
    
    
    # Gaussian
    if len(text) == 0:
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', GaussianNB())])
        
        params = []
        params += params_impute + params_scale
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Gaussian Naive Bayes'
        models += [mod]
    
    
    """ Discriminant analysis """
    
    # LDA
    if len(text) == 0:
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', LinearDiscriminantAnalysis())])
        
        params = []
        params += params_impute + params_scale
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Linear Discriminant Analysis'
        models += [mod]
        
        
        # QDA
        mod = Pipeline([('pre', GetPreprocessor(nums, cats, text)),
                        ('mod', QuadraticDiscriminantAnalysis())])
        
        params = [['pre__nums__imputer', 'categoric', ['mean', 'median', 'most_frequent']],
                  ['pre__nums__scaler', 'categoric', ['MinMax', 'MaxAbs', 'Standard', 'Robust']]]
        
        mod = BayesianOptEstimator(mod, params, scoring, split_type, split_option,max_train_time=max_train_time)
        mod.name = 'Quadratic Discriminant Analysis'
        models += [mod]
    
    return {mod.name: mod for mod in models}


