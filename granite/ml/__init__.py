# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:34:56 2018

@author: Mike Staddon
"""

import pandas as pd
import numpy as np

import plotting, ml

from ml.estimators import GetRegModels, GetClassModels

from sklearn.model_selection import train_test_split

class Experiment():
    def __init__(self, data, target, classification, weights=None,
                 validation_method=None, validation_size=None,
                 holdout_size=None, scorer=None,
                 nums=None, cats=None, text=None,
                 name=None):
        """ Class containing models and training for one experiment
        
        Parameters:
            data: pandas dataframe
                the data to train models on
            target: string
                the target column to predict
            classification: bool
                if false, regression
            weights: string, optional
                weights column for scoring, equal weights if none
            validation_method: string
                either 'cv', or 'test', or full sentences for the UI
            validation_size: int or float
                cv folds for cv, test size for train-test split
            holdout_size: float
                size of holdout data to test models on
            scorer: string
                metric to optimise on
            nums: list
                list of numeric columns to use
            cats: list
                list of categoric columns to use
            text: list
                list of text columns to use
            name: str
                id of the experiment, should be unique
        """
        
        self.data = data
        self.target = target
        self.y = data[target]

        self.name = name if name else 'Untitled Experiment'
        
        if validation_method in [None, 'cv', 'Cross Validation']:
            self.validation_method = 'random cv'
        elif validation_method in ['test', 'Train-Test Split']:
            self.validation_method = 'test'
        else:
            raise ValueError
            
        if validation_size is None:
            if self.validation_method == 'random cv':
                self.validation_size = 5
            else:
                self.validation_size = 0.1
        else:
            self.validation_size = validation_size
            
            
        if weights is None:
            self.weights = None
        else:
            self.weights = self.data[weights]
            
        if holdout_size is None:
            self.holdout_size = 0.1
        else:
            self.holdout_size = holdout_size
            
        self.classification = classification
        
        self.scorer = scorer
        self.nums = nums
        self.cats = cats
        self.text = text
        
        if self.classification:
            func = GetClassModels
            self.models = GetClassModels
        else:
            func = GetRegModels
            
        self.models = func(nums, cats, text,
                           split_type=self.validation_method,
                           split_option=self.validation_size,
                           scoring=self.scorer)
        
        
    def split_data(self, train=True, test=True, weights=False):
        """ Return the train, or test, split of data or weights"""
        
        out = ()
        
        if train or test:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.data[self.target],
                                                                test_size=self.holdout_size,
                                                                random_state=42)
            
            if train:
                out += (X_train, y_train)
            if test:
                out += (X_test, y_test)
        
        if weights:
            if self.weights is None:
                if train:
                    out += (None,)
                if test:
                    out += (None,)
                
        return out

    
    def fit_model(self, model, max_train_time=600):
        """ Optimise chosen model for max_train_time
        
        Arguments:
            model: string
                model to train
            max_train_time: float, optional
                maximum train time in seconds
        """
        
        #Train and test split
        (X_train, y_train,
         X_test, y_test,
         w_train, w_test) = self.split_data(train=True, test=True, weights=True)


        # Fit on train data
        self.models[model].fit(X_train, y_train, sample_weight=w_train, max_train_time=max_train_time)
        
        self.models[model].holdout_score = None
        
        if self.models[model].trained:
            test_score = self.models[model].score(X_test, y_test, sample_weight=w_test)
        else:
            test_score = None
            
        self.models[model].holdout_score = test_score

        
    def cancel_fitting(self, models=None):
        """ Cancel training of models
        
        Parameters:
            models: string or list of strings
                optional, model or models to cancel
        """
            
        if models is None:
            models = self.models.keys()
            
        if type(models) == str:
            models = [models]

        for model in models:
            self.models[model].cancel_training()
        
        
    def predict(self, X, model):
        return self.models[model].predict(X)
        
        
    def predict_prob(self, X, model):
        return self.models[model].predict_prob(X)


    def get_models_summary(self):
        """ Return models, validation score, and holdout score """
        pass
    
    
    def get_predictions(self, model, data=None, append=False):
        """ Return model predictions
        
        Arguements:
            model: str
                model used in prediction
            data: None, str, or dataframe
                if None use the training data. If a string is provided, try
                loading the data. If a dataframe is provided use that
            append: bool, optional
                whether to return the predictions appended to the data
                
        Returns:
            predictions: dataframe
                data frame of predictions, with or without original data
        """
        
        if data is None:
            data = self.data
        elif type(data) == str:
            # What if there's an error??
            data = pd.read_csv(data)
        
        preds = pd.DataFrame(data=self.models[model].predict(data),
                             columns=['Prediction'])
        
        if self.classification:
            probs = pd.DataFrame(data=self.models[model].predict_proba(data),
                                 columns=['Probability '+str(c) for c in self.models[model].classes_])
            
            preds = pd.merge(preds, probs, left_index=True, right_index=True)
            
            
        if append:
            preds = pd.merge(preds, data, left_index=True, right_index=True)
            
        return preds
            
    
    # Get prediction for X, see features that give this result
    def get_prediction_reasons(self, model, X):
        """
        Gets predictions for one row and the features driving it. 
        Reasons are calculated as the difference in predictions if that feature
        was set to the deafult value ie nan
        
        Parameters:
            model: str
                model to predict with
            X: pandas Dataframe
                a 1-row frame to predict on

        returns:
            prediction: predicted value or class
            
            diffs: pandas Dataframe
                difference in predictions due to each feature
            probs: np array
                probabilities for each class, only if classification
        """
        estimator = self.models[model]
        
        new_X = X.copy()
        
        for f in self.nums + self.cats:
            row = X.copy()
            row[f] = [np.nan]
            new_X = new_X.append(row, ignore_index=True)
            
        for t in self.text:
            row = X.copy()
            row[t] = ['']
            new_X = new_X.append(row, ignore_index=True)
            
        # Get predictions
        pred = estimator.predict(new_X)
        prediction = pred[0]
        
        if self.classification:
            index = list(estimator.classes_).index(prediction)
            probs = estimator.predict_proba(new_X)
            pred = probs[:, index]
            probs = probs[0, :]
        else:
            probs = None
    
        # Show differences
        diffs = pd.DataFrame(columns=['Feature', 'Difference'])
        
        for i, f in enumerate(self.nums + self.cats + self.text):
            row = pd.DataFrame()
            row['Feature'] = [f]
            row['Difference'] = [pred[0]-pred[i+1]]
            diffs = diffs.append(row, ignore_index=True)
            
        return prediction, diffs, probs

        
    def plot_training_scores(self, model, fig=None, ax=None):
        """ Plot validation scores, and best score for chosen model """
        plotting.TrainingPlot(self.models[model].total_time,
                              self.models[model].loss_results,
                              scorer=self.scorer,
                              cv=(self.validation_method=='cv'),
                              fig=fig, ax=ax)

        
    def plot_confusion_matrix(self, model, train=True, fig=None, ax=None):
        """ Plot the confusion matrix for model, one train or test set
        """
        X, y = self.split_data(train=train, test=not train)
        ypred = self.models[model].predict(X)
        plotting.ConfusionMatrix(y, ypred, fig=fig, ax=ax)
        
        
    def plot_roc_curve(self, model, train=True, fig=None, ax=None):
        """ Plot the ROC curves
        """
        X, y = self.split_data(train=train, test=not train)
        probs = self.models[model].predict_proba(X)
        plotting.ROCCurve(y, probs, fig=fig, ax=ax)
        
        
    def plot_feature_dependence(self, feature, model=None, cat=None, train=True, bins=None, ylims=None, fig=None, ax=None):
        """ Plot how mean actual and predicted values change with given
        features
        """
        
        if feature in self.nums:
            dtype = 'numeric'
        elif feature in self.cats:
            dtype = 'categoric'
        elif feature in self.text:
            dtype = 'text'
        else:
            return
        
        if model is not None:
            model = self.models[model]
        
        X, y = self.split_data(train=train, test=not train)

        plotting.FeatureDependence(X, y, feature, dtype, model=model, cat=cat, bins=bins, ylims=ylims, fig=fig, ax=ax)
        
        
    def get_feature_dependence(self, feature, model=None, cat=None, train=True):
        """ Get a table of how mean actual and predicted values change with
        given features
        """
        
        if feature in self.nums:
            dtype = 'numeric'
        elif feature in self.cats:
            dtype = 'categoric'
        elif feature in self.text:
            dtype = 'text'
        else:
            print(feature + ' not found!')
            return
        
        if model is not None:
            model = self.models[model]
        
        X, y = self.split_data(train=train, test=not train)
        
        return ml.stats.feature_dependence(X, y, feature, dtype, model=model, cat=cat)
    
    
    def get_feature_importance(self, model):
        mod = self.models[model].best_estimator_
        
        try:
            fi = np.array(mod.steps[-1][1].feature_importances_)
            fi /= fi.max()
        except:
            return None
        
        # Get list of columns, split into category and word
        columns = []
        
        i = 0
        if len(self.nums) > 0:
            columns += self.nums
            i += 1

        if len(self.cats) > 0:
            transformer = mod.steps[0][1].transformer_list[i][1]
            columns += list(transformer.transformed_columns_)
            i += 1
            
        if len(self.text) > 0:
            transformer = mod.steps[0][1].transformer_list[i][1]
            columns += list(transformer.transformed_columns_)
                
        #Our raw feature importances. Need to sum up over variables next
        feature_importance = pd.DataFrame()
        feature_importance['Feature'] = columns
        feature_importance['Importance'] = fi
        
        return feature_importance


class GraniteProject():
    def __init__(self, data, name=None):
        self.data = data
        self.name = name if name else 'Untitled Project'
        self.experiments = []
        
        
    def plot(self, style, fig=None, **plotOptions):
        if style == 'Pair Plot':
            plotting.PairPlot(self.data, fig=fig, **plotOptions)
        elif style == 'Factor Plot':
            plotting.FactorPlot(self.data, fig=fig, **plotOptions)


    def get_all_columns(self):
        """Returns all column names as a list"""
        return list(self.data.columns)
    
    
    def get_numeric_columns(self):
        """Returns all numeric columns in a list"""
        return list(self.data.select_dtypes(include=[np.number]).columns)
    
    
    def get_non_negative_columns(self):
        """Returns all numeric columns that are non negative"""
        nums = self.get_numeric_columns()
        
        non = []
        for num in nums:
            if np.any(self.data[num] < 0):
                continue
            
            if np.any(self.data[num].isnull().values.any()):
                continue
            
            non.append(num)
            
        return non
    
    
    def get_metrics(self, target, dtype, ui=False):
        """Returns possible metrics for a given feature"""
        
        if dtype == 'Numeric':
            metrics = ['R Squared (Recommended)',
                       'Mean Squared Error',
                       'Mean Absolute Error',
                       'Median Absolute Error']
            
            # RMSLE if target is non negative
            if not np.any(self.data[target] < 0):
                metrics += ['Mean Squared Log Error']
                
        else:
            metrics = ['Log Loss (Recommended)',
                       'Accuracy']
            
            # Binary classification scorers
            if len(self.data[target].unique())==2:
                metrics += ['ROC Area Under Curve']
                
        return metrics
    
    
    def get_column_summary(self):
        """ Make a dataframe of feature, type, uniques, missing"""
        df = pd.DataFrame()
        
        df['#'] = [i+1 for i in range(len(self.data.columns))]
        df['Column'] = list(self.data.columns)
        
        dtypes = []
        for col in self.data.columns:
            if col in self.get_numeric_columns():
                dtypes.append('Numeric')
            else:
                dtypes.append('Categoric')
                
        df['Type'] = dtypes
        
        df['Unique'] = [len(self.data[col].unique()) for col in self.data.columns]
        df['Missing'] = [self.data[col].isnull().sum()/len(self.data) for col in self.data.columns]
        
        return df
    
    
    def get_frequencies(self, feature, dtype):
        """ Returns either the category or word count 
        
        Arguments:
            feature: string
                column to get statistics on
            dtype: string
                'categoric' or 'text'. If 'categoric', count each occurence.
                If 'text'
        """
        
        if self.data[feature].nunique() > 10:
            return None
        
        counts = self.data[feature].value_counts()
        
        # Set the index to be the column
        stats = pd.DataFrame()
        stats['Category'] = counts.index
        stats['Count'] = counts.values
        
        return stats
    
    
    def check_experiment_name(self, name):
        for exp in self.experiments:
            if name == exp.name:
                return False
        
        return True


    def make_new_experiment(self, options):        
        self.experiments.append(Experiment(self.data, **options))