# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:33:54 2017

@author: Mike Staddon
"""

import pandas as pd
import numpy as np
from scipy.stats import binned_statistic

from sklearn.feature_extraction.text import CountVectorizer


def grouped_average(df, feature, dtype, bins=None):
    """ Get values averaged over some feature which can be numeric, categoric,
    or text data
    
    Arguments:
        df: pandas dataframe
            data
        feature: string
            column to group values by
        dtype: string, ['numeric', 'categoric', 'text']
             the feature type in this model. Numeric features will have values
            binned and averaged over bins. Categoric features will be average
            by category. Text features will average over words
        bins: int, optional
            maximum number of groups to make
    
    Returns:
        bins: array
            the binned values
        counts: array
            counts in each bin
        stats: pandas dataframe
            each column averaged by bins
    """
    
    # Ignore nan values
    df = df.dropna(subset=[feature])
    
    stats = pd.DataFrame()
    
    if bins is None:
        if dtype == 'numeric':
            bins = 10
        else:
            bins = 25
    
    if dtype == 'numeric':
        # Bin values, get statistics in bins
        counts, bins, _ = binned_statistic(df[feature], df[feature], statistic='count', bins=bins)
        
        for col in df.columns:
            if col == feature:
                continue
            
            # Average value in bins
            stats[col], _, _ = binned_statistic(df[feature], df[col], statistic='mean', bins=bins)
            
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
            
    elif dtype == 'categoric':
        # Average values by category
        group = df.groupby(by=feature)
        
        stats = group.mean()
        counts = group.count().values[:, 0]
        bins = group.count().index.values
    else:
        # Average value by word
        
        # Count the word frequency by row
        counter = CountVectorizer(max_features=100, stop_words='english')
        counted = counter.fit_transform(df[feature])

        # Vocabulary maps words to columns, unmap to get corresponding words
        kvs = counter.vocabulary_
        bins = sorted(list(kvs.keys()), key=lambda k: kvs[k])
        counts = counted.sum(axis=0).transpose().A.flatten()
        
        # Average for each column
        for col in df.columns:
            if col == feature:
                continue
            
            # For each word, take average value of column weighted by word frequency
            stats[col] = counted.multiply(df[col].values.reshape(-1,1)).sum(axis=0).A.flatten() / counts
    
    return stats, counts, bins
    

def feature_dependence(X, y, feature, dtype, bins=None, model=None, cat=None):
    """ Get statistics for how y changes with feature
    
    Arguments:
        X: pandas dataframe
            the data set
        y: pandas series
            target variable
        feature: string
            the feature to get statistics over
        dtype: string, ['numeric', 'categoric', 'text']
            the feature type in this model. Numeric features will have values
            binned and averaged over bins. Categoric features will be average
            by category. Text features will average over words
        model: sklearn estimator, optional
            any model with the predict method. If given, model predictions will
            also be averaged
        cat: string, int, or float, optional
            category for classification problems. If given then we get
            proportions
    
    Returns:
        bins: array
            value bins, or categories, with values averaged in
        y: array
            averaged values
        ypred: array, if model is not None
            averaged model predicted values
    """
    
    
    df = pd.DataFrame()
    df['x'] = X[feature]
    
    if cat is None:
        df['y'] = y
    else:
        df['y'] = (y == cat) * 1.0

    if model is not None:
        if cat is None:
            df['ypred'] = model.predict(X)
        else:
            # Get probability in one class
            index = list(model.classes_).index(cat)
            df['ypred'] = model.predict_proba(X)[:, index]
            
    return grouped_average(df, 'x', dtype, bins=bins)
        

if __name__ == '__main__':
    
    data = pd.read_csv('D:/Kaggle/Titanic/train.csv')
    
    print(*feature_dependence(data, data['Survived'], 'Name', 'text'))
    text = ['dog',
            'cat',
            'cat dog',
            'cat cat cat']
    
    values = [100,
              0,
              1,
              10]
    
    data = pd.DataFrame()
    data['text'] = text
    data['y'] = values
    
    print('Should be:')
    print('cat', 5, (1 * 1 + 3 * 10) / 5)
    print('dog', 2, (1 * 1 + 100 * 1) / 2)
    
    print('Calculated:')
    print(*feature_dependence(data, data['y'], 'text', dtype='text'))