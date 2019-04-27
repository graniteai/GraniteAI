# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:25:05 2017

@author: Mike Staddon

Data preprocessing steps
"""

import pandas as pd
import numpy as np

#Data processing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# Returns the data as is - used when we don't want some other type
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, *args, **kwargs):
        return self
    
    def transform(self, X, y=None, *args, **kwargs):
        return X
    
#Selects columns of the data
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.key]
    

#Returns None - used for training dummy estimators
class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #Return a single column of zeros as the dummy needs no X
        return np.zeros(shape=(X.shape[0], 1))
    

# Scale numeric values and impute missing values
class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns, scaler=None, imputer='mean'):
        self.numeric_columns=numeric_columns
        self.scaler=scaler
        self.imputer=imputer

    # Scale then impute data
    def fit(self, X, y=None, *args, **kwargs):
        steps = [('imputer', Imputer(strategy=self.imputer))]
        
        if self.scaler == 'MinMax':
            scl = MinMaxScaler()
        elif self.scaler == 'MaxAbs':
            scl = MaxAbsScaler()
        elif self.scaler == 'Standard':
            scl = StandardScaler()
        elif self.scaler == 'Robust':
            scl = RobustScaler()
            
        if self.scaler is not None:
            steps += [('scaler', scl)]
        
        self.pipe = Pipeline(steps)
        
        self.pipe.fit(X.reindex(columns=self.numeric_columns))
            
        return self
    
    
    def transform(self, X, y=None, *args, **kwargs):
        return self.pipe.transform(X.reindex(columns=self.numeric_columns))
    
    



# One hot encode categorical data
class CategoricTransformer(BaseEstimator, TransformerMixin):
    #Set up the categorical columns
    def __init__(self, cat_columns):
        self.cat_columns=cat_columns

    # Get the column names of the dummy data
    # More efficient than just making dummies and reading cols
    def fit(self, X, y=None, *args, **kwargs):
        
        X2 = X.reindex(columns=self.cat_columns, copy=True)
        
        #Make the cat columns have the cat datatype;
        for col in self.cat_columns:
            X2[col] = X2[col].apply(str).astype('category')
            
        self.cat_map_ = {col: X2[col].cat.categories for col in self.cat_columns}

        self.dummy_columns_ = {col: ["_".join([str(col), v])
                                        for v in self.cat_map_[col]]
                                       for col in self.cat_columns}
                               
                               
        self.transformed_columns_ = pd.Index(
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns)))
        
        return self
    
    #Only previously seen values will be encoded
    def transform(self, X, y=None, *args, **kwargs):      
        X2 = X.reindex(columns=self.cat_columns, copy=True)
        
        # Include all categories that may be missing
        for col in self.cat_columns:
            X2[col] = X2[col].apply(str).astype('category', categories=self.cat_map_[col])
        
        return pd.get_dummies(X2, sparse=True).reindex(columns=self.transformed_columns_).to_coo()
    
        
#Converts text data
class TextTransformer(BaseEstimator, TransformerMixin):
    #Set up the categorical columns
    def __init__(self, text_columns_):
        self.text_columns_=text_columns_
        
    # Fit each column of text data using CountVectorizer and Tfidf
    def fit(self, X, y=None, *args, **kwargs):
        
        X = X.reindex(columns=self.text_columns_, copy=True, fill_value='')
        
        transformers = []
        #Create a transformer for each text document
        for col in self.text_columns_:
            pipe = (col, Pipeline([('selector', ItemSelector(col)),
                                   ('count', CountVectorizer(min_df=0.005, stop_words='english')),
                                   ('tfidf', TfidfTransformer())]))
            
            transformers.append(pipe)
        
        self.transformer = FeatureUnion(transformers)
        self.transformer.fit(X, y=None)
        
        # Map of text and words to columns
        self.transformed_lengths_ = [len(t[1].steps[1][1].vocabulary_) for t in transformers]
        self.transformed_columns_ = []
        
        # Vocabulary maps words to columns, unmap to get corresponding words
        for i, col in enumerate(self.text_columns_):
            t = transformers[i]
            kvs = t[1].steps[1][1].vocabulary_
            words = sorted(list(kvs.keys()), key=lambda k: kvs[k])
            self.transformed_columns_ += ['{:}_{:}'.format(col, w) for w in words]

        return self
        
    
    def transform(self, X, y=None, *args, **kwargs):
        return self.transformer.transform(X.reindex(columns=self.text_columns_, copy=True, fill_value=''))
    
    
def GetPreprocessor(nums=None, cats=None, text=None):
    nums = nums if nums else []
    cats = cats if cats else []
    text = text if text else []
    
    transformers = []
    
    if len(nums) > 0:
        transformers.append(('nums', NumericTransformer(nums)))
        
    if len(cats) > 0:
        transformers.append(('cats', CategoricTransformer(cats)))
        
    if len(text) > 0:
        transformers.append(('text', TextTransformer(text)))
        
    return FeatureUnion(transformers)