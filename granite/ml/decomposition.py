# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:12:36 2018

@author: Mike Staddon

Performs matrix decomposition such as PCA
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot_projection(X, c=None):
    # Scale components for best performance
    Xt = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(Xt)
    
    Xt = pca.transform(Xt)
    
    plt.scatter(Xt[:, 0], Xt[:, 1], c=c, alpha=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.gcf().set_size_inches(5,5)
    plt.show()
    
    
def plot_components(X):
    # Scale components for best performance
    Xt = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(Xt)
    
    
    # Show 1st component
    comps = pca.components_
    
    for i in range(comps.shape[0]):
        plt.barh(X.columns, comps[i, :])
        plt.title('PC'+str(i + 1))
        plt.show()



if __name__ == '__main__':
    # Toy datasets
    titanic = pd.read_csv('D:/Kaggle/Titanic/train.csv')
    houses = pd.read_csv('D:/Kaggle/House Prices/train.csv')
    
    X = titanic[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']].dropna()
    
    plot_projection(X, c=X['Survived'])
    plot_components(X)
    
    X = houses[['1stFlrSF', '2ndFlrSF', 'SalePrice', 'OverallQual', 'YearBuilt', 'PoolArea']]
    plot_projection(X, c=X['SalePrice'])
    plot_components(X)
    
    
    