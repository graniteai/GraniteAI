# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:03:03 2017

@author: Mike Staddon

Estimators and model fitting
"""

import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import OneHotEncoder


# Custom kernel to handle float and integer values
# Rounds floats with integer value to nearest int
class MixedTypeMatern(Matern):
    def __init__(self, integers=None, categorics=None, num_categories=None, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        """
        integers and categorics contain the indicies of the integer and categoric columns
        
        categories contains the number of categories for each categoric
        """

        super(MixedTypeMatern, self).__init__(length_scale, length_scale_bounds, nu)

        if integers is None:
            self.integers = []
        else:
            self.integers = integers
            
        if categorics is None:
            self.categorics = []
        else:
            self.categorics = categorics
            
        self.num_categories = num_categories
        
        
    # Round numeric values and one hot encode categoric
    def ProcessData(self, X):
        if X is None:
            return None
        
        nums = [i for i in range(X.shape[1]) if i not in self.categorics]
        
        # Process numeric values first
        Xt = np.copy(X)

        # Round integer values and categorics since they are indices
        Xt[:, self.integers+self.categorics] = Xt[:, self.integers+self.categorics].round()

        if len(self.categorics) > 0:
            # Take out numeric values
            Xn = np.copy(Xt[:, nums])
            
            # One hot encode categoric values
            Xc = np.copy(Xt[:, self.categorics])

            enc = OneHotEncoder(n_values=self.num_categories, sparse=False)
            Xc = enc.fit_transform(Xc)

            Xt = np.hstack((Xn, Xc))
        
        return Xt
            

    # Overload the __call__ method
    def __call__(self, X, Y=None, eval_gradient=False):
        # Process the data first
        return super(MixedTypeMatern, self).__call__(self.ProcessData(X), Y=self.ProcessData(Y), eval_gradient=eval_gradient)
    

# Suggest next point to optimise function given previous results
def BayesianOptimise(X, y, bounds, cost=None, integers=None, categorics=None, num_categories=None, return_pi=False, alphas=None):
    """
    X: previously evaluated points
    y: previous results
    bounds: numerical bounds for new X
    cost: cost to evaluate previous results - if not None then return best per
        unit cost
        
    integers: index of integer variables
    categorics: index of categoric variables
    num_categories: number of categories in each categoric variable
    
    return_pi: return probability of improvement or not
    
    alphas: error in y term
    """
    
    X = np.array(X)
    y = np.array(y)
    bounds = np.array(bounds)
    
    # Clean results - set nans to the worst value
    for i in range(len(y)):
        if y[i] in [np.nan, np.inf, -np.inf]:
            y[i] = np.amin(y)
            
    
    if alphas is None:
        alphas = 1e-5
    else:
        alphas = np.array(alphas)
    
    # Remove non categoric info
    num_categories = [n for n in num_categories if n != 0]
    
    if cost is not None:
        cost = np.array(cost)
        
    # Total columns after one hot encoding categoric features
    total_columns = X.shape[1]
    if num_categories is not None:
        total_columns += sum(num_categories) - len(num_categories)
    
    kernel = MixedTypeMatern(integers, categorics, nu=2.5, length_scale=[1.0]*total_columns, num_categories=num_categories)
    
    y_model = gp.GaussianProcessRegressor(kernel=1.0 * kernel + 1.0 * gp.kernels.WhiteKernel(),
                                          n_restarts_optimizer=50,
                                          alpha=0,
                                          normalize_y=True)
    
    y_model.fit(X, y)
    
    # Estimate the cost
    if cost is not None:
        
        kernel = 1.0 * MixedTypeMatern(integers, categorics, nu=2.5, length_scale=[1.0]*total_columns, num_categories=num_categories)
    
        cost_model = gp.GaussianProcessRegressor(kernel=1.0 * kernel + 1.0 * gp.kernels.WhiteKernel(),
                                                 n_restarts_optimizer=50,
                                                 normalize_y=True)
        
        cost_model.fit(X, np.log(cost))
    else:
        cost_model = None
        
    #Expected imrpovement in gaussian process at point x per unit time
    def expected_improvement(x, y_mod, previous_results, cost_mod=None, greater_is_better=True, n_params=1):
        x_to_predict = x.reshape(-1, n_params)
        
        mu, sigma = y_model.predict(x_to_predict, return_std=True)
        
        if greater_is_better:
            loss_optimum = np.max(previous_results)
        else:
            loss_optimum = np.min(previous_results)
    
        scaling_factor = (-1) ** (not greater_is_better)
        
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] == 0.0
    
        # Divide by time
        if cost_mod is None:
            return -1 * ei
        else:
            return -1 * ei / np.exp(cost_model.predict(x_to_predict))

    best_x = None
    best_func_value = 1
    n_restarts = 20
    n_params = len(bounds)

    # Optimise expected improvement for 10 runs
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=expected_improvement,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(y_model, np.array(y), cost_model, True, len(bounds)))

        if res.fun < best_func_value:
            best_func_value = res.fun
            best_x = res.x
            
    # Sample 1000 random points and take best - quicker than locally optimising
    n_random = 10000
    random_x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_random, n_params))
    random_ei = expected_improvement(random_x, y_model, np.array(y), cost_model, True, len(bounds))
    
    # If random is best use that
    if np.amin(random_ei) < best_func_value:
        best_func_value = np.amin(random_ei)
        best_x = random_x[np.argmin(random_ei), :]
    
            
    # Clean categoric and integer values
    best_x = list(best_x)
    for i in range(len(best_x)):
        if i in integers or i in categorics:
            best_x[i] = int(np.round(best_x[i]))
        

    # Probability of improvement
    if return_pi:
        mu, sigma = y_model.predict(np.array(best_x).reshape(1, -1), return_std=True)
        
        mu, sigma = mu[0], sigma[0]
        
        if sigma == 0:
            pi = 1 if mu > np.amax(y) else 0
        else:
            pi = norm.cdf((mu - np.amax(y))/sigma)
        
        return best_x, pi
    else:
        return best_x
