#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    This file contains model comparison and selection related functions
'''

import numpy as np 

def loglike(n, resid):
    """
        from statsmodels api
        https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/linear_model.py    
    """
    n2 = n / 2.0
    ssr = np.sum(resid ** 2)
    return -n2 * np.log(2 * np.pi) - n2 * np.log(ssr / n) - n2 

def compute_bic(n, k, resid):
    """
        Bayes' information criteria from statsmodels api
            https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/linear_model.py
        For a model with a constant :math:`-2llf + \log(n)(df\_model+1)`.
        For a model without a constant :math:`-2llf + \log(n)(df\_model)`.
        
        n=number of observations
        k=number of parameters
        resid=residuals from linear model
    """
    llf = loglike(n, resid)
    return (-2 * llf + np.log(n) * (k))

    
def compute_aic(n, k, resid):
    """
        can have a small sample correction...
        Akaike's information criteria from statsmodels api
            https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/linear_model.py
        For a model with a constant :math:`-2llf + 2(df\_model + 1)`. 
        For a model without a constant :math:`-2llf + 2(df\_model)`.
    """
    llf = loglike(n, resid)
    return -2 * llf + 2 * (k)
