#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    This file contains functions useful for pattern analyses
'''

import scipy
from sklearn.metrics import pairwise_distances
from matrices import * 

def rank(arr, method='average'):
    """
        turn an array of values into corresponding ranked array 
        average seems like a reasonable way to resolve ties
    """
    ranked = rankdata(arr, method=method)
    return ranked
    
def rank_analysis(xlabels, ylabels, df, alternative='greater'):
    coefs = np.array([scipy.stats.kendalltau(sub[xlabels], sub[ylabels])[0] for s, sub in df.iterrows()])
    t,p   = scipy.stats.wilcoxon(coefs[np.isfinite(coefs)], alternative=alternative)
    return coefs, [t,p]

def similarity_analysis(xlabels, ylabels, df, metric_x='euclidean', metric_y='euclidean', alternative='greater'):
    coefs = []
    for s, sub in df.iterrows():
        dists_x = symm_mat_to_ut_vec(pairwise_distances(sub[xlabels].values.reshape(-1, 1)), metric=metric_x)
        dists_y = symm_mat_to_ut_vec(pairwise_distances(sub[ylabels].values.reshape(-1, 1)), metric=metric_y)
        coefs.append(scipy.stats.kendalltau(dists_x, dists_y)[0])
    t,p = scipy.stats.wilcoxon(np.array(coefs)[np.isfinite(coefs)], alternative=alternative)
    return coefs, [t,p]
