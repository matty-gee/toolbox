#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    functions related to nonparametric statistics
'''

import scipy
import pandas as pd
import numpy as np

# evaluate permutation significance
def get_perm_pval(stat, perm_stats, test='2-sided'):
    '''
        Calculate p-value of test statistic compared to permutation generated distribution.
        Finds percentage of permutation distribution that is at least as extreme as the test statistic

        Arguments
        ---------
        stat : float
            Test statistic
        perm_stats : array of floats
            Permutation generated (null) statistics
        test : str (optional)
            P-value to calculate: '2-sided', 'greater' or 'lesser'
            Default: '2-sided'

        Returns
        -------
        float 
            p-value

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    p_greater = np.sum(perm_stats >= stat) / len(perm_stats)
    p_lesser  = np.sum(perm_stats <= stat) / len(perm_stats)
    p_2sided  = min(p_greater, p_lesser)

    if test == '2-sided':   pval = p_2sided
    elif test == 'greater': pval = p_greater
    elif test == 'lesser':  pval = p_lesser

    return pval

def rank_list(aList):
    ''' convert list to ranked list '''
    sortList = aList.copy()
    sortList.sort()
    
    latestRank = 1
    rankDict = {}
    rankFreq = {}
    freqScore = 1
    for i in range(1, len(sortList)):

        if sortList[i] == sortList[i-1]:
            freqScore = freqScore + 1

        if sortList[i] != sortList[i-1]:
            rank = (2*latestRank + freqScore - 1) / 2

            rankDict[sortList[i-1]] = rank
            rankFreq[rank] = freqScore
            latestRank = latestRank + freqScore
            freqScore = 1

    # the last case
    rankDict[sortList[len(sortList)-1]] = (2*latestRank + freqScore - 1) / 2
    rankFreq[(2*latestRank + freqScore - 1) / 2] = freqScore
    
    # replace list scores with rank scores
    allRanks = []
    for i in aList:
        allRanks.append(rankDict[i])
    
    return rankFreq, rankDict, allRanks

def wilcoxon_1samp(data, med_H0=None, alternative='two-sided'):
    '''
        added alternative hypotheses... check that it works well
    '''
    # get an alternative hypothesis: median
    if med_H0 == None:
        med_H0 = sum(np.unique(data)) / len(np.unique(data))

    # do stuff
    d = []
    sgns = []
    for i in data:
        if i != med_H0:
            d = d + [abs(i - med_H0)]
            if i > med_H0:
                sgns = sgns + [1]
            else:
                sgns = sgns + [-1]

    # get a ranked list
    rank_freq, rank_dict, all_ranks = rank_list(d)
    n = len(all_ranks)
    var = n * (n + 1) * (2*n + 1) / 24

    T = 0
    for i in rank_freq:
        T = T + rank_freq[i]**3 - rank_freq[i]
    T = T / 48

    var_adj = var - T
    se_adj = var_adj ** 0.5
    D = n * (n + 1) / 4

    w_plus = 0
    w_min = 0
    for i in range(n):
        if sgns[i] > 0:
            w_plus = w_plus + all_ranks[i]
        else:
            w_min = w_min + all_ranks[i]

    w_adj = (w_plus - D) / se_adj # basically the z value
    
    # if n < 20, use scipy.stats for exact distribution
    if n < 20:
        rank, p = scipy.stats.wilcoxon(data-med_H0, zero_method='wilcox', correction=False, alternative=alternative)
        test_used = 'Wilcoxon one-sample (exact)'
    
    # if n >= 20, use approximation
    else:
        rank = w_plus
        p = scipy.stats.norm.sf(abs(w_adj))
        if alternative == 'two-sided':
            p=p*2            
        test_used = 'Wilcoxon one-sample (appr.)'
    
    # output
    results = pd.DataFrame(np.array([test_used, int(n), float(rank), float(w_adj), float(p)]).reshape(1,-1), 
                           columns=['test','n', 'rank', 'z','p'])
    
    return results