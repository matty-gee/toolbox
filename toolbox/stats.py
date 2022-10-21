import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multi
import scipy

#-------------------------------------------------------------------------------------------
# MCC
#-------------------------------------------------------------------------------------------

def correct_pvals(pvals, method='fdr_bh'):
    return multi.multipletests(pvals, alpha=0.01, method=method)[1]

#-------------------------------------------------------------------------------------------
## CIs
#-------------------------------------------------------------------------------------------

def parametric_ci(coeffs, conf=99):
    '''
        Computes parametric confidence interval: 
        Inputs:
            coeffs: 1-d array coefficients to compute confidence interval over
            conf: confidence level, in integers

        same as: scipy.stats.t.interval
    '''
    from scipy.stats import sem, t
    n  = len(coeffs)
    df = n - 1
    m  = np.mean(coeffs)
    se = sem(coeffs)
    h  = se * t.ppf((1 + (conf/100)) / 2, df)
    # TO DO - use normal dist, rather than t dist, if n >= 30
    return [m - h, m + h]


def bootstrap_ci(coeffs, conf=99, num_samples=5000): 
    '''
        Computes a bootstrapped distribution of coefficients w/ confidence interval
            - ie, sample w/ replacement n num of subjects from the sample, num_samples times
        Inputs: 
            coeffs: 1d array of parameter coefficients  
    '''

    sample_size = len(coeffs) 
    xb = np.random.choice(coeffs, (sample_size, num_samples)) 
    mb = xb.mean(axis=0)
    mb.sort()
    lb = (100-conf)/2
    ub = 100-((100-conf)/2)
    ci = np.percentile(mb, [lb,ub]) 
    return ci

#-------------------------------------------------------------------------------------------
# Effect sizes
#-------------------------------------------------------------------------------------------

def cohens_d_rm(measure_1, measure_2, conf=99):
    '''
        rm = repeated measures
        not clear this will work well with this data, esp the probability analysis, since between measures is r=1 
    '''
    
    from scipy.stats import t
    
    n    = len(sample)
    diff = measure_1 - measure_2
    
    mean_diff = np.mean(diff)
    pearson_r = scipy.stats.pearsonr(measure_1, measure_2)
    std_rm    = (np.std(diff))/np.sqrt(2*(1-pearson_r)) # not going to work w/ perfect correlations
    d_rm      = mean_diff / std_rm
    
    return d_rm


def cohens_d_rm_ci(d_rm, n):
    # can also get a ci for cohens d, although apparently not agreed upon
    se = np.sqrt((2*(1-r)/n) + (d**2/2*n))
    h  = se * t.ppf((1 + (conf/100)) / 2, n - 1)
    ci = [d_rm - h,  d_rm + h]


def cohens_d_av(measure_1, measure_2): 
    '''
        cohens d with average variance: standardized effect size for t-tests for within subjects repeated measures 
        cohens d av = difference in means/mean standard deviation
        
        inputs:
            measure_1:
            measure_2:
        outputs: d
        
        notes: 
            - ignores correlation between measures; there are alternative(s) which take into account
            - can be easily compared to fx sizes for independent samples
        e.g.,: https://www.researchgate.net/post/Calculating-variance-of-Cohens-d-for-repeated-measures-designs
    '''
    
    mean_diff = np.mean(measure_1 - measure_2)
    mean_std  = np.mean((np.std(measure_1), np.std(measure_2)))    
    d_av      = mean_diff / mean_std        
    return d_av


def cohens_d_1samp(sample, popmean=0, hedges=False):
    '''
        cohens d for 1 sample t-test: standardized effect size for 1 sample t-tests
            - can also solve repeated measures cohens d, by inputting the differences between the measures 
        cohens d = (sample mean - null hypothesis mean)/ standard deviation of sample
        
        inputs:
            sample: array of sample estimates
            popmean: null hypothesis
            hedges: boolean indicating whether to use hedges correction to the estimate
        outputs:
            cohens d 
    '''
    d = np.abs((np.mean(sample) - popmean) / np.std(sample))
    return d


def cohens_d_1samp_ci(d, n, conf=99):
    '''
       compute a parametric ci: d +/- se * t_crit
    '''
    # https://www.real-statistics.com/students-t-distribution/one-sample-t-test/confidence-interval-one-sample-cohens-d/
    se = np.sqrt((1/n) + (d**2)/(2*n)) 
    h  = se * t.ppf((1 + (conf/100)) / 2, n - 1)
    return [d - h, d + h]


def hedges_correction(d, df):
    '''
        biass correction for cohens d with small sample (n<20)
        e.g.,: https://stats.stackexchange.com/questions/1850/difference-between-cohens-d-and-hedges-g-for-effect-size-metrics
    '''
    return  d * (1 - (3 / (4 * df - 1)))