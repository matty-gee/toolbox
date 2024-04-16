import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multitest as multi
from statsmodels.formula.api import ols


def nanzscore(arr):
   arr_z = np.array([np.nan] * len(arr)) # create an array of nans the same size as arr
   mask = np.isfinite(arr) # mask the array
   arr_z[mask] = scipy.stats.zscore(arr[mask]) # calculate the zscore of the masked array
   return arr_z

#-------------------------------------------------------------------------------------------
# statistical tests
#-------------------------------------------------------------------------------------------


def run_ind_ttest(array1, array2, assumption_alpha=0.05, alternative='two-sided'):

    ''' 
        Runs an independent t-test on two arrays of data
        Tests for Normality using Shapiro-Wilk test
        Tests for Equal Variances using Levene's test
        Then runs the appropriate t-test (parametric or non-parametric)
        Returns the test type, statistic, and p-value
    '''

    # Check for Normality using Shapiro-Wilk test
    _, p1 = scipy.stats.shapiro(array1)
    _, p2 = scipy.stats.shapiro(array2)
    
    if p1 > assumption_alpha and p2 > assumption_alpha:
        normality = True
    else:
        normality = False
    
    # Check for Equal Variances using Levene's test
    _, p_var = scipy.stats.levene(array1, array2)
    
    if p_var > assumption_alpha:
        equal_var = True
    else:
        equal_var = False
    
    # Run the appropriate t-test
    if normality:
        test_type = 'Parametric'
        stat, p_val = scipy.stats.ttest_ind(array1, array2, equal_var=equal_var, alternative=alternative)
    else:
        test_type = 'Non-parametric'
        stat, p_val = scipy.stats.mannwhitneyu(array1, array2, alternative=alternative)
        
    
    return test_type, stat, p_val


def run_anova(data, dv, ivs, covs=None, typ=None, followup=True):

    """
        Run ANOVA or ANCOVA using statsmodels.

        Parameters:
        - data (DataFrame): The data frame containing the data
        - dependent_var (str): The dependent variable
        - independent_vars (list): A list of independent variables
        - covariates (list, optional): A list of covariates for ANCOVA

        Returns:
        - ANOVA table
    """

    # Create the formula
    formula = f"{dv} ~ {' + '.join(ivs)}"
    if covs is not None:
        formula += ' + ' + ' + '.join(covs)

    # Run the ANOVA
    ols_model = ols(formula, data=data).fit() # fit the model
    if typ is None:
        anova_table = anova_lm(ols_model, typ=2) # assume no interaction
        interx_terms = [term for term in anova_table.index if ':' in term] # test interactions 
        if len(interx_terms) > 0: 
            if np.any(anova_table['PR(>F)'][interx_terms].values < 0.05):
                anova_table = anova_lm(ols_model, typ=3) # rerun as type 3 if significant
    else:
        anova_table = anova_lm(ols_model, typ=typ)

    # if followup:
    # run basic t-tests
    
    return anova_table
# TODO : add assumption checks
# # 2. Check Normality
# for group in df[independent_var].unique():
#     group_data = df[df[independent_var] == group][dependent_var]
#     _, p_value = stats.shapiro(group_data)
#     if p_value > 0.05:
#         print(f'2. Normality: Data for group {group} is normally distributed (p={p_value}).')
#     else:
#         print(f'2. Normality: Data for group {group} is not normally distributed (p={p_value}).')

# # 3. Check Homogeneity of Variances
# _, p_value = stats.levene(*[df[df[independent_var] == group][dependent_var] for group in df[independent_var].unique()])
# if p_value > 0.05:
#     print(f'3. Homogeneity of Variances: Satisfied (p={p_value}).')
# else:
#     print(f'3. Homogeneity of Variances: Not satisfied (p={p_value}).')


#-------------------------------------------------------------------------------------------
# model comparisons
#-------------------------------------------------------------------------------------------


def loglikehood(n, resid):
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
    return (-2 * loglikehood(n, resid) + np.log(n) * (k))
    
def compute_aic(n, k, resid):
    """
        can have a small sample correction...
        Akaike's information criteria from statsmodels api
            https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/linear_model.py
        For a model with a constant :math:`-2llf + 2(df\_model + 1)`. 
        For a model without a constant :math:`-2llf + 2(df\_model)`.
    """
    return -2 * loglikehood(n, resid) + 2 * (k)

def calc_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


#-------------------------------------------------------------------------------------------
# p-values
#-------------------------------------------------------------------------------------------


def correct_pvals(pvals, method='fdr_bh'):
    return multi.multipletests(pvals, alpha=0.01, method=method)[1]

def calc_perm_pvalue(stat, perm_stats, alternative='two-sided'):
    '''
    Calculate p-value of test statistic compared to permutation generated distribution.
    Finds percentage of permutation distribution that is at least as extreme as the test statistic

    Arguments
    ---------
    stat : float
        Test statistic
    perm_stats : array of floats
        Permutation generated (null) statistics
    alternative : str (optional)

    Returns
    -------
    float 
        p-value

    [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    n = len(perm_stats)
    p_greater = (np.sum(perm_stats >= stat) + 1) / (n + 1) # add one for a continuity correction
    p_less    = (np.sum(perm_stats <= stat) + 1) / (n + 1)

    if alternative == 'two-sided':   
        # accounts for scenario where the observed statistic is exactly at the median of the permutation distribution
        return 2 * min(p_greater, p_less) if p_greater + p_less > 1 else 2 * max(p_greater, p_less) 
    elif alternative == 'greater':  
        return p_greater
    elif alternative == 'less':     
        return p_less


#-------------------------------------------------------------------------------------------
# confidence intervals
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
# effect sizes
#-------------------------------------------------------------------------------------------


def cohens_d_repeated(measure_1, measure_2, ci=99):

    n = len(measure_1)
    diff = measure_1 - measure_2
    
    # calculate d
    r = scipy.stats.pearsonr(measure_1, measure_2)
    std_rm = np.std(diff) / np.sqrt(2 * (1 - r)) # not going to work w/ perfect correlations
    d = np.mean(diff) / std_rm

    # calculate ci
    # apparently not agreed upon...?
    se = np.sqrt((2 * (1 - r) / n) + (d**2 / 2 * n))
    h  = se * scipy.stats.t.ppf((1 + (ci / 100)) / 2, n - 1)

    return d, [d - h,  d + h]

def cohens_d_avg(measure_1, measure_2): 
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
    return mean_diff / mean_std 

def cohens_d_1samp(sample, popmean=0, ci=99, hedges=False):
    '''
        cohens d for 1 sample t-test: standardized effect size for 1 sample t-tests
            - can also solve repeated measures cohens d, by inputting the differences between the measures 
        cohens d = (sample mean - null hypothesis mean)/ standard deviation of sample
        compute a parametric ci: d +/- se * t_crit

        inputs:
            sample: array of sample estimates
            popmean: null hypothesis
            hedges: boolean indicating whether to use hedges correction to the estimate
        outputs:
            cohens d 

         https://www.real-statistics.com/students-t-distribution/one-sample-t-test/confidence-interval-one-sample-cohens-d/
    '''

    n = len(sample)

    # calculate d
    d = np.abs((np.mean(sample) - popmean) / np.std(sample))
    if hedges:
        d = hedges_correction(d, len(sample))
    
    # calculate ci
    se = np.sqrt((1/n) + (d**2) / (2*n)) 
    h  = se * scipy.stats.t.ppf((1 + (ci/100)) / 2, n-1)
    return d, [d - h, d + h]

def hedges_correction(d, df):
    '''
        biass correction for cohens d with small sample (n<20)
        e.g.,: https://stats.stackexchange.com/questions/1850/difference-between-cohens-d-and-hedges-g-for-effect-size-metrics
    '''
    return  d * (1 - (3 / (4 * df - 1)))