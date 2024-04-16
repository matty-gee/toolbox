import random, patsy
import scipy
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.stats import chi2
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt

flatten_lists = lambda l: [item for sublist in l for item in sublist]
  
def nanzscore(arr):
   arr_z = np.array([np.nan] * len(arr)) # create an array of nans the same size as arr
   arr_z[np.isfinite(arr)] = scipy.stats.zscore(arr[np.isfinite(arr)]) # calculate the zscore of the masked array
   return arr_z


#-------------------------------------------------------------------------------------------
# Standard regressions
#-------------------------------------------------------------------------------------------

def run_ols(X, y=None, covariates=None, data=None,
            loglik_ratio=False, 
            popmean=0, 
            verbose=False):
    '''
        Run ordinary least squares regression or 1-sample t-test
        Wrapper for statsmodels OLS
        Cleans up strings to be compatible with patsy & statsmodel formulas

        Inputs:
            X: Dataframe, series, list strings, string, or array
            y (optional): Dataframe, series, list strings, string, or array
            data (optional): Dataframe, for use if X & y are strings
            covariates (optional): list of strings, for use if X & y are strings

            popmean (optional): float or int, for use if t-test

            plot (optional): bool, plot the predicted vs observed values
        
        Outputs:
            out_df: Dataframe, summary of regression results
            ols_obj: statsmodels OLS object
    '''

    # helpers
    def clean_string(string):
        if '.' in string:
            string = string.replace('.', '_')
        return string

    def make_dataframe(a):
        if isinstance(a, pd.DataFrame): 
            return a.reset_index(drop=True)
        if isinstance(a, pd.Series): 
            return a.to_frame().reset_index(drop=True)
        else:
            if isinstance(a, list): a = np.array(a)
            if a.ndim == 1: a = a[:, np.newaxis]
            return pd.DataFrame(a)

    def nanzscore(arr):
        arr_z = np.array([np.nan] * len(arr)) # create an array of nans the same size as arr
        arr_z[np.isfinite(arr)] = scipy.stats.zscore(arr[np.isfinite(arr)]) # calculate the zscore of the masked array
        return arr_z

    # handle strings
    if isinstance(X, str) or (isinstance(X, list) and isinstance(X[0], str)):

        # is it a formula?
        if '~' in X:
           
            df = data.copy()
            y, X = patsy.dmatrices(X, df, return_type="dataframe")

        # if not, construct the formula
        else:

            if (isinstance(X, str)): X = [X]
            cols = list(np.unique(flatten_lists([x.split('*') if '*' in x else [x] for x in X] + [[y]])))
            if covariates is not None: 
                if (isinstance(covariates, str)): covariates = [covariates]
                cols.extend(covariates)
            df = data[cols].copy() # restrict to the columns provided (exclude interaction labels)

            # remove '.' & rename columns
            X_labels = X.copy()
            if any(['.' in col for col in df.columns]):
                for i, x_label_ in enumerate(X):
                    X_labels[i] = clean_string(x_label_)
                    df.rename(columns={x_label_:X_labels[i]}, inplace=True)
                df.rename(columns={y:clean_string(y)}, inplace=True)

            # create dataframes from patsy formula
            X_labels = X_labels + covariates if covariates is not None else X_labels
            y, X = patsy.dmatrices(f'{clean_string(y)} ~ ' + ' + '.join(X_labels), df, return_type="dataframe")

    # handle series, dataframes, arrays or lists
    else: 

        # build the regression from arrays
        if y is not None: 
            y = make_dataframe(y)
            if isinstance(y.columns[0], int): y.columns = ['y']
            X = make_dataframe(X)
            if all([isinstance(c, int) for c in X.columns]): X.columns = [f'x{i+1}' for i in range(X.shape[1])]
            X = sm.add_constant(X)
            X.rename(columns={'const':'Intercept'}, inplace=True)

        # if y is None, then its a 1-sample t-test: X ~ intercept
        else:
            y = make_dataframe(X)
            xcol = y.columns[0] if not isinstance(y.columns[0], int) else 'x1'
            X = pd.DataFrame(np.ones(len(X)), columns=[xcol])
            y.rename(columns={y.columns[0]:f' against {popmean}'}, inplace=True) # rename so easier to interpret in output
            y = y - popmean

    # z-score continuous variables (ols only)
    y[y.columns[0]] = nanzscore(y[y.columns[0]])
    if verbose: print(f'Z-scored y: {y.columns[0]}')
    for col in X.columns:
        if (X.dtypes[col] in ['float64', 'int64']) & (col not in ['Intercept', 'const']) & (X[col].nunique() > 2):
            X[col] = nanzscore(X[col])
            if verbose: print(f'Z-scored x: {col}')
        
    # run the regression 
    try: 
        ols    = sm.OLS(y, X, missing='drop').fit()
        out_df = pd.DataFrame(ols.summary2().tables[1])

        if loglik_ratio:
            ll_0 = sm.OLS(y, sm.add_constant(np.ones(len(y))), missing='drop').fit() # null model (intercept-only)
            ll_r, ll_r_p = ols.compare_lr_test(ll_0)[:2] # compare against null
    
    except Exception as e:
        print(f'Error: {e}')
        return X, y

    # clean up the output
    out_df.columns = ['beta', 'se', 't', 'p', '95%_lb', '95%_ub']
    out_df.reset_index(inplace=True)
    out_df.rename(columns={'index': 'x'}, inplace=True)
    out_df['X'] = (' + ').join(X.columns.tolist())
    out_df['y'] = y.columns[0]
    out_df['dof'] = ols.df_resid
    # out_df['z'] = scipy.stats.norm.ppf(1 - out_df['p'] / 2) # z-score for 2-tailed test

    if loglik_ratio: 
        out_df['ll'] = ols.llf
        out_df['llr'] = ll_r
        out_df['llr_p'] = ll_r_p
    out_df['adj_rsq'] = np.round(ols.rsquared_adj, 3)
    out_df['bic'] = np.round(ols.bic, 2)
    out_df['aic'] = np.round(ols.aic, 2)
    out_df['p_right'] = [p/2 if b > 0 else 1-p/2 for b, p in zip(out_df['beta'].values, out_df['p'].values)]
    out_df['p_left']  = [p/2 if b < 0 else 1-p/2 for b, p in zip(out_df['beta'].values, out_df['p'].values)]

    if loglik_ratio:
        out_df = out_df[['X', 'y', 'x', 'dof', 'll', 'llr', 'llr_p', 'adj_rsq', 'bic', 'aic', 'beta', 'se', 
                         '95%_lb', '95%_ub', 't', 'p', 'p_left', 'p_right']]
    else:
        out_df = out_df[['X', 'y', 'x', 'dof', 'adj_rsq', 'bic', 'aic', 'beta', 'se', 
                         '95%_lb', '95%_ub', 't', 'p', 'p_left', 'p_right']]
    if 'against' in y.columns[0]: out_df['X'] = 't-test'

    return out_df, ols

# TESTS: 
# x = data['affil_mean_mean']
# popmean = 0
# t, p = scipy.stats.ttest_1samp(x.values, popmean=popmean, alternative='less')
# print(f't={t}, p={p}')

# # t-test checks
# display(run_ols(x.values, popmean=popmean)[0]) # array
# display(run_ols(list(x), popmean=popmean)[0]) # list
# display(run_ols(x, popmean=popmean)[0]) # series

# # ols checks
# display(run_ols(data['affil_mean_mean'].values, data['memory_mean_main'].values)[0]) # 2 arrays
# display(run_ols(data['affil_mean_mean'], data['memory_mean_main'])[0]) # 2 series
# display(run_ols('affil_mean_mean', 'memory_mean_main', data)[0]) # 2 strings and a dataframe

#-------------------------------------------------------------------------------------------
# Inference
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

def likelihood_ratio(ll_0, ll_1):
    # log likelihood for m_0 (H_0) must be <= log likelihood of m_1 (H_1)
    # checked against statsmodels
    return(-2 * (ll_0 - ll_1))

def likelihood_ratio_test(ll_0, ll_1, df=1):
    # df is the difference in number of parameters between the two models; usually 1
    # checked against statsmodels
    lr = likelihood_ratio(ll_0, ll_1)
    p  = 1 - chi2(df).cdf(lr) # since lr follows χ2
    return(lr, p)

def compute_rse(y, predicted):
    rss = np.sum((y - predicted) ** 2)
    return (rss / (len(y) - 2)) ** 0.5

def compute_r2_adjusted(r2, n, p):
    '''
    r2: r2 score
    n: number of observations
    p: number of features
    '''
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


#-------------------------------------------------------------------------------------------
# Non-standard/non-linear regressions
#-------------------------------------------------------------------------------------------


class ExponentialRegression:
    '''
    TODO: add __init__, description
    [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
        
    def fit(self, X, y):
        
        # define inputs
        if X.ndim == 1: X = X[:,np.newaxis]
        self.X = X.astype(float)
        self.y = y.astype(float)
        
        # log transform y
        # the amount I add seems to affect fit... 
        log_y = np.log(self.y + 0.0000001) # add a tiny amount to avoid infs with 0 values
        
        # fit log w/ transformed y with ols
        self.exp_reg    = LinearRegression().fit(self.X, log_y)
        self.coef_lin   = self.exp_reg.coef_ 
        self.intercept_ = self.exp_reg.intercept_
   
        return self

    def predict(self, X):
        
        if X.ndim == 1: X = X[:,np.newaxis]
        X = X.astype(float)

        # take exp (inverse transform of log) to get predictions
        # y = exp(int) * exp(dot(beta, x))
        self.yhat = np.exp(self.intercept_) * np.exp(self.coef_lin @ X.T)
        
        # compute r2
        self.r2     = r2_score(self.y, self.yhat)
        self.r2_adj = compute_r2_adjusted(self.r2, self.X.shape[0], self.X.shape[1])

        return self.yhat
    
class PolynomialRegression:
    '''
    TODO: add __init__, description
    [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    def transform_features(self, X):
        
        from sklearn.preprocessing import PolynomialFeatures
        if X.ndim == 1:
            X = X.reshape(-1,1)
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        poly_X = poly.fit_transform(X)
        
        return poly_X
    
    def fit(self, X, y, degree):
        
        # define inputs
        self.X = X
        self.y = y
        self.degree = degree
        
        # transform features
        poly_X = self.transform_features(self.X)
        
        # fit with ols
        self.poly_reg  = LinearRegression().fit(poly_X, y) # fitted object
        self.coef_      = self.poly_reg.coef_
        self.intercept_ = self.poly_reg.intercept_

        return self

    def predict(self, X):
        
        poly_X    = self.transform_features(X)
        self.yhat = self.poly_reg.predict(poly_X)  
        
        # compute r2
        self.r2     = r2_score(self.y, self.yhat)
        self.r2_adj = compute_r2_adjusted(self.r2, self.X.shape[0], self.X.shape[1])
        
        return self.yhat


#-------------------------------------------------------------------------------------------
# Plotting
#-------------------------------------------------------------------------------------------


def regression_jointplots(xs, y, data, hue=None, dots_color='blue', run_ols=False,
                          height=6, ratio=5, space=0.2, kde_levels=0, x_jitter=.25, y_jitter=.25, xlabels=None, ylabel=None, margin_norm=False):
    """
    TAKEN FROM: https://jehyunlee.github.io/2020/10/03/Python-DS-35-seaborn_matplotlib2/
    
    -------------------
    Input Parameters
    -------------------
    xs      : (list or str) feature name(s) of data
    y       : (str) feature name of data
    data    : (pandas.DataFrame)
    hue     : (str) semantic variable that is mapped to determine the color of plot elements. Semantic variable that is mapped to determine the color of plot elements.
    
    height  : (float) size of the figure
    ratio   : (float) ratio of the joint axes height to marginal axes height.
    space   : (float) space between the joint and marginal axes
    
    xlabels : (list or str) xlabels
    ylabel  : (str) ylabel
    margin_norm : (boolean) if True, kdeplots at marginal axes have same scale.
    """
    ### 1. input check
    # input type
    assert isinstance(xs, list) or isinstance(xs, str)
    if isinstance(xs, list):
        assert all([isinstance(x, str) for x in xs])
    else:
        xs = [xs]
        
    if xlabels != None:
        assert isinstance(xlabels, list) or isinstance(xlabels, str)
        if isinstance(xlabels, list):
            assert all([isinstance(xlabel, str) for xlabel in xlabels])
        else:
            xlabels = [xlabels]
    
    if ylabel != None:
        assert isinstance(ylabel, str)
    
    if hue != None:
        assert isinstance(hue, str)
    
    # input data
    assert all([x in data.columns for x in xs])
    assert y in data.columns
    if hue != None:
        assert hue in data.columns
    
    ### 2. figure
    h_margin = height / (ratio + 1)
    h_joint = height - h_margin
    
    if isinstance(xs, list):
        n_x = len(xs)
    else:
        n_x = 1
    
    widths = [h_joint] * n_x + [h_margin]
    heights = [h_margin, h_joint]
    ncols = len(widths)
    nrows = len(heights)
    
    fig = plt.figure(figsize=(sum(widths), sum(heights)))
    
    ### 3. gridspec preparation
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, 
                            width_ratios = widths, height_ratios = heights,
                            wspace=space, hspace=space)
    
    ### 4. setting axes
    axs = {}
    for i in range(ncols * nrows):
        axs[i] = fig.add_subplot(spec[i//ncols, i%ncols])
    
    ### 5. jointplots (scatterplot + kdeplot)
    for i, x in enumerate(xs, ncols):
        if i == ncols:
            legend=True
        else:
            legend=False
            
        sns.kdeplot(x=x, y=y, data=data, hue=hue, alpha=0.3, ax=axs[i], levels=kde_levels, zorder=2, legend=False)
        sns.regplot(x=x, y=y, data=data, x_jitter=x_jitter, y_jitter=y_jitter, ax=axs[i], color=dots_color)
        
        # run statsmodel regression to get a pvalue
        if run_ols:
            X = sm.add_constant(data[x])
            pvalue = sm.OLS(data[y], X, missing='drop').fit().pvalues[1]
            axs[i].text(0.02, 0.95, f'ols pvalue={np.round(pvalue,5)}', horizontalalignment='left', 
                        size='large', color='black', transform=axs[i].transAxes)
        # if want to do hue, overlay multiple regplots...
    
    ### 6. kdeplots at marginal axes
    axs[ncols-1].axis("off")
    axes_mx = list(range(ncols-1))
    axes_my = 2*ncols - 1
    
    for i, x in zip(axes_mx, xs):
        sns.kdeplot(x=x, data=data, hue=hue, fill=True, ax=axs[i], levels=kde_levels, zorder=2, legend=False)
        axs[i].set_xlim(axs[i+ncols].get_xlim())
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].spines["left"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    
    sns.kdeplot(y=y, data=data, hue=hue, fill=True, ax=axs[axes_my], levels=kde_levels, zorder=2, legend=False)
    axs[axes_my].set_ylim(axs[ncols].get_ylim())
    axs[axes_my].set_ylabel("")
    axs[axes_my].set_yticklabels([])
    axs[axes_my].spines["bottom"].set_visible(False)
    axs[axes_my].spines["top"].set_visible(False)
    axs[axes_my].spines["right"].set_visible(False)
    
    if margin_norm == True:
        hist_range_max = max([axs[m].get_ylim()[-1] for m in axes_mx] + [axs[axes_my].get_xlim()[-1]])
        for i in axes_mx:
            axs[i].set_ylim(0, hist_range_max)
        axs[axes_my].set_xlim(0, hist_range_max)
        
    ### 7. unnecessary elements removal
    # 7.1. labels and ticklabels
    axes_j = list(range(ncols, 2*ncols-1))
    for i in axes_j:
        if i != ncols:
            axs[i].set_ylabel("")
            axs[i].set_yticklabels([])
    
    # 7.2. marginal axes
    for i in axes_mx:
        if i != 0:
            axs[i].set_ylabel("")
        axs[i].grid(False)
        axs[i].set_yticklabels([])
        yticks = axs[i].get_yticks()
        ylim = axs[i].get_ylim()
        for ytick in yticks:
            if 0 < ytick < ylim[-1]:
                axs[i].text(axs[i].get_xlim()[0], ytick, str(ytick), 
                            fontdict={"verticalalignment":"center"})
      
    axs[axes_my].grid(False)
    axs[axes_my].set_xticklabels([])
    axes_my_xticks = axs[axes_my].get_xticks()
    axes_my_xlim = axs[axes_my].get_xlim()
    for xtick in axes_my_xticks:
        if 0 < xtick < axes_my_xlim[-1]:
            axs[axes_my].text(xtick, axs[axes_my].get_ylim()[0], str(xtick), 
                              rotation=270, fontdict={"horizontalalignment":"center"})
    
    # 7.3. labels
    font_label = {"color": "black", "fontsize":"large"}
    labelpad = 12
    for i, x in zip(axes_j, xlabels):
        axs[i].set_xlabel(x, fontdict=font_label, labelpad=labelpad)
        if i == ncols:
            axs[i].set_ylabel(ylabel, fontdict=font_label, labelpad=labelpad)
    
    # remove density plot titles
    axs[0].set_ylabel("", fontdict=font_label, labelpad=labelpad)
    axs[2*ncols-1].set_xlabel("", fontdict=font_label, labelpad=labelpad)
    
    fig.align_ylabels([axs[0], axs[ncols]])
    fig.align_xlabels([axs[x] for x in range(ncols, 2*ncols)])
    plt.tight_layout()
    
    plt.show()
    
    return fig


#  def run_ols_cv():
#      # run model(s)
#     if n_splits is not None:

#         train_pvalues, train_rsquared_adj, train_bic = [], [], [] 
#         test_rmse, test_rsquared_adj, test_bic = [], [], []

#         for f, (train_ix, test_ix) in enumerate(KFold(n_splits=n_splits, random_state=0, shuffle=True).split(X)):
            
#             X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
#             y_train, y_test = zscore(y.iloc[train_ix, :]), zscore(y.iloc[test_ix, :])

#             # fit the model on training data
#             ols_obj = sm.OLS(y_train, X_train, missing='drop').fit()
#             train_pred = ols_obj.predict(X_train)
#             y_train = y_train.values.reshape(-1)
#             if len(X_labels) > 1: train_pvalues.append(list(ols_obj.pvalues[1:1+len(X_labels)].values))
#             else: train_pvalues.append(ols_obj.pvalues[1])
#             train_rsquared_adj.append(ols_obj.rsquared_adj)
#             train_bic.append(ols_obj.bic)

#             # intercept = ols_obj.params[0], coefs = ols_obj.params[1:].values
#             # pred_y = intercept + np.dot(coefs, test.iloc[:,:-1].T) # intercept + (num_features .* (num_features, observations))

#             # evaluate testing data performance 
#             test_pred = ols_obj.predict(X_test)
#             y_test = y_test.values.reshape(-1)
#             k = len(X_labels_) # predictors
#             n = len(df_) # observations
#             test_resid = y_test - test_pred

#             # rmse
#             test_rmse.append(np.mean((test_resid) ** 2))

#             # bic
#             test_bic.append(compute_bic(n, k, test_resid))

#             # r-squared
#             r2 = r2_score(y_test, test_pred)
#             adj_r2 = 1 - ((1-r2)*(n-1)/(n-k-1))
#             test_rsquared_adj.append(adj_r2)

    
