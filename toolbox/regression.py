import random
import scipy
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, PoissonRegressor, HuberRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------------------
# Standard regressions
#-------------------------------------------------------------------------------------------

def run_ols(X_labels_, y_label_, df_, covariates=None, n_splits=None, plot=False):
    '''
        Run ordinary least squares regression
        Renames variables as needed to work with statsmodels api

        Arguments
        ---------
        X_labels_ : list of str
            Independent variables, reflecting column names in df_
            Can specify interactions: e.g., 'variable_1 * variable_2'
        y_label_ : str
            Dependent varibale, reflecting a column name in df_
        df_ : pd.DataFrame
            Includes all data to run regression
        covariates : list of str (optional)
            Variables to include but not output results for 
            Default: None
        n_splits : int (optional)
            If want cross-validation 
            Default: None
        plot : bool (optional)
            _description_ 
            Default: False

        Returns
        -------
        pd.DataFrame 
            Results

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    
    # organize X & y 
    df = df_.copy()
    # df = df.fillna(df.mean()) # maybe add a warning?
    
    # rename x labels
    X_labels = X_labels_.copy()
    for i, x_label_ in enumerate(X_labels_):
        if '.' in x_label_:
            x_label = x_label_.replace('.', '_')
            X_labels[i] = x_label
            df.rename(columns={x_label_:x_label}, inplace=True)
    
    # rename y label
    y_label = y_label_
    if '.' in y_label: y_label = y_label.replace('.', '_')
    df.rename(columns={y_label_:y_label}, inplace=True)
    
    if covariates is not None: X_ = X_labels + covariates
    else: X_ = X_labels
    
    f = y_label + ' ~ ' + ' + '.join(X_)
    y, X = patsy.dmatrices(f, df, return_type="dataframe")
    
    # run model(s)
    if n_splits is not None:

        train_pvalues = []
        train_rsquared_adj = []
        train_bic = [] 
        
        test_rmse = []
        test_rsquared_adj = []
        test_bic = []

        for f, (train_ix, test_ix) in enumerate(KFold(n_splits=n_splits, random_state=0, shuffle=True).split(X)):
            
            X_train = X.iloc[train_ix, :]            
            X_test  = X.iloc[test_ix, :]
            y_train = scipy.stats.zscore(y.iloc[train_ix, :])
            y_test  = scipy.stats.zscore(y.iloc[test_ix, :])          

            # fit the model on training data
            ols_obj = sm.OLS(y_train, X_train, missing='drop').fit()
            train_pred = ols_obj.predict(X_train) 
            y_train = y_train.values.reshape(-1)
            if len(X_labels) > 1: train_pvalues.append(list(ols_obj.pvalues[1:1+len(X_labels)].values))
            else: train_pvalues.append(ols_obj.pvalues[1])
            train_rsquared_adj.append(ols_obj.rsquared_adj)
            train_bic.append(ols_obj.bic)
            
            # intercept = ols_obj.params[0], coefs = ols_obj.params[1:].values
            # pred_y = intercept + np.dot(coefs, test.iloc[:,:-1].T) # intercept + (num_features .* (num_features, observations))

            # evaluate testing data performance 
            test_pred = ols_obj.predict(X_test) 
            y_test = y_test.values.reshape(-1)
            k = len(X_labels_) # predictors
            n = len(df_) # observations
            test_resid = y_test - test_pred
            
            # rmse
            test_rmse.append(np.mean((test_resid) ** 2))
            
            # bic
            test_bic.append(compute_bic(n, k, test_resid))
            
            # r-squared
            r2 = r2_score(y_test, test_pred)
            adj_r2 = 1 - ((1-r2)*(n-1)/(n-k-1))
            test_rsquared_adj.append(adj_r2)

            if plot:
                # this was hsowing prediction accuracy, rather than just the correlaiton between...
                # sns.scatterplot(x=train_pred, y=y_train, alpha=0.6, color='darkblue', ax=axs[0])
                # axs[0].set_xlim(np.min(y_train),np.max(y_train))
                # axs[0].set_ylim(np.min(y_train),np.max(y_train))
                # axs[0].plot([np.min(train_pred), np.max(train_pred)], [np.min(y_train), np.max(y_train)], color='grey', ls="--")
                
                fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))
                sns.regplot(x=train_pred, y=y_train, scatter_kws={'alpha':0.6}, color='darkblue', ax=axs[0])
                axs[0].set_xlabel('Predicted (training), fold=' + str(f+1), fontsize=15)
                axs[0].set_ylabel('Observed (training), fold=' + str(f+1), fontsize=15)
                sns.regplot(x=test_pred, y=y_test, scatter_kws={'alpha':0.6}, color='darkblue', ax=axs[1])

                axs[1].set_xlabel('Predicted (testing), fold=' + str(f+1), fontsize=15)
                axs[1].set_ylabel('Observed (testing), fold=' + str(f+1), fontsize=15)
                plt.tight_layout()
            
        index = ['fold_' + str(f+1) for f in np.arange(0,n_splits)]
        train_pvalues = pd.DataFrame(train_pvalues, index=index, columns=[l + '_train-pvalue' for l in X_labels])
        train_rsquared_adj = pd.DataFrame(train_rsquared_adj, index=index, columns=['train-r_squared_adj'])
        train_bic = pd.DataFrame(train_bic, index=index, columns=['train-BIC'])
        
        test_rsquared_adj = pd.DataFrame(test_rsquared_adj, index=index, columns=['test-r_squared_adj'])
        test_bic = pd.DataFrame(test_bic, index=index, columns=['test-BIC'])
        test_rmse = pd.DataFrame(test_rmse, index=index, columns=['test-RMSE'])
        
        cv_df = pd.concat([train_pvalues, train_bic, train_rsquared_adj, test_bic, test_rsquared_adj, test_rmse], axis=1)
        mean_df = pd.DataFrame(np.mean(cv_df, 0)).T
        mean_df.index = ['mean']
        output_df = pd.concat([cv_df, mean_df], 0)
        
        return output_df

    else:

        ols_obj = sm.OLS(y, X, missing='drop').fit()
        cols    = list(ols_obj.pvalues[1:].index)
        pvalues = [p for p in ols_obj.pvalues[1:]]
        betas   = [p for p in ols_obj.params[1:]]            
        res     = np.array(pvalues + betas + [ols_obj.bic, ols_obj.rsquared_adj]).reshape(1,-1)
        output_df = pd.DataFrame(res, columns=[l + '_pvalue' for l in cols] + [l + '_beta' for l in cols] + ['BIC', 'r_squared-adj'])
        
        if plot:
            pred_y = ols_obj.predict(X)
            obs_y = y.values.reshape(-1)
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.scatterplot(x=pred_y, y=obs_y, alpha=0.6, color='darkblue')
            ax.plot([np.min(pred_y), np.max(pred_y)], [np.min(obs_y), np.max(obs_y)], color='grey', ls="--")
            ax.set_xlabel('Predicted', fontsize=15)
            ax.set_ylabel('Observed', fontsize=15)
            plt.show()
                
        return  output_df, ols_obj
        
#-------------------------------------------------------------------------------------------
# Non-standard/non-linear
#-------------------------------------------------------------------------------------------

class ExponentialRegression:
    '''
    TODO: add __init__, description
    [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
        
    def fit(self, X, y):
        
        # define inputs
        if X.ndim == 1: X = X.reshape(-1,1)
        self.X = X.astype(float)
        self.y = y.astype(float)
        
        # log transform y
        log_y = np.log(self.y + 0.00001) # add a tiny amount to avoid infs with 0 values
        
        # fit with ols
        self.exp_reg = LinearRegression().fit(self.X, log_y)
        self.coef_lin = self.exp_reg.coef_ 
        self.intercept_ = self.exp_reg.intercept_
        
        return self

    def predict(self, X):
        
        if X.ndim == 1: X = X.reshape(-1,1)
        X = X.astype(float)

        # inverse transform to get predictions
        # y = exp(int) * exp(beta * x)
        predictions = np.exp(self.intercept_) * np.exp(self.coef_lin @ X.T)
        
        return predictions
    

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
        self.poly_reg = LinearRegression().fit(poly_X, y) # fitted object
        self.coef_ = self.poly_reg.coef_
        
        return self

    def predict(self, X):
        
        poly_X = self.transform_features(X)
        predictions = self.poly_reg.predict(poly_X)  
        
        return predictions
       
#-------------------------------------------------------------------------------------------
# Performance metrics & plots
#-------------------------------------------------------------------------------------------

def root_squared_error(y, predicted):
    rss = np.sum((y - predicted) ** 2)
    rse = (rss / (len(y) - 2)) ** 0.5
    return rse


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

#-------------------------------------------------------------------------------------------
# Orthogonalization - CAUTION: all trying to use gram schmidt but may give diff results? needs more testing
#-------------------------------------------------------------------------------------------

# def gram_schmidt(X):
#     ''' 
#         orthogonalize columns in matrix, using Gram-Schmidt Orthogonalization.
#         order matters
#         return in same shape as original
#     '''
#     Q, R = np.linalg.qr(X)
#     return Q.T
 

# def orthogonalize(U, eps=1e-15):
#     """
#     Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
#     If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
#     will be 0.
    
#     Args:
#         U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
#         eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
#     Returns:
#         (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
#             not linearly independent, then the last n-r cols are zeros.
    
#     Examples:
#     ```python
#     >>> import numpy as np
#     >>> import gram_schmidt as gs
#     >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
#     array([[ 0.81923192, -0.57346234],
#        [ 0.57346234,  0.81923192]])
#     >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
#     array([[ 0.81923192 -0.57346234  0.          0.        ]
#        [ 0.57346234  0.81923192  0.          0.        ]])
#     ```
#     """
#     U = np.array(U)
#     n = len(U[0])
#     V = U.T # work with transpose(U) for ease
#     for i in range(n):
#         prev_basis = V[0:i]     # orthonormal basis before V[i]
#         coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
#         V[i] -= np.dot(coeff_vec, prev_basis).T # subtract projections of V[i] onto already determined basis V[0:i]
#         if np.linalg.la.norm(V[i]) < eps:
#             V[i][V[i] < eps] = 0.   # set the small entries to 0
#         else:
#             V[i] /= np.linalg.la.norm(V[i])
#     return V.T
