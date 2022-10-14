import random
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from circ_stats import circ_corrcl, circ_corrcc

# symmetrical correlation matrices
def corr_matrices(data_df):
    '''
        matrix: dataframe, with subs x variables
        outputs: correlation and p-value matrices
    '''
    
    feature_names = data_df.columns.values
    data_df = data_df.T # transpose to variables x sub
    
    r  = np.corrcoef(data_df)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = data_df.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p  = np.zeros(shape=r.shape)
    
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    
    corr_mat = pd.DataFrame(r, columns=feature_names, index=feature_names)
    pval_mat = pd.DataFrame(p, columns=feature_names, index=feature_names)
    
    return corr_mat, pval_mat
    
def plot_corr_mat(corr_mat, size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''   
    fig, ax = plt.subplots(figsize=(size, size))
    ax.grid(False)
    cax = ax.matshow(corr_mat, cmap='jet', vmin=-1, vmax=1)
    plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=90)
    plt.yticks(range(len(corr_mat.columns)), corr_mat.columns)
    cbar = fig.colorbar(cax, ticks = [-1, 0, 1], aspect = 40, shrink = .8)

# asymmetrical correlation matrices
def get_asymmetrical_corrmat(x_df_, y_df_, linear_correlation=scipy.stats.pearsonr, permutations=None):
    '''
        x_df & y_df should dfs of shape num_subs x num_variables
        robust to diff. amounts of missingness for diff. variables
        will do nonparametric correlations if specified & angular-angular & angular-linear correlations 
        also will do permutation testing
    '''
    
    x_df = x_df_.copy()
    y_df = y_df_.copy()

    y_cols = y_df.columns
    x_cols = x_df.columns
    
    # avoid problems with data type
    x_df[x_cols] = x_df[x_cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    y_df[y_cols] = y_df[y_cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    
    # turn any angles into radians...
    y_angles = ['angles' in c for c in y_cols]
    x_angles = ['angles' in c for c in x_cols]
    y_df[y_cols[y_angles]] = y_df[y_cols[y_angles]].apply(np.deg2rad) 
    x_df[x_cols[x_angles]] = x_df[x_cols[x_angles]].apply(np.deg2rad) 

    # for outputting
    coefs = np.zeros((len(y_cols), len(x_cols)))
    pvals = np.zeros((len(y_cols), len(x_cols)))
    
    for yi, y_col in enumerate(y_cols):
        for xi, x_col in enumerate(x_cols):

            # screen out subs w/ nans & infs 
            finite_mask = np.isfinite(x_df[x_col].values) * np.isfinite(y_df[y_col].values) 
            x = x_df[x_col][finite_mask]
            y = y_df[y_col][finite_mask]
            
            # correlation 
            if ('angles' not in x_col) & ('angles' not in y_col):
                correlation = linear_correlation 
            elif (('angles' in x_col) & ('angles' not in y_col)):
                correlation = circ_corrcl
            elif (('angles' not in x_col) & ('angles' in y_col)):
                correlation = circ_corrcl 
                y = x_df[x_col][finite_mask] # flip the order of circular & linear variables
                x = y_df[y_col][finite_mask]
            elif ('angles' in x_col) & ('angles' in y_col):
                correlation = circ_corrcc
            coefs[yi,xi], pvals[yi,xi] = correlation(x, y)
            
            # permutation significance testing
            if permutations: 
                perm_coefs = np.zeros(permutations)
                y_ = y.copy().values
                for n in range(permutations):
                    random.shuffle(y_)
                    perm_coefs[n], _ = correlation(x, y_)
                pvals[yi,xi] = get_perm_pval(coefs[yi,xi], perm_coefs, test='2-sided')
                
    coef_df = pd.DataFrame(coefs, index=y_cols, columns=x_cols)
    pval_df = pd.DataFrame(pvals, index=y_cols, columns=x_cols)
    return coef_df, pval_df

def plot_asymmetrical_corrmat(coef_df, pval_df, figsize=(15,15)):
    '''
    '''
    x_names  = coef_df.columns
    y_names  = coef_df.index
    coef_mat = coef_df.values.copy()
    pval_mat = pval_df.values.copy()

    # masked correlation plot
    masked_corrs = coef_mat * ((pval_mat < 0.05) * 1) 
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked_corrs, cmap="RdBu_r", vmax=1, vmin=-1) #np.max(masked_corrs)
    plt.grid(False)
    
    # 0.05: *, 0.01: **, 0.001: ***, <0.0001: ****
    for i in np.arange(len(y_names)):
        for j in np.arange(len(x_names)):
            if coef_mat[i,j] > 0.50:
                color = 'white'
            else:
                color = 'black'
            if pval_mat[i,j] < 0.0001:
                sig = '****'
            elif pval_mat[i,j] < 0.001:
                sig = '***'
            elif pval_mat[i,j] < 0.01:
                sig = '**'
            elif pval_mat[i,j] < 0.05:
                sig = '*'
            else:
                sig = ' '
            text = ax.text(j, i, sig, ha="center", va="center", color=color, fontsize=20)
            
    im_ratio = coef_mat.shape[0]/coef_mat.shape[1] # (height_of_image / width_of_image)
    cbar = plt.colorbar(ax.get_children()[-2], ax=ax, orientation='vertical', fraction=0.044*im_ratio)
    cbar.set_label(label='Correlation', size=20)
    cbar.ax.tick_params(labelsize=9)

    ax.set_yticks(np.arange(len(y_names)))
    ax.set_yticklabels(y_names, fontsize=15)
    ax.set_xticks(np.arange(len(x_names)))
    ax.set_xticklabels(x_names, fontsize=15, rotation=90)

    # add grid lines
    ax.set_xticks(np.arange(-.5, len(x_names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_names), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    
    ax.set_title('Correlations', fontsize=20)

    return fig