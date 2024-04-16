
import itertools

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits import axes_grid1
import seaborn as sns
from pylab import *
# import mplcursors

import numpy as np
import pandas as pd
import scipy

from general_utils import digitize_matrix

#----------------------------------------------------------------------------------------------
# defaults
#----------------------------------------------------------------------------------------------

alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] # for stars
tick_fontsize, label_fontsize, title_fontsize = 10, 13, 15
legend_title_fontsize, legend_label_fontsize = 12, 10
subfig_letter_fontsize = 16
suptitle_fontsize = title_fontsize * 1.5
ec, lw = 'black', 1
bw = 0.15 # this is a proportion of the total??


# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots

#----------------------------------------------------------------------------------------------
# palettes etc
#----------------------------------------------------------------------------------------------

def random_colors(num_colors=10):
    from random import randint
    return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(num_colors)]

def color_converter(colors):
    """ converts a list of colors to a dataframe with color name, hex, and RGB values
        input: list of colors
        output: dataframe with color name, hex, and RGB values
    """
    converted_colors = []
    for color in colors:
        # check if it's a string color name
        if isinstance(color, str) and color in mcolors.CSS4_COLORS:
            rgb = mcolors.CSS4_COLORS[color]
            rgb_normalized = mcolors.hex2color(rgb)  # normalized RGB
            converted_colors.append((color, rgb, rgb_normalized))
        # check if it's a hex color
        elif isinstance(color, str) and color.startswith('#'):
            color_name = None  # cannot reverse lookup color name from hex
            rgb_normalized = mcolors.hex2color(color)
            converted_colors.append((color_name, color, rgb_normalized))
        # check if it's a rgb color
        elif isinstance(color, tuple) and len(color) == 3:
            color_name = None  # cannot reverse lookup color name from RGB
            rgb = mcolors.rgb2hex(color)
            converted_colors.append((color_name, rgb, color))
        else:
            print(f"Invalid color {color}")
    return pd.DataFrame(converted_colors, columns=['color_name', 'hex', 'rgb'])

def get_cmap_colors(cmap):
    # eg: cmap = cm.get_cmap('plasma', 101)
    return [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

def plot_all_colors():
    
    colors = mcolors._colors_full_map #dictionary of all colors
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items()) #sort them by hsv
    sorted_names = [name for hsv, name in by_hsv]    
    fig = plt.figure(figsize=(17,35),dpi=200)
    grid = GridSpec(195,6,hspace=1,wspace=2)
    counter = 0
    while counter < 1163:
        column = (counter//195)%6
        row = counter%195
        ax = fig.add_subplot(grid[row,column])
        ax.axis('off')
        ax.axhline(0,linewidth=15,c=sorted_names[counter])
        color_str = sorted_names[counter]
        #recall that xkcd colors must be prefaced with "xkdc:". To save space, I'll take that out
        #and replace with an asterisk so we can still identify them
        if 'xkcd' in color_str:
            color_str = color_str[5:]+'*'
        ax.text(-0.03,0.5,color_str,ha='right',va='center')
        counter+=1
    else:
        ax = fig.add_subplot(grid[-4,-1])
        ax.axis('off')
        ax.text(0,0,'* = xkcd')
    #   fig.savefig('matplotlib named colors.png', bbox_inches='tight') #uncomment to save figure

def make_palette(color_list, plot=False):
    pal = sns.color_palette(color_list)
    if plot: sns.palplot(pal)
    return pal

def hex_to_rgb(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return np.array([int(hex_str[i:i+2], 16) for i in range(1,6,2)])

def convert_to_rgb(c):
    if c[0] == "#": # hex
        return hex_to_rgb(c)
    elif isinstance(c, str): # string word
        return np.array(mcolors.to_rgb(c))
    else: # already rgb
        return np.asarray(c)

def get_color_gradient(c1, c2, n=10):
    """ Given two colors, returns a color gradient with n colors """
    assert n > 1, "n must be greater than 1"
    c1_rgb = convert_to_rgb(c1)
    c2_rgb = convert_to_rgb(c2)
    mix_pcts   = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def make_cmap(colors):
    return sns.color_palette(colors, as_cmap=True)


#----------------------------------------------------------------------------------------------
# saving 
#----------------------------------------------------------------------------------------------


def save_figure(fig, fname, formats=None):
    if isinstance(formats, str): 
        formats = [formats]
    elif formats is None: formats = ['png']
    formats = [format[1:] if format.startswith('.') else format for format in formats]
    if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.svg'):
        fname = fname[:-4]
    for format in formats:
        fig.savefig(f'{fname}.{format}',format=format, bbox_inches='tight', dpi=1200)

def save_figures_pdf(figs, fname, dpi=300):
    # will write multiple figures to a pdf and save it
    pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
    for fig in figs:  # save the figure to file
        pdf.savefig(fig)
    pdf.close()


#----------------------------------------------------------------------------------------------
# structured subplots
#----------------------------------------------------------------------------------------------


def create_subplots(grid_size, irregular_axes=None, figsize=(10,10), annotate=False):
    """
    Create a grid of subplots using matplotlib.

    Args:
        grid_size (tuple): The size of the grid in (rows, columns).
        irregular_axes (dict, optional): A dictionary of irregular axes positions and shapes. 
            The keys are the positions in the grid (row, column) and the values are the shapes (rows, columns).
        figsize (tuple, optional): The size of the figure in inches. Defaults to (10, 10).
        annotate (bool, optional): Whether to annotate the subplots with their index. Defaults to False.

    Returns:
        tuple: A tuple containing the created figure and a list of axes objects.

    Examples:
        >>> grid_size = (3, 3)
        >>> fig, axs = create_subplots(grid_size)
        >>> len(axs)
        9
    """
    fig = plt.figure(figsize=figsize)

    # initialize a grid to keep track of occupied cells
    grid_occupancy = [[False]*grid_size[1] for _ in range(grid_size[0])]

    axs = []

    # make any irregular axes
    if irregular_axes is not None:
        for idx, shape in irregular_axes.items():
            axs.append(plt.subplot2grid(grid_size, idx, colspan=shape[1], rowspan=shape[0]))
            # Mark the cells as occupied
            for i, j in itertools.product(range(shape[0]), range(shape[1])):
                if idx[0]+i < grid_size[0] and idx[1]+j < grid_size[1]:
                    grid_occupancy[idx[0]+i][idx[1]+j] = True 

    # create 1x1 subplots in the remaining cells
    for i, j in itertools.product(range(grid_size[0]), range(grid_size[1])):
        if not grid_occupancy[i][j]:
            axs.append(plt.subplot2grid(grid_size, (i, j)))

    # add text to each ax 
    if annotate: 
        for i, ax in enumerate(axs):
            ax.text(0.5, 0.5, f'axs[{i}]', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    return fig, axs


#----------------------------------------------------------------------------------------------
# generic plots
#----------------------------------------------------------------------------------------------


def plot_scatter(x, y, ax=None, color=None, **kwargs):
    if ax is None: ax = plt.gca()
    if color is None: color = 'grey'
    sns.scatterplot(x=x, y=y, color=color, size=75, edgecolor='black', 
                    linewidth=1, alpha=0.75, ax=ax)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    # remove legend
    ax.get_legend().remove()
    return ax

def plot_regplot(x, y, ax=None, color=None, **kwargs):
    if ax is None: ax = plt.gca()
    if color is None: color = 'grey'
    sns.regplot(x=x, y=y, color=color, 
                scatter_kws={'s': 75, 'edgecolor': ec, 'linewidth': lw, 'alpha': .75}, 
                ax=ax, **kwargs)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    return ax

def plot_barplot(x, y, ax=None, color=None, pal=None, **kwargs):
    if ax is None: ax = plt.gca()
    sns.barplot(x=x, y=y, color=color, palette=pal, 
                edgecolor='black', 
                ax=ax, **kwargs)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    return ax

def plot_histplot(vals, ax=None, color=None, kde=True, **kwargs):
    if ax is None: ax = plt.gca()
    if color is None: color = 'grey'
    sns.histplot(vals, ax=ax, 
                 kde=kde,
                 color=color,
                 edgecolor=ec, linewidth=1.5, **kwargs)
    return ax

def plot_kdeplot(vals, ax=None, color=None, **kwargs):
    if color is None: color = 'grey'
    if ax is None: ax = plt.gca()
    sns.kdeplot(vals, ax=ax, color=color,
                fill=True, alpha=.5, edgecolor=ec, linewidth=lw+.05, **kwargs)
    return ax   

def plot_rdm(rdm, digitize=False, lower_tri=False,
             ax=None, outline=False, 
             cmap='Purples', 
             cbar=None, cbar_title='Distance', 
             vmin=None, vmax=None, figsize=(10,10)):
    
    if digitize: 
        rdm = digitize_matrix(rdm)

    if lower_tri:
        mask = np.zeros_like(rdm)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = False

    with sns.axes_style("white"):

        lw = 1 if outline else 0
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(rdm, mask=mask, 
                    vmin=vmin, vmax=vmax, 
                    square=True, 
                    xticklabels=False, yticklabels=False,
                    cbar=False, 
                    cmap=cmap,                  
                    linewidths=lw, linecolor='black',
                    ax=ax)
        if outline:
            if lower_tri:
                for x in range(1, rdm.shape[0]):
                    ax.vlines(x, x-.1, x+1, linewidth=1, color='black')
                    ax.hlines(x, x-1.12, x, linewidth=1, color='black')
                ax.vlines(0, 1, rdm.shape[0], linewidth=1, color='black')
                ax.hlines(rdm.shape[0], 1, rdm.shape[0]-1, linewidth=2, color='black')
            else:
                ax.vlines(0, 0, rdm.shape[0], linewidth=3, color='black')
                ax.vlines(rdm.shape[0], 0, rdm.shape[0], linewidth=3, color='black')
                ax.hlines(0, 0, rdm.shape[0], linewidth=3, color='black')
                ax.hlines(rdm.shape[0], 0, rdm.shape[0], linewidth=3, color='black')            
        if cbar:
            fig = ax.get_figure()
            cax  = fig.add_axes([.9, 0.14, 0.01, 0.33])
            cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='vertical')
            cbar.set_ticks([12, 98])
            cbar.ax.tick_params(size=0)
            cbar.set_ticklabels(["Close", "Far"], fontsize=tick_fontsize)
            if cbar_title is not None:
                cbar.ax.set_ylabel(cbar_title, fontsize=label_fontsize, rotation=270, labelpad=0)
            cbar.outline.set_edgecolor('black')
        
    return ax

def plot_lower_tri(rdm, outline=False, cbar=None, vmax=100, figsize=(10,5), digitize=True):
    
    if digitize: 
        rdm = digitize_matrix(rdm)

    # to get rid of upper half & diagonal
    mask = np.zeros_like(rdm)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):

        fig, ax = plt.subplots(figsize=figsize)
        sns.set(font_scale=1.5)
        ax = sns.heatmap(rdm, mask=mask, vmax=100, 
                            xticklabels=False, yticklabels=False,
                            cbar=cbar,
                            cmap="Purples",
                            cbar_kws={'orientation': 'horizontal',
                                    'shrink': .35},
                            square=True)
        if outline:
            for x in range(1,63):
                ax.vlines(x, x-.1, x+1, linewidth=.5, color='black')
                ax.hlines(x, x-1.12, x, linewidth=.5, color='black')
            ax.vlines(0, 1, 63, linewidth=.5, color='black')
            ax.hlines(63, 1, 62, linewidth=.5, color='black')
        if cbar:
        #     ax.collections[0].colorbar.set_label("Distance")
            ax.collections[0].colorbar.set_ticks([10,100])
            ax.collections[0].colorbar.set_ticklabels(["Close", "Far"])
            # [attr for attr in dir(ax.collections[0].colorbar)] # show methods
            
        plt.show()
        
    return fig


#----------------------------------------------------------------------------------------------
# add things to existing plots
#----------------------------------------------------------------------------------------------

def annotate_plot(plot, labels):
    """
        This function takes a scatter plot object and a list of labels,
        and annotates the scatter plot points with the labels using mplcursors.
        
        Parameters:
        scatter: The scatter plot object to annotate.
        labels: A list of labels corresponding to the scatter plot points.
        """
    cursor = mplcursors.cursor(plot, hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        # Define how each annotation should look
        sel.annotation.set_text(labels[sel.target.index])
        sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.5)  # Set annotation's background color
        sel.annotation.set_fontsize(12)

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def add_subfig_letter(ax, letter, x=-0.05, y=1.1, fontsize=12, fontweight='bold'):
    ax.text(x, y, letter, transform=ax.transAxes, 
            fontsize=fontsize, fontweight=fontweight, va='top', ha='right')
    
def plot_significance(ax, pvalue, sig_level=4, color=None, x=0.0, y=0.98, dx=0.015, fontsize=17):
    alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    if color is None: color = 'k'
    for i, alpha in enumerate(alphas[:sig_level]):
        if pvalue < alpha:
            ax.text(x+dx*i, y, "*", color=color, fontsize=fontsize, 
                    ha="left", va="top", transform=ax.transAxes)  

def add_sig_stars(x, y, df, demo_controls, ax, color):
    _, ols_obj = run_ols([x], y, df, covariates=demo_controls, n_splits=None, plot=False)
    x_change = 0.04    
    sigs = [0.05,0.01,0.005,0.001,0.0005,0.0001]
    for s, sig in enumerate(sigs):
        if ols_obj.pvalues[x] < sig:
            ax.text(0.225-x_change*s, 0.98, "*", color=color, fontsize=15, ha="left", va="top", transform=ax.transAxes)  

def plot_space_density(x, y, figsize=(5,5), ax=None, regression=False):
    
    '''
    '''

    # calculate the point density
    xy = np.vstack([x, y])
    z  = gaussian_kde(xy)(xy) 

    # sort so that the bright spots are on top
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # plot
    fig, axs = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'hspace': 0, 'wspace': 0,
                                                                'width_ratios': [5, 1], 'height_ratios': [1, 5]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")

    axs[1,0].set_ylim([-1,1])
    axs[1,0].set_yticks([-1,-.5,0,.5,1])
    axs[1,0].set_xlim([-1,1])
    axs[1,0].set_xticks([-1,-.5,0,.5,1])
    axs[1,0].axhline(y=0, color='black', linestyle='-', zorder=-1)
    axs[1,0].axvline(x=0, color='black', linestyle='-', zorder=-1)
    axs[1,0].set_xlabel('Affiliation')
    axs[1,0].set_ylabel('Power')

    # plot distributions on the sides
    sns.distplot(x, bins=20, ax=axs[0,0], color="Black")
    axs[0,0].set_xlim([-1,1])
    sns.distplot(y, bins=20, ax=axs[1,1], color="Black", vertical=True)
    axs[1,1].set_ylim([-1,1])

    # plot density
    axs[1,0].scatter(x, y, c=z, s=100)

    # plot regression
    if regression: sns.regplot(x, y, scatter=False, color='black', ax=axs[1,0])
    
    plt.show()

def plot_ols_models_metrics(results_df, metric='train_BIC', colors=None):
    
    if colors is None: colors = []
    # assign colors
    models = np.unique(results_df['model'])
    n_models = len(models)
    n_folds = list(results_df['model']).count(models[0])
    results_df['color'] = np.repeat(np.arange(n_models), n_folds, 0)

    if len(colors) == 0:
        colors = random_colors(num_colors=n_models)
        # color = iter(plt.cm.cool(np.linspace(0, 1, n_models))) # need more different colors
        # colors = []
        # for i in range(n_models): colors.append(list(next(color)))

    metric_df = results_df[results_df['metric'] == metric]
    mean_df = metric_df[metric_df['fold'] == 'mean']

    if 'IC' in metric: 
        ascending = True
    else:
        ascending = False
    mean_df.sort_values('value', ascending=ascending, inplace=True)
    ordered_models = mean_df['model'].values
    ordered_colors = mean_df['color'].values

    fig, ax = plt.subplots(figsize=(15,7))
    ax = sns.barplot(x='model', y='value', hue='fold', data=metric_df, order=ordered_models)
    for i, bar in enumerate(ax.patches):
        c = i % n_models
        bar.set_color(colors[ordered_colors[c]])
        bar.set_edgecolor(".2")
    ax.set_ylabel(metric, fontsize=20)
    ax.set_ylim(np.min(metric_df['value'])-np.min(metric_df['value'])*.05, 
                np.max(metric_df['value'])+np.max(metric_df['value'])*.05)
    ax.set_xticklabels([x for x in ordered_models], rotation=60, ha='right')
    plt.legend([],[], frameon=False)
    plt.show()


#----------------------------------------------------------------------------------------------
# create a nice annotated correlation matrix
#----------------------------------------------------------------------------------------------


def corrfunc(x, y, xy=(0.05, 0.9), corr=scipy.stats.pearsonr, **kws):
    coef, p = corr(x, y)
    if p <= 0.001:  p_stars = '***'
    elif p <= 0.01: p_stars = '**'
    elif p <= 0.05: p_stars = '*'
    else          : p_stars = ''
    ax = plt.gca()
    ax.annotate('coef = {:.2f} '.format(coef) + p_stars,
                xy=xy, xycoords=ax.transAxes)

def annotate_colname(x, **kws):
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes, fontweight='bold')

def corr_matrix(df):
    g = sns.PairGrid(df, palette=['red'])    
    g.map_upper(sns.regplot, scatter_kws={'s':10})
    g.map_diag(sns.distplot)
    g.map_diag(annotate_colname)
    g.map_lower(sns.kdeplot, cmap='Blues_d')
    g.map_lower(corrfunc)
    #     # Remove axis labels, as they're in the diagonals.
    #     for ax in g.axes.flatten():
    #         ax.set_ylabel('')
    #         ax.set_xlabel('')
    return g

def corr_mat_plot(df, size = 10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    # Compute the correlation matrix for the received dataframe
    corr_mat = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    ax.grid(False)
    cax = ax.matshow(corr_mat, cmap='jet', vmin=-1, vmax=1)
    plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=90)
    plt.yticks(range(len(corr_mat.columns)), corr_mat.columns)
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks = [-1, 0, 1], aspect = 40, shrink = .8)
    
    return corr_mat

def filter_sig_correlations(corr_mat, p_mat, thresh=0.00001):
    '''
    '''
    # output the relationships with significant pvalues from a correlation matrix

    # turn matrices in flattened upper tri 
    pval_ut = symm_mat_to_ut_vec(p_mat.values)
    corr_ut = symm_mat_to_ut_vec(corr_mat.values)

    # annoying way to get the upper tri comparisons
    ut_indices = np.triu_indices_from(corr_mat,1)
    ut_indices = np.array(ut_indices).reshape(2,len(ut_indices[0])).T
    corr_names = np.array([[corr_mat.index[i], corr_mat.columns[j]] for i, j in ut_indices])

    # only print out the significant relationship
    p_mask = symm_mat_to_ut_vec((p_mat < thresh).values)

    filtered_corrs = [('corr('+c[0]+','+c[1]+')', 'r='+str(np.round(r,4)), 'p='+str(np.round(p,5))) 
                         for c, r, p 
                         in zip(corr_names[p_mask], corr_ut[p_mask], pval_ut[p_mask])]
    return filtered_corrs

def get_asymmetrical_corr_mat(x_df, y_df):
    '''
        x_df & y_df should dfs of shape num_subs x num_variables
        any angular data should be in degrees & have 'angles' in col name
        maybe: for egocentric angles just do cos(angles_rads) and then run whatever correlation...?
    '''
    y_names, x_names = y_df.columns, x_df.columns
    coef_mat, pval_mat = np.zeros((len(y_names), len(x_names))), np.zeros((len(y_names), len(x_names)))
    for y,y_name in enumerate(y_names):
        for x,x_name in enumerate(x_names):
            finite_mask = np.isfinite(x_df[x_name].values) & np.isfinite(y_df[y_name].values)
            if ('angles' not in x_name) & ('angles' not in y_name):
                coef_mat[y,x], pval_mat[y,x] = scipy.stats.pearsonr(x_df[x_name][finite_mask], y_df[y_name][finite_mask])
            elif ('angles' in x_name) & ('angles' not in y_name):
                coef_mat[y,x], pval_mat[y,x] = circ_corrcl(x_df[x_name][finite_mask], y_df[y_name][finite_mask])
            elif ('angles' not in x_name) & ('angles' in y_name):
                coef_mat[y,x], pval_mat[y,x] = circ_corrcl(y_df[y_name][finite_mask], x_df[x_name][finite_mask])
            elif ('angles' in x_name) & ('angles' in y_name):
                coef_mat[y,x], pval_mat[y,x] = circ_corrcc(y_df[y_name][finite_mask], x_df[x_name][finite_mask])
    coef_df = pd.DataFrame(coef_mat, index=y_names, columns=x_names).astype(float)
    pval_df = pd.DataFrame(pval_mat, index=y_names, columns=x_names).astype(float)
    return coef_df, pval_df

def plot_asymmetrical_corrmat(coef_df, pval_df, alpha=.05):
    '''
    '''
    x_names  = coef_df.columns
    y_names  = coef_df.index
    coef_mat = coef_df.values
    pval_mat = pval_df.values
    
    # masked correlation plot
    masked_corrs = coef_mat * ((pval_mat < alpha) * 1)
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(masked_corrs, cmap="RdBu", vmax=np.max(masked_corrs), vmin=-np.max(masked_corrs))
    
    for i in np.arange(len(y_names)):
        for j in np.arange(len(x_names)):
            if pval_mat[i,j] < alpha:
                ax.text(j, i, np.round(pval_mat[i,j], 7),
                        ha="center", va="center", color="w", fontsize=5)
                
    im_ratio = coef_mat.shape[0]/coef_mat.shape[1] # (height_of_image / width_of_image)
    # cbar = plt.colorbar(ax.get_children()[-2], ax=ax, label='correlation coefficient',
    #                     orientation='vertical', fraction=0.044*im_ratio)
    # cbar.ax.tick_params(labelsize=7.5)

    ax.set_yticks(np.arange(len(y_names)))
    ax.set_yticklabels(y_names, fontsize=11)
    ax.set_xticks(np.arange(len(x_names)))
    ax.set_xticklabels(x_names, fontsize=11, rotation=90)
    ax.set_title('Correlations (p<'+str(alpha)+' uncorr)', fontsize=15)
    
    plt.show()
    
    return fig
