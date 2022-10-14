import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from pylab import *
sns.set(style='white')

from scipy import stats

########################################################
# palettes etc
########################################################

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

########################################################
# should it be here or in an rsa folder? 
# or rename it and make it more general?
########################################################

def plot_rdm(rdm, lower_tri=False, v=(0,100), size=10, cbar=None, cbar_labels=[]):
    '''
        inputs:
            rdm: representational dissimilarity matrix 
            lower_tri: boolean to mask for upper triangle
            cbar: if True, plots cbar 
            vmax: defaults to 100, expecting a digitized rdm
            size: side size of square plot
    '''
    # to get rid of upper half & diagonal
    if lower_tri: 
        mask = np.zeros_like(rdm)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = False
        
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(size,size))
        # sns.set(font_scale=1.5)
        ax = sns.heatmap(rdm, vmin=v[0], vmax=v[1], mask=mask,
                         xticklabels=False, yticklabels=False,
                         cmap="Purples",
                         cbar=cbar,
                         cbar_kws={'orientation': 'horizontal','shrink': .68},
                         linewidths=2, linecolor='black',
                         square=True)
        if cbar:
            ax.collections[0].colorbar.outline.set(visible=True, lw=1, edgecolor="black")
            ax.collections[0].colorbar.set_ticks([0,100])
            if len(cbar_labels) == 0:
                ax.collections[0].colorbar.set_ticklabels(["Similar", "Dissimilar"])
            else:
                ax.collections[0].colorbar.set_ticklabels([cbar_labels[0], cbar_labels[1]])
            # [attr for attr in dir(ax.collections[0].colorbar)] # show methods
        plt.show()
    return fig

########################################################
# create an annotated correlation matrix - TODO: improve
########################################################

def corrfunc(x, y, xy=(0.05, 0.9), corr=stats.pearsonr, **kws):
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

