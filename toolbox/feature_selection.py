import numpy as np
from scipy.cluster import hierarchy
from scipy import stats
from matplotlib import pyplot as plt

def cluster_correlated_features(X):
    
    '''
        ideally would want dendrogram added to heatmap
    '''

    # get correlations and clustering
    corr_mat = stats.spearmanr(X)[0]
    corr_linkage = hierarchy.ward(corr_mat)

    # plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35,20))

    # dendrogram
    dendro = hierarchy.dendrogram(corr_linkage,  labels=X.columns.tolist(),
                                  ax=ax1, leaf_rotation=90, leaf_font_size=20)

    # heatmap
    dendro_idx = np.arange(0, len(dendro['ivl']))
    hm = ax2.imshow(corr_mat[dendro['leaves'], :][:, dendro['leaves']], cmap="RdBu_r")
    cbar = plt.colorbar(hm)
    cbar.set_label("Correlation coefficient", fontsize=15)

    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical', size=20)
    ax2.set_yticklabels(dendro['ivl'], size=20)
    ax2.set_xticks(np.arange(-.5, len(dendro_idx), 1), minor=True)
    ax2.set_yticks(np.arange(-.5, len(dendro_idx), 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=2)

    fig.tight_layout()
    plt.show()
    
    return dendro['ivl']