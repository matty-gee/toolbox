import warnings
warnings.filterwarnings('ignore')

import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

#--------------------------------------------------------------------
# plotting standards
#--------------------------------------------------------------------


tab10_colors = sns.color_palette("tab10").as_hex()
general_pal = [tab10_colors[0], tab10_colors[2], tab10_colors[4], tab10_colors[5], tab10_colors[7]]
roi_pal, distance_pal = sns.color_palette("Paired"), sns.color_palette("Purples")

alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] # for stars
tick_fontsize, label_fontsize, title_fontsize = 10, 13, 15
legend_title_fontsize, legend_label_fontsize = 12, 10
suptitle_fontsize = title_fontsize * 1.5
ec, lw = 'black', 1
bw = 0.15 # this is a proportion of the total??
figsize, facet_figsize = (5, 5), (5, 7.5)

affil_color = "#4374B3"
power_color = "#FF0B04"


#--------------------------------------------------------------------
# task info 
#--------------------------------------------------------------------


sys.path.insert(0, f"{os.path.expanduser('~')}/Dropbox/Projects/social_navigation_analysis/social_navigation_analysis")
from info import decision_trials, character_roles
trial_roles      = decision_trials['char_role'].values
character_labels = decision_trials.sort_values(by=['slide_num'])['char_role_num'].values
character_colors = ['orange', 'red', 'blue', 'pink', 'yellow', 'grey']
character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss', 'neutral']
character_colors_trials = [character_colors[character_roles.index(role.lower())] for role in trial_roles]


#--------------------------------------------------------------------
# data wrangling
#--------------------------------------------------------------------


def reshape_dataframe(df, cols):
    """
    Takes a DataFrame and a list of lists of column names.
    Returns a 3D array with shape (p, n, m), where p is the number of rows in the DataFrame,
    n is the number of lists, and m is the length of each list.
    """
    if not all(isinstance(item, list) for item in cols):
        raise ValueError("All items in cols must be lists.")
    array_3d = []
    for col_list in cols:
        selected_cols = df[col_list]
        array_3d.append(selected_cols.values)
    return np.array(array_3d).transpose(1, 0, 2)

def get_coords(df, which='task', include_neutral=False):
    
    if include_neutral: roles = character_roles
    else: roles = [role for role in character_roles if role != 'neutral']
    if which == 'task':
        return reshape_dataframe(df, [[f'affil_mean_{role}', f'power_mean_{role}'] for role in roles])
    elif which == 'dots':
        return reshape_dataframe(df, [[f'{role}_dots_affil', f'{role}_dots_power'] for role in roles])

def mask_trials_out(trial_ixs):
    '''
        flexible masking for trialwise matrices...
    '''
    mask_rdm = np.ones((63,63))
    for ix in trial_ixs:
        mask_rdm[ix,:] = 0
        mask_rdm[:,ix] = 0
    return symm_mat_to_ut_vec(mask_rdm)

def remove_neutrals(arr):
    '''remove the neutral trials from array: trials 15,16,36
    '''
    return np.delete(arr, np.array([14,15,35]), axis=0) 

def add_neutrals(arr, add=None):
    '''
        add values for neutral trials into array; trials 15,16,36
        inputted array should have length 60; outputted will have length 63
    '''
    
    if add is None:
        add = [0,0]
    temp = arr.copy()
    for row in [14,15,36]:
        temp = np.insert(temp, row, add, axis= 0)
    return temp

def organize_by_character(data, remove_neutrals=False):

    if data.shape[0] == 63:
        trialwise_char_roles = decision_trials['char_role_num'].values
        if remove_neutrals:
            n_chars = 5 
        else: 
            n_chars = 6
    elif data.shape[0] == 60:
        trialwise_char_roles = remove_neutrals(decision_trials['char_role_num'].values)
        n_chars = 5
    if isinstance(data, np.ndarray):
        if data.ndim == 1: 
            return [data[trialwise_char_roles == c] for c in range(1,n_chars+1)]
        else: 
            return [data[trialwise_char_roles == c,:] for c in range(1,n_chars+1)]
    elif isinstance(data, pd.DataFrame):
        return [data[trialwise_char_roles == c] for c in range(1, n_chars+1)]


#--------------------------------------------------------------------
# other helpers
#--------------------------------------------------------------------


def character_cv(n_trials=63):
    if n_trials == 60:
        roles = remove_neutrals(character_labels)
    elif n_trials == 63:
        roles = character_labels
    train_ixs, test_ixs = [], []
    for i in range(1,6):
        train_ixs.append(np.where(roles != i)[0])
        test_ixs.append(np.where(roles == i)[0])
    return zip(train_ixs, test_ixs)

def get_char_rdm(betas):

    char_rdm = np.zeros((5,5))

    for i in np.arange(5):

        char1_ixs   = np.where(decision_trials['role'].values == character_roles[i])[0]
        char1_betas = betas[char1_ixs, :]

        for j in np.arange(5):

            # get average correlation distance between character roles' betas
            char2_ixs   = np.where(decision_trials['role'].values == character_roles[j])[0]
            char2_betas = betas[char2_ixs, :]

            assert len(char1_betas) == 12, f'Expected 12 trials for {character_roles[i]}, got {len(char1_betas)}'
            assert len(char2_betas) == 12, f'Expected 12 trials for {character_roles[j]}, got {len(char2_betas)}'

            corr = []
            for ii in np.arange(len(char1_betas)):
                for jj in np.arange(len(char2_betas)):
                    corr.append(scipy.stats.pearsonr(char1_betas[ii,:], char2_betas[jj,:])[0])
            corr_dist = 1 - np.mean(corr)
            char_rdm[i,j] = corr_dist
            char_rdm[j,i] = corr_dist
            
    return char_rdm

def make_random_2d_locations(n): 
    # generate a 2xn array of random numbers between 0 and 1 uniformly distributed
    return np.random.rand(2, n)


#--------------------------------------------------------------------
# plotting
#--------------------------------------------------------------------


def add_dot(ax, xy=(0,0), color='black', s=None):
    # put a scatter point at the origin
    if s is None: s = ax.get_xlim()[1] * 0.05
    ax.scatter(xy[0], xy[1], color=color, s=s, zorder=10)
    return ax

def add_pov(ax, s=None, color='red'):
    if s is None:
        s = ax.get_xlim()[1] * 1000
    ax.scatter(1.05, 0, color=color, s=s, zorder=1)

def plot_social_space(ax=None, 
                      label_fontsize=None,
                      reverse_power=False,
                      linewidth=2.5,
                      figsize=(5,5),
                      origin_size=100):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False

    # scale font sizes with size of figure
    if label_fontsize is None: # try to infer... doesnt work with multipl axdes
        figsize  = plt.gcf().get_size_inches()
        label_fontsize = int(figsize[0] * 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Affiliation', fontsize=label_fontsize)
    ax.set_ylabel('Power',  fontsize=label_fontsize)

    # clean up
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    
    ax.set_xticklabels(['Low','Neutral','High'], fontsize=label_fontsize/1.5)
    if reverse_power:
        ax.set_yticklabels(['High','Neutral','Low'], rotation=90, va='center', fontsize=label_fontsize/1.5)
    else:
        ax.set_yticklabels(['Low','Neutral','High'], rotation=90, va='center', fontsize=label_fontsize/1.5)

    # remove ticks
    ax.tick_params(axis='both', which='both', length=0)

    # add grid
    for v in [1.05, -1.05]:
        # get the ax size 
        ax.axhline(v, color='k', linestyle='-', linewidth=linewidth, alpha=1, zorder=0)
        ax.axvline(v, color='k', linestyle='-', linewidth=linewidth, alpha=1, zorder=0)
    
    ax.axhline(0, color='k', linestyle='-', linewidth=linewidth*.4, alpha=0.75, zorder=0)
    ax.axvline(0, color='k', linestyle='-', linewidth=linewidth*.4, alpha=0.75, zorder=0)
    
    for v in [.5, -.5]:
        ax.axhline(v, color='k', linestyle='--', linewidth=linewidth*.4, alpha=0.25, zorder=0)
        ax.axvline(v, color='k', linestyle='--', linewidth=linewidth*.4, alpha=0.25, zorder=0)

    # add unit/basis vectors
    ax.arrow(-1, -1, 0, 1/6, head_width=0.05, head_length=0.05, fc='k', ec='k', zorder=-1, alpha=1)
    ax.arrow(-1, -1, 1/6, 0, head_width=0.05, head_length=0.05, fc='k', ec='k', zorder=-1, alpha=1)

    if origin_size is not None:
        add_dot(ax, color='black', s=origin_size)

    return (fig, ax) if return_fig else ax