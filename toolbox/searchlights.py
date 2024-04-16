#!/usr/bin/env python

import os, sys, glob, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import nilearn
from nilearn import image, plotting
from nilearn.glm import cluster_level_inference, threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference

# custom code
base_dir = '/sc/arion/projects/OlfMem/mgs/2D_place'
sys.path.insert(0, f'{base_dir}/Code/toolbox') 
sys.path.insert(0, f'{base_dir}/Code/social_navigation_analysis/social_navigation_analysis') 
from utils_project import *
from mvpa import *
from info import decision_trials, character_roles, task
from general_utils import merge_dfs, symm_mat_to_ut_vec, ut_vec_to_symm_mat
from trajectories import *
from socialspace import remove_neutrals, organize_by_character


#---------------------------------------------------------------------------------------------------------
# image functions
#---------------------------------------------------------------------------------------------------------


def flip_sign_img(img):
    return image.math_img("-img", img=img)

def center_percent_img(img, chance=0.50):
    return image.math_img(f"img - {chance}", img=img)


#---------------------------------------------------------------------------------------------------------
# 1st level searchlight functions
#---------------------------------------------------------------------------------------------------------

# dimension ps diff. searchlight
def run_dimension_sl(func_fname, func_mask_fname, method='rsa'):

    # simple affiliation v. power classification
    np.random.seed(33)

    # set up
    out_dir = f"{('/').join(func_fname.split('/')[:-3])}/searchlights"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    sub_id = func_fname.split('/')[-2]

    # prepare labels     
    dims = remove_neutrals((decision_trials['dimension'] == 'affil').values.astype(int))
    dims_rdv = symm_mat_to_ut_vec(pairwise_distances(dims[:, np.newaxis], metric='hamming'))
    if method == 'rsa':
        sl_f = sl_rsa
        bcvar = dims_rdv
    elif method == 'svc':
        sl_f = sl_svc
        bcvar = dims
    elif method == 'svc_gs':
        sl_f = sl_svc_gridsearch
        bcvar = dims

    # run searchlight
    sl = Searchlight(sl_f=sl_f, shape='ball', radius=3, min_prop=0.10, num_sls=10)
    sl.prepare(func_fname, func_masks=func_mask_fname, bcvar=bcvar)    
    sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_dimension-{method}')

def run_ps_sl(func_fname, func_mask_fname, model='character', radius=3):

    np.random.seed(33)

    # set up
    out_dir = f"{('/').join(func_fname.split('/')[:-3])}/searchlights"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    sub_id = func_fname.split('/')[-2]

    # prepare labels  
    if model == 'character':  

        # character pattern similarity: (mean ps within character) - (mean ps between characters)
        chars     = remove_neutrals(decision_trials['char_role_num'].values.astype(int))[:, np.newaxis]
        model_rdm = pairwise_distances(chars, metric='hamming')

    elif model == 'dimension':

        # dimension pattern similarity: (mean ps within dimension) - (mean ps between dimensions)
        affil     = remove_neutrals((decision_trials['dimension'] == 'affil').values.astype(int))[:, np.newaxis]
        model_rdm = pairwise_distances(affil, metric='hamming')

    model_rdv = symm_mat_to_ut_vec(model_rdm)
    model_rsv = 1 - model_rdv # similarity
    model_rsm = ut_vec_to_symm_mat(model_rsv)
    assert np.sum(model_rdv == 0) == np.sum(model_rsv), 'similarity & dissimilarity vectors do not match'

    # fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    # sns.heatmap(dims_rdm, cmap='Greys', square=True, cbar=False, ax=axs[0])
    # sns.heatmap(dims_rsm, cmap='Greys', square=True, cbar=False, ax=axs[1])
    # plt.tight_layout()

    # run searchlight: pass in the similarity vector
    sl = Searchlight(sl_f=sl_ps, shape='ball', radius=radius, min_prop=0.10, num_sls=10)
    sl.prepare(func_fname, func_masks=func_mask_fname, bcvar=model_rsv)    
    sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_{model}-ps_cube{radius}mm')



    # elif model == 'dimension_diff':

    #     # this needs more... gotta change the sl_ps function...

    #     #--------------------------------------------------------------------------------------------------------------------
    #     # dimension pattern similarity difference: (mean ps within 1st dimension) - (mean ps within 2nd dimension)
    #     #--------------------------------------------------------------------------------------------------------------------

    #     affil = remove_neutrals((decision_trials['dimension'] == 'affil').values.astype(int))[:, np.newaxis]
    #     affil_rsm, power_rsm = np.zeros((60,60)), np.zeros((60,60))
    #     for i, j in list(itertools.combinations(np.where(affil == 1)[0], 2)):
    #         affil_rsm[i, j], affil_rsm[j, i] = 1, 1
    #     for i, j in list(itertools.combinations(np.where(affil == 0)[0], 2)):
    #         power_rsm[i, j], power_rsm[j, i] = 1, 1

    #     affil_rsv = symm_mat_to_ut_vec(affil_rsm)
    #     power_rsv = symm_mat_to_ut_vec(power_rsm)
    #     assert (np.sum(affil_rsv == 1) + np.sum(power_rsv == 1)) == 870, 'masks do not equal 870'

    #     # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     # sns.heatmap(affil_rsm, ax=axs[0], cmap='Greys', cbar=False)
    #     # sns.heatmap(power_rsm, ax=axs[1], cmap='Greys', cbar=False)
    #     # plt.tight_layout()


# character identity ps diff. searchlight
def get_character_rdv(c):
    # checkin out: # plot_rdm(ut_vec_to_symm_mat(get_character_rdv(1)))
    char_rdm = np.ones((60, 60))
    char_trials = remove_neutrals(decision_trials['char_role_num'].values)
    trial_nums = np.where(char_trials == c)[0]
    for r in trial_nums:
        for c in trial_nums:
            char_rdm[c, r] = 0 
    return symm_mat_to_ut_vec(char_rdm)

def sl_character(brain_data, sl_mask, myrad, bcvar):
    '''
        Run representational similarity analysis in a brainiak searchlight 

        Arguments
        ---------
        brain_data : matrix
        sl_mask : matrix
            Boolean, mask for voxels to run searchlight over
        myrad : float
            Radius of searchlight [not sure if this is necessary]
        bcvar : list, dictionary (optional)
            Another variable to use with searchlight kernel
            For ex: condition labels so that you can determine 
            to which condition each 3D volume corresponds

        Returns
        -------
        float 
            Coefficient

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    # get searchlight volume & representational similarity vector
    neural_sl  = get_sl_data(brain_data[0], sl_mask)
    neural_sl  = remove_neutrals(neural_sl)
    neural_rsv = symm_mat_to_ut_vec(1 - get_corrdist_matrix(neural_sl))

    # get ps diff for each character
    ps_diffs = []
    for c in range(1,6):
        pred_rsv = 1 - get_character_rdv(c)
        ps_diffs.append(np.mean(neural_rsv[pred_rsv==1]) - np.mean(neural_rsv[pred_rsv==0])) # should be positive
    return np.mean(ps_diffs)

def run_character_sl(func_fname, func_mask_fname, radius=3):

    np.random.seed(33)

    # set up
    out_dir = f"{('/').join(func_fname.split('/')[:-3])}/searchlights"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    sub_id = func_fname.split('/')[-2]

    # run searchlight: pass in the similarity vector
    sl = Searchlight(sl_f=sl_character, shape='ball', radius=radius, min_prop=0.10, num_sls=10)
    sl.prepare(func_fname, func_masks=func_mask_fname, bcvar=None)    
    sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_character-ps_cube{radius}mm')


# familiarity rsa searchlight
def sl_familiarity(brain_data, sl_mask, myrad, bcvar):
    '''
        Run representational similarity analysis in a brainiak searchlight 

        Arguments
        ---------
        brain_data : matrix
        sl_mask : matrix
            Boolean, mask for voxels to run searchlight over
        myrad : float
            Radius of searchlight [not sure if this is necessary]
        bcvar : list, dictionary (optional)
            Another variable to use with searchlight kernel
            For ex: condition labels so that you can determine 
            to which condition each 3D volume corresponds

        Returns
        -------
        float 
            Coefficient

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    # get neural rdv from searchlight volume
    neural_sl = get_sl_data(brain_data[0], sl_mask)
    neural_sl = remove_neutrals(neural_sl)
    neural_rdv = symm_mat_to_ut_vec(get_corrdist_matrix(neural_sl))

    # get familiarity rdv
    familiarity_rdm = pairwise_distances(remove_neutrals(decision_trials['char_decision_num'].values[:,np.newaxis]))
    familiarity_rdv = symm_mat_to_ut_vec(familiarity_rdm)
    # plot_rdm(familiarity_rdm[:,np.newaxis])))

    return scipy.stats.kendalltau(familiarity_rdv, neural_rdv)[0]

def run_familiarity_sl(func_fname, func_mask_fname, radius=3):

    np.random.seed(33)

    # set up
    out_dir = f"{('/').join(func_fname.split('/')[:-3])}/searchlights"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    sub_id = func_fname.split('/')[-2]

    # run searchlight: pass in the similarity vector
    sl = Searchlight(sl_f=sl_familiarity, shape='ball', radius=radius, min_prop=0.10, num_sls=10)
    sl.prepare(func_fname, func_masks=func_mask_fname, bcvar=None)    
    sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_familiarity_cube{radius}mm')


# relationship trajectory rsa searchlight
def sl_trajsim(brain_data, sl_mask, myrad, bcvar):
    '''
        Run representational similarity analysis in a brainiak searchlight 

        Arguments
        ---------
        brain_data : matrix
        sl_mask : matrix
            Boolean, mask for voxels to run searchlight over
        myrad : float
            Radius of searchlight [not sure if this is necessary]
        bcvar : list, dictionary (optional)
            Another variable to use with searchlight kernel
            For ex: condition labels so that you can determine 
            to which condition each 3D volume corresponds

        Returns
        -------
        float 
            Coefficient

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    # get behavioral trajectories
    behav_trajs = bcvar['behav_trajs'] 
    behav_metric = bcvar['behav_metric'] 
    neural_metric = bcvar['neural_metric'] 

    # get neural trajectories from searchlight volume
    neural_sl = get_sl_data(brain_data[0], sl_mask)
    neural_sl = remove_neutrals(neural_sl)

    # if want to reduce dimensionality, do it here...
    neural_trajs = organize_by_character(neural_sl)

    behav_rdv  = symm_mat_to_ut_vec(pairwise_trajectory_distances(behav_trajs, metric=behav_metric))
    neural_rdv = symm_mat_to_ut_vec(pairwise_trajectory_distances(neural_trajs, metric=neural_metric))
    if neural_metric == 'correlation':
        neural_rdv = 1 - np.arctanh(1 - neural_rdv)
    return scipy.stats.kendalltau(behav_rdv, neural_rdv)[0]

def run_trajsim_sl(func_fname, func_mask_fname=None, roi_mask=None, 
                   behav_metric='euclidean', neural_metric='correlation', radius=3):

    ''' wrapper to run trajsim searchlight '''

    np.random.seed(33)

    # set up
    out_dir = f"{('/').join(func_fname.split('/')[:-3])}/searchlights"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # get behavioral trajectories
    sub_id = func_fname.split('/')[-2]
    behav = load_behavior(sub_id, neutrals=False)
    bcvar_dict = {
        'behav_trajs': organize_by_character(
            behav[['affil_coord', 'power_coord']].values
        )
    }
    bcvar_dict['behav_metric'] = behav_metric
    bcvar_dict['neural_metric'] = neural_metric

    # run searchlight kernel
    sl = Searchlight(sl_f=sl_trajsim, shape='ball', radius=radius, min_prop=0.10, num_sls=10)
    sl.prepare(func_fname, func_masks=func_mask_fname, roi_masks=roi_mask, bcvar=bcvar_dict)
    if (roi_mask is not None) and ('bilateral_HPC' in roi_mask):
        sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_trajsim-ew_bilateral_HPC_cube{radius}mm') # ew for elementwise
    else:
        sl.run(save=True, out_prefix=f'{out_dir}/{sub_id}_trajsim-ew_cube{radius}mm')


#---------------------------------------------------------------------------------------------------------
# 2nd level tests
#---------------------------------------------------------------------------------------------------------


def find_sl_imgs(fname_pattern, sl_dir):

    sl_niis = glob.glob(f'{sl_dir}/*{fname_pattern}*')
    sl_niis = [s for s in sl_niis if int(s.split('/')[-1].split('_')[0]) in incl] # filter out subjects not in incl
    print(f'Found {len(sl_niis)} searchlight images')

    # split sl_niis into two samples if sub_id > 2 or not
    sl_dict = {'Initial': {'sub_ids':[], 'imgs':[]}, 'Validation': {'sub_ids':[], 'imgs':[]}, 'Combined': {'sub_ids':[], 'imgs':[]}}
    for sl_nii in sl_niis:
        sub_id = sl_nii.split('/')[-1].split('_')[0]
        if len(sub_id) > 2: 
            sl_dict['Validation']['sub_ids'].append(int(sub_id))
            sl_dict['Validation']['imgs'].append(sl_nii)
        else:   
            sl_dict['Initial']['sub_ids'].append(int(sub_id))           
            sl_dict['Initial']['imgs'].append(sl_nii)
        sl_dict['Combined']['sub_ids'].append(int(sub_id))
        sl_dict['Combined']['imgs'].append(sl_nii)
        
    print(f'Initial n = {len(sl_dict["Initial"]["sub_ids"])}')
    print(f'Validation n = {len(sl_dict["Validation"]["sub_ids"])}')
    return sl_dict

def run_sl_ttest(analysis, shape, sl_dir, samples=None, fwhm=6, parametric=True):

    sl_dict = find_sl_imgs(analysis, shape, sl_dir)
    samples = samples or ['Initial', 'Validation', 'Both']
    ttest_dir = f'{sl_dir}/ttest'
    if not os.path.exists(ttest_dir): os.mkdir(ttest_dir)

    stat_imgs = []
    for sample in samples:

        # get images etc
        if sample == 'Both':
            sample_imgs, sub_dfs = [], []
            for s, samp in enumerate(['Initial', 'Validation']):
                sub_ids = [s.split('/')[-1].split('_')[0] for s in sl_dict[samp]['imgs']]
                sub_dfs.append(pd.DataFrame({'subject_label': sub_ids, 'sample': s}))
                sample_imgs.extend(sl_dict[samp]['imgs'])
            sub_df = pd.concat(sub_dfs)
            design_matrix = nilearn.glm.second_level.make_second_level_design_matrix(sub_df['subject_label'].values, confounds=sub_df)
        else:
            sample_imgs = sl_dict[sample]['imgs']
            design_matrix = None

        # define filename
        example_img = sample_imgs[0].split('/')[-1]
        img_info = ('_').join(example_img.split('_')[1:3])
        fname = f'{ttest_dir}/Sl_{analysis}-{img_info}_{sample}-n{len(sample_imgs)}_{fwhm}mm-fwhm' 
        if not parametric: fname = f'{fname}_tfce-neglog10p'

        if not os.path.exists(f'{fname}.nii.gz'):
            print(f'Computing t-test for {sample} sample')

            if parametric: 
                stat_img = compute_ttest(sample_imgs, design_matrix=design_matrix, 
                                         second_level_contrast='intercept', fwhm=fwhm)

            else: # tfce in a gray matter image
                gm_img   = nilearn.masking.compute_multi_brain_mask(sample_imgs, threshold=0.25, connected=False, opening=2, mask_type='gm')
                stat_img = compute_permutation_ttest(sample_imgs, design_matrix=design_matrix, mask_img=gm_img,
                                                     second_level_contrast='intercept',
                                                     fwhm=fwhm, tfce=True)

            stat_img.to_filename(f'{fname}.nii.gz')
        else:
            print(f'Already ran for {sample} sample')
            stat_img = image.load_img(f'{fname}.nii.gz')
        stat_imgs.append(stat_img)
        
    return stat_imgs
