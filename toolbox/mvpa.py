import time
import numpy as np

import nibabel as nib
import nilearn as nil
# import brainiak.searchlight.searchlight

import time
from sklearn.pipeline import make_pipeline, Pipeline
from nilearn.image import load_img, resample_to_img, binarize_img
from nilearn.masking import compute_brain_mask, compute_multi_brain_mask, intersect_masks
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from images import get_nifti_info, save_as_nifti
import utils

#-------------------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------------------

# create a sklearn pipeline
def create_pipeline(estimator, steps=[]):
    
    pl_dict = {'scaler': StandardScaler(), 
               'selector': VarianceThreshold(),
               'pca': PCA(n_components = 25)}
    
    return Pipeline([(step, pl_dict[step]) for step in steps] + [('estimator', estimator)])


#-------------------------------------------------------------------------------------------
# RDMs
#-------------------------------------------------------------------------------------------


def get_rdm(betas):
    '''
        Computes representational similarity & dissimilarity matrices from betas
        Drops any voxels (columns) with no variation, assuming these are just noise

        Arguments
        ---------
        betas : array
            With shape : (num_estimates, num_voxels)

        Returns
        -------
        rdm : array
            representational dissimilarity matrix

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    with np.errstate(divide='ignore'): # ignore divide by zero warnings
        betas = VarianceThreshold().fit_transform(betas) # drop unvarying voxels
        rdm   = 1 - np.arctanh(np.corrcoef(betas)) # fisher-z transformed correlations
        np.fill_diagonal(rdm, 0) # 0s become -inf after arctanh 
    return rdm


#-------------------------------------------------------------------------------------------
# Analyses
#-------------------------------------------------------------------------------------------


def fit_huber(X, y, epsilon=1.75, alpha=0.0001):
    '''
        Run huber regression 
        Huber regressors minimizes squared loss for non-outliers and absolute loss for outliers

        Arguments
        ---------
        rdv_X : array
            _description_
        rdv_y : array
            _description_
        epsilon : float (optional)
            Controls how many observations should be considered outliers: smaller, more robust to outliers
            In sklearn default epsilon = 1.35; 
            Default: 1.75
        alpha : float (optional)
            Default regularization strength = 0.0001 
            Default: 0.0001

        Returns
        -------
        array 
            Coefficients

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    huber = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=100)
    huber.fit(X, y)
    return huber.coef_


def calc_ps(patterns):
    ''' calculates avg and std of fisher z-transformed correlations of patterns
    '''
    rsv = utils.symm_mat_to_ut_vec(1 - get_rdm(patterns)) # fisher z-transformed correlation
    return [np.mean(rsv), np.std(rsv)]


def fit_mds(rdm, metric=True, n_components=5, digitize=False):
    '''
        Computes multidimensional scaling from a precomputed representational dissimilarity matrix

        Arguments
        ---------
        rdm : array
            _description_
        n_components : int (optional)
            How many components to compute 
            Default: 5

        Returns
        -------
        embedding : array
            _description_
        kruskal_stress : float
            _description_

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    if digitize: rdm = utils.digitize_matrix(rdm)
    mds = MDS(n_components=n_components, metric=metric,
              dissimilarity="precomputed", random_state=85)  
    mds_fitted = mds.fit(rdm)

    # Kruskal's stress: scaled version of raw stress
    # -- https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    # -- raw stress = 1/2 * sum(euclidean dists between embedded coords - inputted dists)^2
    kruskal_stress = np.sqrt(mds_fitted.stress_ / (0.5 * np.sum(rdm ** 2))) 
    return mds_fitted.embedding_, kruskal_stress


def bin_distances(distances, n_bins=2, encode='ordinal', strategy='quantile'):
    '''
        _summary_

        Arguments
        ---------
        distances : _type_
            _description_
        n_bins : int (optional, default=2)
            _description_ 
        encode : str (optional, default='ordinal')
            _description_ 
        strategy : str (optional, default='quantile')
            _description_ 

        Returns
        -------
        _type_ 
            _description_

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # get data into correct shape
    if isinstance(distances, list):
        distances = np.array(distances)
    if distances.ndim == 1:
        distances = distances.reshape(-1,1)
        
    binner = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    binner.fit(distances)  
    bin_ixs = binner.transform(distances).flatten()
    bin_counts = [np.sum(bin_ixs==b) for b in range(n_bins)]
    
    if np.sum(bin_counts)/len(distances) != 1.0: 
        print('The bin counts dont match the number of trials')
    mean_dists = np.array([np.mean(distances[bin_ixs==b]) for b in range(n_bins)])
    
    return bin_ixs, mean_dists


#-------------------------------------------------------------------------------------------
# Searchlights
#-------------------------------------------------------------------------------------------


# get data for searchlight
def get_sl_data(brain_data, sl_mask, verbose=False):
    ''' returns searchlight data of shape: (num_volumes, num_voxels)'''

    assert brain_data.shape[0:3] == sl_mask.shape, 'The brain and mask have different shapes'
    num_vols = brain_data.shape[3]
    assert num_vols in [5, 60, 63], f'Number of volumes is prob. wrong: {num_vols}'
    sl_data  = brain_data.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], num_vols).T
    
    if verbose:
        print(f"SL data shape: {brain_data.shape}")
        print(f"SL data shape after reshaping: {sl_data.shape}")
        print(f"SL mask shape: {sl_mask.shape}\n")
    
    return sl_data


# single subject searchlights
def sl_huber_rsa(brain_data, sl_mask, myrad, bcvar):
    '''
        Run representational similarity analysis w/ huber regression in a brainiak searchlight 

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

    t1 = time.time()
    
    # get searchlight volume 
    data_sl = get_sl_data(brain_data[0], sl_mask)

    # transform data into rdv
    neural_rdv = symm_mat_to_ut_vec(get_rdm(data_sl))

    # get behavior/label/etc 
    predictor_rdvs = bcvar

    # regress brain onto behavior
    huber = HuberRegressor(epsilon=1.75, alpha=0.0001, max_iter=50)
    huber.fit(predictor_rdvs, neural_rdv)
    estimate = huber.coef_[0]

    print('Kernel duration: %.2f\n\n' % (time.time() - t1))
    
    return estimate


def sl_mds(data, sl_mask, myrad, bcvar):

    '''
        Computes multidimensional scaling from a precomputed representational dissimilarity matrix

        Arguments
        ---------
        rdm : array
            _description_
        bcvar : dict of n_components & metric (boolean)

        Returns
        -------
        kruskal_stress : float
            _description_

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    # get data for searchlight
    sl_data = get_sl_data(data[0], sl_mask)

    # get representational dissimilarity matrix
    rdm = get_rdm(sl_data)
    assert rdm.shape in [(5,5), (60,60), (63,63)], f'Neural RDM is prob. wrong shape: {rdm.shape}'

    # fit MDS
    n_components, metric = bcvar['n_components'], bcvar['metric']
    mds = MDS(n_components=n_components, metric=metric, 
              dissimilarity="precomputed", random_state=85)  
    mds.fit(rdm)

    # Kruskal's stress: scaled version of raw stress
    # -- https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    # -- raw stress = 1/2 * sum(euclidean dists between embedded coords - inputted dists)^2
    kruskal_stress = np.sqrt(mds.stress_ / (0.5 * np.sum(rdm ** 2))) 

    return kruskal_stress



#-------------------------------------------------------------------------------------------
# DEV
#-------------------------------------------------------------------------------------------

# character cv iterator: train on 4, test on 1

# role_num = decision_details['role_num'].reset_index(drop=True)
# role_num = role_num[role_num != 9] # exclude neutral
# cv_iterable = []
# for c in np.arange(1,6):
#     test  = list(role_num[role_num == c].index)
#     train = list(set(role_num.index) - set(test))
#     cv_iterable.append((train, test))