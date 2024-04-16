import time
import numpy as np
import nibabel as nib
import nilearn as nil
import scipy

# import brainiak.searchlight.searchlight

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score

from images import get_nifti_info, save_as_nifti
from utils import symm_mat_to_ut_vec, digitize_matrix


#-------------------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------------------


def create_pipeline(estimator, steps=None):
    
    if steps is None: steps = []
    pl_dict = {'scaler': StandardScaler(), 
               'selector': VarianceThreshold(),
               'pca': PCA(n_components = 25)}

    return Pipeline([(step, pl_dict[step]) for step in steps] + [('estimator', estimator)])


#-------------------------------------------------------------------------------------------
# RDMs
#-------------------------------------------------------------------------------------------


def get_corrdist_matrix(betas):
    '''
        Helper to representational similarity & dissimilarity matrices from betas
        Drops any voxels (columns) with no variation, assuming these are just noise
        Transforms correlations to fisher-z scale, then turns into correlation distances
        
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
# Analysis-specific functions
#-------------------------------------------------------------------------------------------


def fit_huber(X, y, epsilon=1.75, alpha=0.0001):
    '''
        Run huber regression 
        Huber regressors minimizes squared loss for non-outliers and absolute loss for outliers

        Arguments
        ---------
        X : array
            _description_
        y : array
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
    rsv = symm_mat_to_ut_vec(1 - get_corrdist_matrix(patterns)) # fisher z-transformed correlation
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
    if digitize: rdm = digitize_matrix(rdm)
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


class Searchlight:

    '''
        Run brainiak searchlight, in single or across multiple subjects
        Basic steps of searchlight:
        - 0) prepare brain data & mask to run in
        - 1) create a searchlight object
        - 2) distribute brain data & mask to searchlights
        - 3) broadcast data needed for (all) searchlights to (all) searchlights
        - 4) compute function (ie, "kernel") in each searchlight & save estimate in center voxel
        - 5) save results as a nifti

        Arguments
        ---------
        func_fnames : list of str
            Path(s) to functional nifti paths
        masks : list of str
            Path(s) to binary mask to compute searchlight in (e.g., mask from fmriprep)
        sl_kernel : function to compute in searchlights 
            The searchlight function to be distributed (e.g., SVM, etc)
        other_masks : list of str or float (optional)
            To intersect with masks
            Default: None
        bcvar : list, dictionary, etc (optional)
            A variable to broadcast to all searchlight kernels
            Example: condition labels for each 3D volume
        shape : str (optional)
            Shape of searchlight: 'cube', 'ball', 'diamond'
            Default: 'ball'
        radius : int (optional)
            Radius of searchlight in voxels
            Default: 3
        min_prop : float (optional)
            Minimum proportion of voxels that need to be 'active' to have sl computed
            Default: 0.10
        num_sls : int (optional)
            Number of sls to run concurrently
            Default: 10

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    def __init__(self, sl_f, shape='ball', radius=3,
                 min_prop=0.10, num_sls=10):
    
        self.sl_f = sl_f
        self.radius = radius
        self.min_prop = min_prop
        self.num_sls = num_sls

        sl_shapes = {'cube': brainiak.searchlight.searchlight.Cube,
                     'ball': brainiak.searchlight.searchlight.Ball,
                     'diamond': brainiak.searchlight.searchlight.Diamond}    
        self.shape = sl_shapes[shape]
        
    def prepare(self, func_imgs, func_masks=None, 
                roi_masks=None, bcvar=None):

        self.func_imgs  = func_imgs
        self.func_masks = func_masks
        self.roi_masks  = roi_masks
        self.bcvar      = bcvar

        #-------------------------------------------------------------------------------------------
        # load subject(s) functional images into matrix
        #-------------------------------------------------------------------------------------------

        # string -> single subject
        if isinstance(func_imgs, str):
            print("Loading single subject's functional image")
            self.brain_data = [nib.load(func_imgs).get_fdata()] # put both in lists for cnsistency
            func_imgs = [func_imgs]

        # list -> multiple subjects
        elif isinstance(func_imgs, list):
            print("Loading multiple subjects' functional images")
            self.brain_data = [nib.load(f).get_fdata() for f in func_imgs]
            assert len({s.shape for s in self.brain_data}) == 1, 'Brain(s) are diff. shapes'

        _, self.vox_size, self.affine_matrix = get_nifti_info(func_imgs[0])
        n_subs = len(func_imgs)

        #-------------------------------------------------------------------------------------------
        # masking 
        #   basic masking logic: func mask + roi mask
        #   if multiple func masks (list), get intersection
        #   if multiple roi masks (list), get union 

        #   None + None -> nilearn computed brain mask
        #   None + mask -> roi mask
        #   mask + None -> subject mask
        #   mask + mask -> intersection of subject and roi mask
        #   None + float -> above GM probability threshold
        #   mask + float ->  above GM probability threshold within subject mask
        #-------------------------------------------------------------------------------------------

        # prepare functional mask(s)
        if func_masks is None: # whole-brain @ 50% threshold

            func_mask = compute_multi_brain_mask(func_imgs, mask_type='whole-brain', 
                                                 threshold=0.5, connected=False) 
            
        else:

            # ensure its a list & length matches number of subjects
            if isinstance(func_masks, str): func_masks = [func_masks]
            assert len(func_masks) == n_subs, 'Number of masks does not match number of subjects'
            if n_subs == 1: # single subject 
                func_mask = func_masks
            elif n_subs > 1: # multiple masks: get intersection (threshold == 1)
                func_mask = intersect_masks(func_masks, threshold=1, connected=False)
            
        self.func_mask = binarize_img(func_mask)
        assert np.all(np.isin(self.func_mask.get_fdata(), [0,1])), 'Functional mask is not binary'

        # prepare brain mask(s)
        # - TODO: maybe also allow for GM % WITHIN an ROI?
        if roi_masks is None: # just use functional mask
            brain_mask = self.func_mask  

        elif isinstance(roi_masks, float): # float: grey matter prob. threshold, intersected with functional mask

            brain_mask = compute_brain_mask(self.func_mask, mask_type='gm', threshold=roi_masks, connected=False) 
            assert np.sum(brain_mask.get_fdata()) <= np.sum(self.subject_mask.get_fdata()), \
                    'Region mask should =< incl. voxels than subject mask'

        else: # str or list: ROI mask(s), intesected with functional mask

            # if multiple masks (in list), get union
            if isinstance(roi_masks, list): 
                roi_mask = intersect_masks(roi_masks, threshold=0, connected=False)
            else:
                roi_mask = roi_masks

            # resample brain mask to functional image 
            roi_mask = resample_to_img(roi_mask, func_imgs[0], interpolation="nearest") # "nearest" keeps mask binary
            assert np.all(np.isin(roi_mask.get_fdata(), [0,1])), 'Region mask is not binary'

            # if there is a functional mask, intersect it with the region mask
            if func_masks is not None:
                brain_mask = intersect_masks([self.func_mask, roi_mask], threshold=1, connected=False)
            else:
                brain_mask = roi_mask

        self.brain_mask = brain_mask.get_fdata().squeeze()

        print(f'Number of subjects: {n_subs}')
        for d in self.brain_data: 
            assert d.shape[:3] == self.brain_mask.shape, \
            f'Brain(s) and mask are different shapes: {d.shape[:3]} vs. {self.brain_mask.shape}'
        print(f'Brain data shape: {self.brain_data[0].shape}')
        print(f'Searchlight is a {self.shape} with radius={self.radius}')

    def run(self, save=True, out_prefix=''):

        #-------------------------------------------------------------------------------------------
        # run searchlight
        #-------------------------------------------------------------------------------------------

        t1 = time.time()

        # build searchlight object w/ specific shape & size
        sl = brainiak.searchlight.searchlight.Searchlight(sl_rad=self.radius,
                                                          shape=self.shape, 
                                                          min_active_voxels_proportion=self.min_prop,
                                                          max_blk_edge=self.num_sls) 

        # distribute brain data & mask to searchlights
        sl.distribute(self.brain_data, self.brain_mask)

        # broadcast data needed for all searchlights (e.g., behavior or labels for modeling)
        if self.bcvar is not None: sl.broadcast(self.bcvar)

        # run kernel function in diff. searchlights in parallel
        sl_result = sl.run_searchlight(self.sl_f, pool_size=1) 

        print('Searchlight duration: %.2f\n\n' % (time.time() - t1))

        # output
        if not save: return sl_result
        save_as_nifti(sl_result, f'{out_prefix}_result.nii', self.affine_matrix, self.vox_size)
        # save_as_nifti(self.brain_mask, f'{out_prefix}_mask.nii', self.affine_matrix, self.vox_size)

def get_sl_data(brain_data, sl_mask, verbose=False):
    ''' Returns searchlight data of shape: (num_volumes, num_voxels) '''
    assert brain_data.shape[:3] == sl_mask.shape, f'The brain and mask have different shapes {brain_data.shape} vs {sl_mask.shape}'
    num_vols = brain_data.shape[3]
    assert num_vols in [5, 60, 63], f'Number of volumes is prob. wrong: {num_vols}'
    
    # reshape 
    sl_data  = brain_data.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], num_vols).T

    # remove nans/0 voxels
    sl_data = VarianceThreshold().fit_transform(sl_data)
    if verbose:
        print(f"SL data shape: {brain_data.shape}")
        print(f"SL data shape after reshaping: {sl_data.shape}")
        print(f"SL mask shape: {sl_mask.shape}\n")
    return sl_data

def sl_rsa(brain_data, sl_mask, myrad, bcvar):
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

    # get predicted rdv
    pred_rdv = bcvar 

    # get searchlight volume
    data_sl = get_sl_data(brain_data[0], sl_mask)
    if len(pred_rdv) == (60**2 - 60) / 2:
        data_sl = remove_neutrals(data_sl)
    neural_rdv = symm_mat_to_ut_vec(get_corrdist_matrix(data_sl))

    return scipy.stats.kendalltau(pred_rdv, neural_rdv)[0]

def sl_ps(brain_data, sl_mask, myrad, bcvar):
    '''
        Run an average pattern similarity analysis in a brainiak searchlight 

        Arguments
        ---------
        brain_data : matrix
        sl_mask : matrix
            Boolean, mask for voxels to run searchlight over
        myrad : float
            Radius of searchlight [not sure if this is necessary]
        bcvar : array
            an array indicating trial pairs that should be different (0) or same (1)
            
        Returns
        -------
        float 
            The difference in pattern similarity between the two conditions

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    # get predicted rdv
    pred_rsv = bcvar 

    # get searchlight volume
    data_sl = get_sl_data(brain_data[0], sl_mask)
    if len(pred_rsv) == (60**2 - 60) / 2:
        data_sl = remove_neutrals(data_sl)
    
    # use Fisher's z-transformed Pearsons correlation for pattern similarity
    neural_rsv = symm_mat_to_ut_vec(1 - get_corrdist_matrix(data_sl)) 

    # calculate pattern similarity difference
    return np.mean(neural_rsv[pred_rsv==1]) - np.mean(neural_rsv[pred_rsv==0]) 

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

    # transform neural data into rdv with correlation distance
    neural_rdv = symm_mat_to_ut_vec(get_corrdist_matrix(data_sl))

    # get behavior/label/etc 
    predictor_rdvs = bcvar

    # regress brain onto behavior
    huber = HuberRegressor(epsilon=1.75, alpha=0.0001, max_iter=50)
    huber.fit(predictor_rdvs, neural_rdv)
    print('Kernel duration: %.2f\n\n' % (time.time() - t1))
    return huber.coef_[0]

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
    rdm = get_corrdist_matrix(sl_data)
    assert rdm.shape in [(5,5), (60,60), (63,63)], f'Neural RDM is prob. wrong shape: {rdm.shape}'

    # fit MDS
    n_components, metric = bcvar['n_components'], bcvar['metric']
    mds = MDS(n_components=n_components, metric=metric, 
              dissimilarity="precomputed", random_state=85)  
    mds.fit(rdm)

    # Kruskal's stress: scaled version of raw stress
    # -- https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    # -- raw stress = 1/2 * sum(euclidean dists between embedded coords - inputted dists)^2

    return np.sqrt(mds.stress_ / (0.5 * np.sum(rdm ** 2)))

def sl_svc_gridsearch(brain_data, sl_mask, myrad, bcvar): 

    ''' 
        Linear SVC w/ 5-fold classification and inner loop gridsearch for best C
    '''

    if np.sum(sl_mask) < 14: return -1 # not enough voxels

    # pull out the data
    labels = bcvar 
    sl_data = get_sl_data(brain_data[0], sl_mask)
    if len(labels) == 60: sl_data = remove_neutrals(sl_data)

    # outer loop: cross-validation
    accs = []
    skf  = StratifiedKFold(n_splits=5)
    for train, test in skf.split(sl_data, labels):

        # get fold splits
        train_data, test_data   = sl_data[train, :], sl_data[test, :]
        train_labels, test_labels = labels[train], labels[test]    

        # run inner loop: gridsearch cv for best C
        skf_inner  = StratifiedKFold(n_splits=3)
        parameters = {'C':[0.01, 0.1, 1, 10]}
        inner_clf  = GridSearchCV(SVC(kernel='linear'), parameters,
                                  cv=skf_inner, return_train_score=True)
        inner_clf.fit(train_data, train_labels)
        C_best = inner_clf.best_params_['C']
        
        # outer loop: test accuracy
        clf = SVC(kernel='linear', C=C_best)
        clf.fit(train_data, train_labels)
        accs.append(clf.score(test_data, test_labels)) 

    return np.mean(accs)

def sl_svc(brain_data, sl_mask, myrad, bcvar): 

    ''' Linear SVC (C=1) w/ 5-fold classification '''

    if np.sum(sl_mask) < 14: return -1 # not enough voxels

    # pull out the data
    labels = bcvar 
    sl_data = get_sl_data(brain_data[0], sl_mask)
    if len(labels) == 60: sl_data = remove_neutrals(sl_data)

    # 5-fold cross-validation
    accs = []
    skf  = StratifiedKFold(n_splits=5)
    for train, test in skf.split(sl_data, labels):

        # get fold splits
        train_data, test_data     = sl_data[train, :], sl_data[test, :]
        train_labels, test_labels = labels[train], labels[test]    

        # train & test
        clf = SVC(kernel='linear', C=1)
        clf.fit(train_data, train_labels)
        accs.append(clf.score(test_data, test_labels)) 

    return np.mean(accs)



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

# multi subject searchlight - maybe not ideal
def sl_svm_group(brain_data, sl_mask, myrad, bcvar):

    '''
        Run an SVM searchlight across subjects...
        Probably not the way to do this...
        Instead do classification on a parcellation... 

        Arguments
        ---------
        data : array
            Brain data
        sl_mask : bool array
            Boolean mask to run searchlight in 
        myrad : _type_
            _description_
        bcvar : list
            A label for the classification

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # get sl volumes
    X = [get_sl_data(sub_data, sl_mask).flatten() for sub_data in brain_data] # (subs, features) 

    # get labels
    y = list(bcvar)

    # run svm: think about adding hyperparameter optimization in an inner-loop?
    svm = SVC(kernel='linear', C=1)
    acc = cross_val_score(svm, X, y, cv=5)

    return np.mean(acc)

