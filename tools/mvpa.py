#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    Functions that make pattern analyses easier
'''

import time
import numpy as np

import nibabel as nib
import nilearn as nil
import brainiak.searchlight.searchlight

from sklearn.pipeline import make_pipeline, Pipeline
from nilearn.image import load_img, binarize_img, resample_to_img
from nilearn.masking import compute_brain_mask
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

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
    betas = VarianceThreshold().fit_transform(betas) # drop unvarying voxels
    rdm   = 1 - np.arctanh(np.corrcoef(betas)) # fisher-z transformed correlations
    return rdm

########################################################################################################
## ROI-based mvpa 
########################################################################################################

def fit_huber(X, y, epsilon = 1.75, alpha = 0.0001):
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

def avg_ps(betas):
    rsm = 1 - get_rdm(betas)
    ps  = np.mean(symm_mat_to_ut_vec(rsm)) # mean fisher z-transformed correlation
    return ps

def fit_mds(rdm, n_components=5):
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
    rdm_dgz = digitize_matrix(rdm)
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=85)  
    mds_fitted = mds.fit(rdm_dgz)
    embedding = mds_fitted.embedding_

    # Kruskal's stress: scaled version of raw stress
    # -- https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    # -- raw stress = 1/2 * sum(euclidean dists between embedded coords - inputted dists)^2
    kruskal_stress = np.sqrt(mds_fitted.stress_ / (0.5 * np.sum(rdm ** 2))) 

    return embedding, kruskal_stress

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

def fit_clf(clf, X, y, cv_splits=6, scale=True):

    # initialize 

    eval_df = pd.DataFrame()
    pred_dfs = []
    
    # cross-validator - maybe just pass an iterator
    if isinstance(cv_splits, int):
        # stratified to balance each fold by class (i.e., character) 
        folds = StratifiedKFold(n_splits=cv_splits, random_state=76, shuffle=True).split(X, y)
    elif (isinstance(cv_splits, str)) & (cv_splits=='loo'): 
        folds = LeaveOneOut().split(X, y)
    else:
        folds = cv_splits

    # get rid of voxels with 0/nan values
    X = VarianceThreshold().fit_transform(X) 
    
    # cross-validated decoding
    for k, (train, test) in enumerate(folds):

        # if standardizing, fit a scaling model on training folds
        if scale: 
            scaler  = StandardScaler().fit(X[train])     
            X_train = scaler.transform(X[train])
            X_test  = scaler.transform(X[test])
        else:
            X_train = X[train]
            X_test  = X[test]

        # fit classifier on training folds
        decoder = clone(clf)
        decoder.fit(X_train, y[train]) 

        # predict on held out fold
        y_preds = decoder.predict(X_test)
        pred_df = pd.DataFrame(np.vstack([test, y_preds, y[test], (y_preds == y[test]) * 1]).T,
                            columns=['ix0', 'pred', 'actual', 'correct'])
        pred_df.insert(0, 'split', k)

        # evaluate performance
        eval_df.loc[k, 'split'] = k
        eval_df.loc[k, 'acc'] = decoder.score(X_test, y[test])
        eval_df.loc[k, 'balanced_acc'] = balanced_accuracy_score(y[test], y_preds)
        eval_df.loc[k, 'f1'] = f1_score(y[test], y_preds, average='weighted')
        eval_df.loc[k, 'phi'] = matthews_corrcoef(y[test], y_preds)
        eval_df.loc[k, 'dice'] = sp.spatial.distance.dice(y[test], y_preds)
        
        # get probabilities
        if hasattr(decoder, 'predict_proba'): 
            y_probas = decoder.predict_proba(X_test)
            for p, y_probas_ in enumerate(y_probas.T):
                pred_df['probas_class' + str(p)] = y_probas_

        pred_dfs.append(pred_df)
    
    pred_df = pd.concat(pred_dfs)
    pred_df.reset_index(inplace=True, drop=True)

    # output dict
    clf_dict = {'cv_splits': cv_splits, 
                'preds': pred_df,
                'eval': eval_df} 

    return clf_dict

########################################################################################################
## Searchlights
########################################################################################################

def get_sl_data(brain_data, sl_mask, verbose=True):
    ''' returns searchlight data of shape: (num_volumes, num_voxels)'''
    assert brain_data.shape[0:3] == sl_mask.shape, 'The brain and mask have different shapes'
    num_vols = brain_data.shape[3]
    reshaped = brain_data.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], num_vols).T
    
    if verbose:
        print(f"Searchlight data shape: {brain_data.shape}")
        print(f"Searchlight data shape after reshaping: {data_sl.shape}")
        print(f"Searchlight mask shape: {sl_mask.shape}\n")
    
    return reshaped

def create_pipeline(estimator, steps=[]):
    
    pl_dict = {'scaler': StandardScaler(), 
               'selector': VarianceThreshold(),
               'pca': PCA(n_components = 25)}
    
    return Pipeline([(step, pl_dict[step]) for step in steps] + [('estimator', estimator)])

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

    # transform volume (optional)
    neural_rdv = symm_mat_to_ut_vec(get_rdm(data_sl))

    # get behavior/label/etc 
    predictor_rdvs = bcvar

    # run stats
    huber = HuberRegressor(epsilon=1.75, alpha=0.0001, max_iter=50)
    huber.fit(predictor_rdvs, neural_rdv)
    estimate = huber.coef_[0]

    print('Kernel duration: %.2f\n\n' % (time.time() - t1))
    
    return estimate

def clf_cv(pl, X, y, cv, gridsearch=False, parameters=None):
    
    if gridsearch:
        acc = []
        for test, train in cv.split(X, y):

            # inner loop: optimize pipeline w/ grid search 
            grid = GridSearchCV(pl, parameters, cv=cv, return_train_score=False)
            grid.fit(X[train], y[train])

            # outer loop: test best pipeline
            acc.append(grid.score(X[test], y[test]))
    else: 
        acc = cross_val_score(pl, X, y, cv=cv)
        
    return acc.mean()

def sl_svm(data, mask, myrad, bcvar, pl_steps=['scaler'], gridsearch=False):
    '''
        Run SVM in a brainiak searchlight within single subject

        Arguments
        ---------
        data : list of matrices
            Includes at least 1 subject's 4D matrix inside it
            Can include multiple subjects; most useful if want to run SLs within small volume and not save individual niftis
        mask : matrix
            Boolean mask for voxels to run searchlight over
        myrad : float
            Radius of searchlight [not sure if this is necessary..?]
        bcvar : list, dictionary (optional)
            Another variable to use with searchlight kernel
            E.g., condition labels for each trial/observation
        pl_steps : list of strings (optional)
            Pre-SVM pipeline steps; set to an empty list if want no preprocessing
            Default : 'scaler'
        gridsearch : boolean (optinal)
            Whether to optimize with grid search in inner loop 
            Default: False
        
        Returns
        -------
        float 
            Average 5-fold cross-validated accuracy

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    t1 = time.time()
    
    accuracy = []
    
    for X, y in zip(data, bcvar):
        
        # get searchlight volume
        X = get_sl_data(X, mask)

        # run svm
        svm = SVC(kernel='linear')
        pl  = create_pipeline(svm, steps=pl_steps)
        params = {'estimator__C': [0.0001, 0.01, 0.5, 1]} 
        cv  = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=23) 
        acc = clf_cv(pl, X, y, cv, gridsearch=gridsearch, parameters=params)

        accuracy.append(acc.mean())
        
    print('Kernel duration: %.2f\n\n' % (time.time() - t1))
    
    if len(data) == 1: return accuracy[0]
    else:              return accuracy

def sl_svm_group(data, sl_mask, myrad, bcvar):
    '''
        Run an SVM searchlight on subject-level

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

        Returns
        -------
        _type_ 
            _description_

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # get sl volumes
    X = [get_sl_data(sub_data, sl_mask).flatten() for sub_data in data] # (subs, features) 
    
    # get labels
    y = [bcv for bcv in bcvar]

    # run svm
    svm = SVC(kernel='linear', C=1)
    acc = cross_val_score(svm, X, y, cv=5)
    
    return acc.mean()

def run_sl(images, masks, sl_kernel, bcvar=None, other_masks=None, shape='ball', radius=3, min_prop=0.10, max_blk_edge=10):
    '''
        Run brainiak searchlight, save nifti 

        Arguments
        ---------
        images : list of str
            Path(s) to functional nifti paths
        masks : list of str
            Path(s) to binary mask to compute searchlight in (e.g., mask from fmriprep)
        sl_kernel : function
            The searchlight function to be distributed
        other_masks : list of str or float (optional)
            If want to intersect mask_path with mask(s)
            Default: None
        bcvar : list, dictionary, etc (optional)
            A variable to broadcast to all searchlight kernels
            Example: condition labels for each 3D volume
        shape : str (optional)
            Shape of searchlight: 'cube', 'ball', 'diamond'
            Default: 'ball'
        radius : int (optional)
            Radius of searchlight
            Default: 3
        min_prop : float (optional)
            Minimum proportion of voxels that need to be 'active' to have sl computed
            Default: 0.10
        max_blk_edge : int (optional)
            Number of sls to run concurrently
            Default: 10

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    sl_shapes = {'cube': brainiak.searchlight.searchlight.Cube,
                 'ball': brainiak.searchlight.searchlight.Ball,
                 'diamond': brainiak.searchlight.searchlight.Diamond}

    # load subject(s) images
    data = [nib.load(f).get_data() for f in images]
    assert all_equal_arrays([s.shape for s in data]), 'Brain(s) are different shapes'
        
    # load/compute a single boolean mask
    if isinstance(other_masks, float): # float : compute a GM mask from sub masks using 'other_mask' as threshold
        if len(masks) == 1: compute_mask = compute_brain_mask # single subject
        else:               compute_mask = compute_multi_brain_mask
        mask = compute_mask(masks, mask_type='gm', threshold=other_masks, connected=False) 
    else: # intersect mask(s) w/ sub masks
        if other_masks is None: 
            all_masks = masks
        else:
            if isinstance(other_masks, list): # multiple masks in a list
                other_masks = intersect_masks(masks, threshold=0, connected=False) # union of 'other_masks' (threshold == 0)
            other_masks = resample_to_img(masks, images[0], interpolation="nearest") # resample to correct image (use "nearest" to keep binary)
            all_masks   = masks + [other_masks]
        mask = intersect_masks(all_masks, threshold=1, connected=False) # intersection of all masks (threshold == 1)    
    for d in data: assert d.shape[0:3] == mask.shape, 'Brain(s) and mask are different shapes'
        
    print(f'Number of subjects: {len(data)}') 
    print(f'Func data shape: {data[0].shape}') 
    print(f'Searchlight is a {shape} with radius: {radius}')

    # build & distribute sl to run
    t1 = time.time()
    sl = brainiak.searchlight.searchlight.Searchlight(sl_rad=radius, shape=sl_shapes[shape], 
                                                      min_active_voxels_proportion=min_prop,
                                                      max_blk_edge=max_blk_edge) 
    sl.distribute([data], mask)
    if bcvar is not None: sl.broadcast(bcvar) # broadcast data needed for all sls       
    sl_result = sl.run_searchlight(sl_kernel, pool_size=1) 
    t2 = time.time()
    print('Searchlight duration: %.2f\n\n' % (t2 - t1))
    
    return sl_result

########################################################################################################
# character cv iterator: train on 4, test on 1

# role_num = decision_details['role_num'].reset_index(drop=True)
# role_num = role_num[role_num != 9] # exclude neutral
# cv_iterable = []
# for c in np.arange(1,6):
#     test  = list(role_num[role_num == c].index)
#     train = list(set(role_num.index) - set(test))
#     cv_iterable.append((train, test))