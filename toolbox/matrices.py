#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    Functions that make manipulating matrices easier
'''

import sklearn as skl
import nilearn as nil
import nibabel as nib
import numpy   as np
import pandas  as pd

from sklearn.metrics import pairwise_distances

def bootstrap_matrix(matrix, random_state=None):
    ''' shuffles similarity matrix based on recommendation by Chen et al., 2016 '''
    s = skl.utils.check_random_state(random_state)
    n = matrix.shape[0]
    bs = sorted(s.choice(np.arange(n), size=n, replace=True))
    return matrix[bs, :][:, bs]

def digitize_matrix(matrix, n_bins=10): 
    '''
        Digitize an input matrix to n bins (10 bins by default)
        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    matrix_bins = [np.percentile(np.ravel(matrix), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
    matrix_vec_digitized = np.digitize(np.ravel(matrix), bins = matrix_bins) * (100 // n_bins) # compute the vector digitized value 
    matrix_digitized = np.reshape(matrix_vec_digitized, np.shape(matrix)) # reshape to matrix
    matrix_digitized = (matrix_digitized + matrix_digitized.T) / 2  # force symmetry in the plot
    return matrix_digitized

def symmetrize_matrix(a):
    """
        Return a symmetrized version of NumPy array a.

        Values 0 are replaced by the array value at the symmetric
        position (with respect to the diagonal), i.e. if a_ij = 0,
        then the returned array a' is such that a'_ij = a_ji.

        Diagonal values are left untouched.

        a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
        for i != j.
    """
    return a + a.T - np.diag(a.diagonal())

def fill_in_upper_tri(sym_mat, diagonal=0):
    sym_mat = sym_mat + sym_mat.T - np.diag(np.diag(sym_mat))
    np.fill_diagonal(sym_mat, diagonal)
    return sym_mat

def symm_mat_labels_to_vec(labels, upper=True):
    '''
        lower or upper triangle label pairs from symm matrices, excl. diagonal
    '''
    n = len(labels)
    mat = pd.DataFrame(np.zeros((n, n)))
    for r in range(0, n):
        for c in range(r+1, n):
            mat.loc[r, c] = labels[r] + '_and_' + labels[c]
            mat.loc[c, r] = labels[c] + '_and_' + labels[r]
    if upper:
        ut  = pd.DataFrame(np.triu(mat, k=1)) # matches mat df
        vec = ut.values.flatten()
    else:
        lt  = pd.DataFrame(np.tril(mat, k=-1)) 
        vec = lt.values.flatten()
    return vec[vec!=0] 

def sort_symm_mat(mat, vec):
    
    ''' Sorts rows/columns of a symmetrical matrix according to a separate vector '''
    sort = vec.argsort()
    return mat[sort][:, sort]

def make_symm_mat_mask(orig_ixs, size=(63)):
    '''
        make a boolean mask for a symmetrical matrix of a certain size using the col/row ixs from original variables

        Arguments
        ---------
        ixs : list-like
            the ixs of the original variable to be set to True
        size : tuple (optional, default=(63))
            size of the symmetrical matrix

        Returns
        -------
        boolean mask 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    if type(size) is not int: 
        assert size[0] == size[1], 'need a symmetrical matrix; change size parameter'
        size = size[0]
    mask = np.ones((size, size))
    for ix in orig_ixs:
        mask[ix,:] = 0
        mask[:,ix] = 0
    return symm_mat_to_ut_vec(mask == 1)

def ut_vec_pw_dist(x, metric='euclidean'):
    if (isinstance(x, pd.Series)) or (isinstance(x, pd.DataFrame)): x = x.values
    if x.ndim == 1:  x = x.reshape(-1,1)
    return symm_mat_to_ut_vec(pairwise_distances(x, metric=metric))

# change shape 
def symm_mat_to_ut_vec(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    vec_ut = mat[np.triu_indices(len(mat), k=1)]
    return vec_ut

def ut_mat_to_symm_mat(mat):
    ''' go from upper tri matrix to symmetrical matrix '''
    for i in range(0, np.shape(mat)[0]):
        for j in range(i, np.shape(mat)[1]):
            mat[j][i] = mat[i][j]
    return mat

def ut_vec_to_symm_mat(vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat   = ut_vec_to_ut_mat(vec)
    symm_mat = ut_mat_to_symm_mat(ut_mat)
    return symm_mat

def ut_vec_to_ut_mat(vec):
    '''
        go from vectorized/flattened upper tri to a upper tri matrix
            1. solve to get matrix size: matrix_len**2 - matrix_len - 2*vector_len = 0
            2. then populate upper tri of a m x m matrix with the vector elements 
    '''
    
    # solve quadratic equation to find size of matrix
    from math import sqrt
    a = 1; b = -1; c = -(2*len(vec))   
    d = (b**2) - (4*a*c) # discriminant
    roots = (-b-sqrt(d))/(2*a), (-b+sqrt(d))/(2*a) # find roots   
    if False in np.isreal(roots): # make sure roots are not complex
        raise Exception('Roots are complex') # dont know if this can even happen if not using cmath...
    else: 
        m = int([root for root in roots if root > 0][0]) # get positive root as matrix size
        
    # fill in the matrix 
    mat = np.zeros((m,m))
    vec = vec.tolist() # so can use vec.pop()
    c = 0  # excluding the diagonal...
    while c < m-1:
        r = c + 1
        while r < m: 
            mat[c,r] = vec[0]
            vec.pop(0)
            r += 1
        c += 1
    return mat

def remove_diag(arr):
    arr = arr.copy()
    np.fill_diagonal(arr, np.nan)
    return arr[~np.isnan(arr)].reshape(arr.shape[0], arr.shape[1] - 1)
