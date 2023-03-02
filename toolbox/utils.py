import time, re
import pandas as pd
import numpy as np
import functools
import itertools
from sklearn.metrics import pairwise_distances

#-------------------------------------------------------------------------------------------
# math 
#-------------------------------------------------------------------------------------------

def exponential_decay(init, decay, time_steps):
    # f(t) = i(1 - r) ** t
    return np.array([init * (1 - decay) ** step for step in range(time_steps)])


def linear_decay(init, decay, time_steps):
    # f(t) = i - r * t
    # dont let it decay past 0
    return np.array([max((0, init - decay * step)) for step in range(time_steps)])


def running_mean(X, step=2):
    ''' 
        a mean over a period of time
    '''
    X = np.array(X)
    cumsum = np.cumsum(np.insert(X, 0, 0)) 
    return (cumsum[step:] - cumsum[:-step]) / float(step)


def cumulative_mean(X):
    cumsum = np.cumsum(X, 0)
    cumnum = np.cumsum(np.arange(1,len(X)+1, 0))
    return cumsum / cumnum

#-------------------------------------------------------------------------------------------
# checking
#-------------------------------------------------------------------------------------------

def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except Exception:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except Exception:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')


def get_function_sourcecode(func):
    import inspect
    lines = inspect.getsource(func)
    print(lines)


def all_equal(iterable):
    ''' check if all items in a list are identical '''
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def all_equal_arrays(L):
    ''' check if difference in List L along axis of stacking == 0  '''
    return (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()


def is_numeric(input_str):
    ''' check if all characters in a string are numeric '''
    return all(char.isdigit() for char in input_str)


def is_finite(array):
    return array[np.isfinite(array)]


def get_unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#-------------------------------------------------------------------------------------------
# list & array manipulation
# TODO: simplify these into a smaller set of more robust functions
#-------------------------------------------------------------------------------------------

def flatten_nested_lists(L):
    return [i for sL in L for i in sL]


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
    x = np.array(x)
    if x.ndim == 1:  x = x.reshape(-1,1)
    return symm_mat_to_ut_vec(pairwise_distances(x, metric=metric))

 
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


def ut_vec_to_symm_mat(ut_vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat   = ut_vec_to_ut_mat(ut_vec)
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

#-------------------------------------------------------------------------------------------
# text parsing
#-------------------------------------------------------------------------------------------

def remove_nonnumeric(str_):
    return re.sub(r'[^0-9]', '', str_)
    

def remove_nontext(string):
    return re.sub(r'\W+', '', string)


def find_pattern(strings, pattern, verbose=0):
    if '*' in pattern:
        if verbose: print('Replacing wildcard * with regex .+')
        pattern = pattern.replace('*','.+')
    regex = re.compile(pattern)
    return [s for s in strings if re.match(pattern, s)]


def get_strings_matching_substrings(strings, substrings):
    ''' return strings in list that partially match any substring '''
    matches = [any(ss in s for ss in substrings) for s in strings]
    return list(np.array(strings)[matches])


def is_numeric(input_str):
    ''' check if all characters in a string are numeric '''
    return all(char.isdigit() for char in input_str)

#-------------------------------------------------------------------------------------------
# pandas dataframes
#-------------------------------------------------------------------------------------------

def merge_dfs(df_list, on=None, how='inner'):
    return functools.reduce(lambda x, y: pd.merge(x, y, on=on, how=how), df_list)


def move_cols_to_front(df, cols):
    return df[cols + [c for c in df if c not in cols]]


def reset_sort(df):
    return df.sort_values(by=['sub_id'], ascending=False).reset_index(drop=True)

#-------------------------------------------------------------------------------------------
# numpy structured arrays
#-------------------------------------------------------------------------------------------

def join_struct_arrays(arrays):
    ''' merge multiple numpy structured arrays qucikly '''
    # TODO: right now fails if the arrays have overlapping column name 
    sizes   = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n       = len(arrays[0])
    joint   = np.empty((n, offsets[-1]), dtype=np.uint8) # turn into integer 8 
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset+size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), []) # fix dtyping back to original
    return joint.ravel().view(dtype)


def remove_struct_field(a, *fieldnames_to_remove):
    ''' remove a colum from a numpy structured array '''
    return a[[ name for name in a.dtype.names if name not in fieldnames_to_remove]]

#-------------------------------------------------------------------------------------------
# combos, permutations etc
#-------------------------------------------------------------------------------------------

def combos(arr, k=2):
    """ 
        arr: np.array to get combos from
        r: num of combos
    """
    return list(itertools.combinations(arr, k))


def pairs(items): 
    return list((set(itertools.compress(items, mask)) for mask in itertools.product(*[[0,1]] * len(items))))[1:]

#-------------------------------------------------------------------------------------------
# performance
#-------------------------------------------------------------------------------------------

def time_function(func, *args, iters=1):
    '''
        time a functions execution
        TODO: add arg for iterations
    '''
    diffs = []
    for i in range(iters):
        start = time.time()
        ret   = func(*args)
        end   = time.time()
        diffs.append(end - start)
    return f'{np.mean(diffs)}s average over {iters} iterations'