'''
    generic helpers
'''

import time, re, json, functools, itertools
import pandas as pd
import numpy as np
from six.moves import cPickle as pickle
from io import StringIO
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state


#-------------------------------------------------------------------------------------------
# converting from xy to ij coordinates
# to deal with indexing in numpy arrays....
# not sure when I used these last... should add more checks etc...
#-------------------------------------------------------------------------------------------


def xy_to_ij(xy, n):
    ''' 
        convert (x, y) coordinates to (i, j) numpy indices
        n is the size of the grid in one dimension
        assumes a symmetrical grid, with odd number of rows/cols
        flip x,y (i=y, j=x) and subtract 1/2 the grid size (n-1) / 2 to get indices
    '''
    if n % 2 == 0: raise ValueError(f'Error: n must be odd, got {n}')
    return np.array([int((n - 1) / 2 - xy[1]), int(xy[0] + (n - 1) / 2)])

def ij_to_xy(ij, n):
    ''' 
        convert (i, j) numpy indices to (x, y) coordinates
    '''
    if n % 2 == 0: raise ValueError(f'Error: n must be odd, got {n}')
    return np.array([int(ij[1] - (n - 1) / 2), int((n - 1) / 2 - ij[0])])

def test_xy_to_ij():
   
    # Test converting from (x, y) coordinates to (i, j) numpy indices
    n = 5 # so -2 to 2

    xy = [1, 2] 
    expected_ij = np.array([0, 3])
    actual_ij   = xy_to_ij(xy, n)
    assert np.array_equal(expected_ij, actual_ij), \
            f'Error: expected {expected_ij}, but got {actual_ij}'
    
    xy = [-1, -1]
    expected_ij = np.array([3, 1])
    actual_ij   = xy_to_ij(xy, n)
    assert np.array_equal(expected_ij, actual_ij), \
            f'Error: expected {expected_ij}, but got {actual_ij}'
    
    xy = [1, -1]
    expected_ij = np.array([3, 3])
    actual_ij   = xy_to_ij(xy, n)
    assert np.array_equal(expected_ij, actual_ij), \
            f'Error: expected {expected_ij}, but got {actual_ij}'

test_xy_to_ij()

def create_transition_matrix_from_coords(XY, size=13, nstep=1):

    # convert xy to ij, ij to flat index
    IJ = np.vstack([xy_to_ij(xy, size) for xy in XY])
    ixs = [convert_index_2d_to_1d(ij, size) for ij in IJ] # flatten ij

    # create nstep transition matrix
    n_locs = size ** 2
    T = np.zeros((n_locs, n_locs))
    for i in range(nstep, len(ixs)):
        curr_loc = ixs[i-nstep]
        next_loc = ixs[i]
        T[curr_loc, next_loc] += 1
    return T / np.sum(T)

def convert_index_2d_to_1d(ij, shape):
    # Convert 2d ij index to do 1d index (C-order)
    if isinstance(shape, int): shape = (shape, shape)
    return np.ravel_multi_index(ij, shape, order='C')

def convert_index_1d_to_2d(i, shape):
    # Convert 1d index to 2d ij index (C-order)
    return np.unravel_index(i, shape, order='C')


#-------------------------------------------------------------------------------------------
# saving/loading files
#-------------------------------------------------------------------------------------------


def pickle_file(file_, filename_, protocol=4):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f, protocol=protocol)
    f.close()

def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=json_encoder)
    f.close()

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

class json_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


#-------------------------------------------------------------------------------------------
# dataframe helpers
#-------------------------------------------------------------------------------------------


def find_cols(data, substr):
    return [c for c in data.columns if substr in c]

def print_df(df):
    print(df.to_markdown())

def finite_mask(df):
    return df[np.isfinite(df)]


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
# checking things
#-------------------------------------------------------------------------------------------


def get_attributes(obj):
    return list(obj.__dict__.keys())

def get_methods(object, spacing=30):
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
            print((f'{str(method.ljust(spacing))} '
                    + processFunc(str(getattr(object, method).__doc__)[:90])))
        except Exception:
            print(f'{method.ljust(spacing)}  getattr() failed')

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


flatten_lists = lambda x: [item for sublist in x for item in sublist]

def unique_in_order(iterable):
    ''' 
        Returns a list of unique items in the order in which they first appeared in the iterable
    '''
    seen = set()
    unique_list = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def get_rdm(values, metric='euclidean'):
    values = np.array(values)
    if values.ndim == 1: values = values.reshape(-1, 1)
    return pairwise_distances(values, metric=metric)

def get_rdv(values, metric='euclidean'):
    values = np.array(values)
    if values.ndim == 1: values = values.reshape(-1, 1)
    rdm = pairwise_distances(values, metric=metric)
    return symm_mat_to_ut_vec(rdm)

def compute_symmetric(df, func):
    ''' compute pairwise tests, returns a symmetric matrix '''
    columns = df.columns
    n = len(columns)
    results = pd.DataFrame(index=columns, columns=columns)
    for i, j in itertools.product(range(n), range(n)):
        results.iloc[i, j] = results.iloc[j, i] = func(df[columns[i]], df[columns[j]])
    return results

def shuffle_symm_mat(symm_mat, n_shuffles=100):
    """
    Randomly permutes the rows and columns of a symmetric matrix multiple times.

    Parameters:
    symm_mat (numpy.ndarray): The symmetric matrix to shuffle. Must be square.
    n_shuffles (int): The number of shuffled matrices to generate.

    Returns:
    List[numpy.ndarray]: List of shuffled symmetric matrices.
    """
    if symm_mat.shape[0] != symm_mat.shape[1]:
        raise ValueError("Input matrix must be square")

    shuffled_matrices = []
    for _ in range(n_shuffles):
        orders = np.random.permutation(np.arange(len(symm_mat)))
        shuffled_matrices.append(symm_mat[orders][:,orders]) # rows and columns
    return shuffled_matrices

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
digitize_rdm = digitize_matrix # old name

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

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

def remove_diag(arr):
    arr = arr.copy()
    np.fill_diagonal(arr, np.nan)
    return arr[~np.isnan(arr)].reshape(arr.shape[0], arr.shape[1] - 1)

def sort_symm_mat(mat, vec):
    
    ''' Sorts rows/columns of a symmetrical matrix according to a separate vector '''
    sort = vec.argsort()
    return mat[sort][:, sort]

def shuffle_symm_mat(symm_mat):
    orders = np.random.permutation(np.arange(len(symm_mat)))
    return symm_mat[orders][:,orders]

def permute_symm_mat(symm_mat, n_perms=1000):
    ''' permute a symmetrical matrix (columns and rows in same way) n_perms times
        returns a list of permuted symmetrical matrices'''
    n = symm_mat.shape[0]
    permuted = []
    for _ in range(n_perms):
        permutation = np.random.permutation(n)
        permuted.append(symm_mat[:, permutation][permutation, :])
    return permuted

def symm_mat_labels_to_vec(labels, upper=True):
    '''
        lower or upper triangle label pairs from symm matrices, excl. diagonal
    '''
    n = len(labels)
    mat = pd.DataFrame(np.zeros((n, n)))
    for r in range(n):
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

def symm_mat_to_ut_vec(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    if isinstance(mat, pd.DataFrame):
        mat = mat.values
    return mat[np.triu_indices(len(mat), k=1)]

def ut_mat_to_symm_mat(mat):
    '''
        go from upper tri matrix to symmetrical matrix
    '''
    for i in range(np.shape(mat)[0]):
        for j in range(i, np.shape(mat)[1]):
            mat[j][i] = mat[i][j]
    return mat

def ut_vec_to_symm_mat(ut_vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat   = ut_vec_to_ut_mat(ut_vec)
    return ut_mat_to_symm_mat(ut_mat)

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

def ut_vec_pw_dist(x, metric='euclidean'):
    x = np.array(x)
    if x.ndim == 1:  x = x.reshape(-1,1)
    return symm_mat_to_ut_vec(pairwise_distances(x, metric=metric))

def get_pw_dist_ut_vec(coords, metric='euclidean'):
    return symm_mat_to_ut_vec(pairwise_distances(coords, metric=metric))

def subset_symm_mat(symm_mat, mask):
    """
        Extracts a subset of a similarity matrix based on a specified mask.

        Parameters:
        symm_mat (np.ndarray): A square similarity matrix from which the subset is to be extracted.
        mask (list or np.ndarray): Contains booleans or indices of the rows/columns to be included in the subset

        Returns:
        np.ndarray: A subset of the similarity matrix corresponding to the specified mask.

        Raises:
        ValueError: If the similarity_matrix is not square or if the mask contains invalid indices.

        Example:
        >>> similarity_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mask = np.array([0, 2])
        >>> subset_similarity_matrix(similarity_matrix, mask)
        array([[1, 3],
            [7, 9]])
    """

    # Check if the similarity matrix is square
    if symm_mat.shape[0] != symm_mat.shape[1]:
        raise ValueError("The similarity matrix must be square.")

    # If mask is boolean, turn into indices
    if isinstance(mask[0], bool):
        mask = np.where(mask)[0]

    # Check if the mask contains valid indices
    if not np.all(np.isin(mask, range(symm_mat.shape[0]))):
        raise ValueError("The mask contains invalid indices.")

    # Create the subset similarity matrix
    return np.array([[symm_mat[i, j] for j in mask] for i in mask])


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

def print_df(df):
    print(df.to_markdown())


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


def circle_shift(data, axis=0):
    """ Performs a random circular shift of data """
    rand_int = np.random.randint(1, len(data)) # random integer between 1 and len(data)
    return np.roll(data, rand_int, axis=axis)

def bootstrap_symm_matrix(matrix, random_state=None):
    ''' 
        shuffles similarity matrix for bootstrapping analysis
        shuffle rows and columns, with replacement
        based on recommendation by Chen et al., 2016 
    '''
    rows, cols = matrix.shape
    assert rows == cols, "Matrix must be square"    
    s = check_random_state(random_state)
    bs = sorted(s.choice(np.arange(rows), size=rows, replace=True))
    return matrix[bs, :][:, bs] 

def shuffle_symm_matrix(matrix, random_state=None):
    ''' 
        shuffle the row and columns of symmetrical matrix in same way 
    '''
    rows, cols = matrix.shape
    assert rows == cols, "Matrix must be square"
    s = check_random_state(random_state) 
    perm = np.arange(rows)
    s.shuffle(perm)
    return matrix[perm, :][:, perm]

def get_combos(arr, k=2):
    """ 
        arr: np.array to get combos from
        r: num of combos
    """
    return list(itertools.combinations(arr, k))

def get_pairs(items):
    return list((set(itertools.compress(items, mask)) for mask in itertools.product(*[[0,1]] * len(items))))[1:]

def get_unique_permutations(seq):
    """
    Yield only unique permutations of seq in an efficient way.

    A python implementation of Knuth's "Algorithm L", also known from the 
    std::next_permutation function of C++, and as the permutation algorithm 
    of Narayana Pandita.
    """

    # Precalculate the indices we'll be iterating over for speed
    i_indices = list(range(len(seq) - 1, -1, -1))
    k_indices = i_indices[1:]

    # The algorithm specifies to start with a sorted version
    seq = sorted(seq)

    while True:
        yield seq

        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # Introducing the slightly unknown python for-else syntax:
            # else is executed only if the break statement was never reached.
            # If this is the case, seq is weakly decreasing, and we're done.
            return

        # Get item from sequence only once, for speed
        k_val = seq[k]

        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break

        # Swap them in the most efficient way
        (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                           # 0 0 1 1 1 1 0 0

        # Reverse the part after but not                           k
        # including k, also efficiently.                     0 0 1 1 0 0 1 1
        seq[k + 1:] = seq[-1:k:-1]

def get_derangements(x):
    return np.array(
        [copy.copy(s)
        for s in get_unique_permutations(x)
        if all(a != b for a, b in zip(s, x))])

def shuffle_array(arr, random_state=None):
    ''' 
        shuffle an array randomly
    '''
    s = check_random_state(random_state) 
    perm = np.arange(len(arr))
    s.shuffle(perm)
    return arr[perm]

def unique_arrays(list_of_arrays):

    '''
    Removes duplicate arrays from a list of arrays 
    '''

    # Convert lists to tuples
    list_of_tuples = [tuple(lst) for lst in list_of_arrays]
    
    # Convert list of tuples to a set to remove duplicates
    unique_tuples = set(list_of_tuples)
    
    # Convert back to list of lists
    return [list(tup) for tup in unique_tuples]


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