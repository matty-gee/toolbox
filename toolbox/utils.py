import time
import re
import pandas as pd
import numpy as np
from functools import reduce
from itertools import compress, product, combinations 
from io import StringIO
from pathlib import Path
from itertools import groupby

def all_equal(iterable):
    ''' check if all items in a list are identical '''
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def all_equal_arrays(L):
    ''' check if difference in List L along axis of stacking == 0  '''
    return (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()

########################################################################################################
# text parsing
########################################################################################################

def remove_nontext(string):
    return re.sub(r'\W+', '', string)


def get_strings_matching_substrings(strings, substrings):
    ''' return strings in list that partially match any substring '''
    matches = [any(ss in s for ss in substrings) for s in strings]
    return list(np.array(strings)[matches])


def is_numeric(input_str):
    ''' check if all characters in a string are numeric '''
    return all(char.isdigit() for char in input_str)

########################################################################################################
# for dataframes
########################################################################################################

def merge_dfs(df_list, on='sub_id', how='inner'):
    return reduce(lambda x, y: pd.merge(x, y, on=on, how=how), df_list)


def move_cols_to_front(df, cols):
    return df[cols + [c for c in df if c not in cols]]


def is_finite(array):
    return array[np.isfinite(array)]


def reset_sort(df):
    return df.sort_values(by=['sub_id'], ascending=False).reset_index(drop=True)

########################################################################################################
# combos, permutations etc...
########################################################################################################

def combos(arr, k):
    """ 
        arr: np.array to get combos from
        r: num of combos
    """
    return list(combinations(arr, k))


def pairs(items): 
    return list((set(compress(items, mask)) for mask in product(*[[0,1]] * len(items))))[1:]

########################################################################################################
## performance
########################################################################################################

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