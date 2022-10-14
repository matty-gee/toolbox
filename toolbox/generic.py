#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
    general purpose functions - e.g., functions for finding files, manipulating dataframes
    fairly self explanatory
'''

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
# finding files, etc
########################################################################################################

def find_files(path, file_pattern):
    '''
        Find files that follow a pattern, using pathlib's Path 

        Arguments
        ---------
        path : str
            Path name where to search
        file_pattern : str
            Can include a wildcard

        Returns
        -------
        list of Path objects 
            _description_

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    return list(Path(path).glob(file_pattern))

class DisplayablePath(object):
    
    '''
        make a neat path display
        taken from: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    '''
    
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True
    
    @classmethod
    def write_txt(self, paths=None):
        '''
            by Matthew Schafer
            write a text file w/ the structure
        '''
        if paths is None:
            raise Exception('Please run DisplayablePath.make_tree() first and then run this method with the resulting paths')
        dir_struct = [path.displayable() for path in paths]
        with open('directory_structure.txt', 'w') as f:
            f.write('\n'.join(dir_struct))  
    
    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        '''
            to display the paths
        '''
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))
    
def is_not_hidden(path):
    ''' skip hidden files) '''
    return not path.name.startswith(".")

def is_not(path, ignore=(".", "__pycache__")):
    ''' skip paths that start w/ any of the 'ignore' patterns
    '''
    return not path.name.startswith(ignore)    

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
# reading & saving files
########################################################################################################

from six.moves import cPickle as pickle 
import json
from json import JSONEncoder

# PICKLED FILES
def pickle_file(file_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f)
    f.close()

def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file

# JSON FILES
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=json_encoder)

class json_encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return JSONEncoder.default(self, obj)

def read_excel(path: str, sheet_name=0, index_col=None) -> pd.DataFrame:
    '''
        Turn excel into csv in memory to speed up reading (~50% speedup)
    '''
    from xlsx2csv import Xlsx2csv # inside here so only is loaded if needed (not all machines can install this package apparently..)
    buffer = StringIO()
    Xlsx2csv(path, outputencoding="utf-8", sheet_name=sheet_name).convert(buffer)
    buffer.seek(0)
    df = pd.read_csv(buffer, index_col=index_col)
    return df

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