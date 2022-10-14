import time
import random
import pandas as pd
import numpy as np

import sklearn
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit

## classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

class ClassifierGridSearch(object):

    '''
        Runs grid search over dictionaries containing classifiers and parameters to search
        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    def __init__(self, classifier_dict, params_dict):
        """
            Accepts a dictionary of classifiers and parameter grids
        """
        self.classifier_dict = classifier_dict
        self.params_dict = params_dict
    
    def search(self, X, y, **grid_kwargs):
        """
            search hyperparameter space
            kwargs: sklearn args - scoring metric, cv, n_folds, etc
        """
        self.X = X
        self.y = y
        self.grid_searches = {} # to output

        for c, (self.name, classifier) in enumerate(self.classifier_dict.items()): 
            start_time = time.time()
            print('Running GridSearchCV for %s.' % self.name)
            search_params = self.params_dict[self.name] # get the parameters grid
            grid_search = GridSearchCV(classifier, search_params, **grid_kwargs) # create object
            grid_search.fit(self.X, self.y) # fit
            self.grid_searches[self.name] = grid_search # create a grid search dict
            run_time = np.round(time.time() - start_time, 3)
            # print("Time to run %s: %s seconds " % (self.name, run_time, '\n'))
        print('Done.')

    def summary(self, sort_by='mean_test_score'):
        """
            Summarize grid search results
        """
        # get results
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)

        # turn results into dataframe    
        df = pd.concat(frames)
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]

        return df

def permutation_clf_cv(clf, X, y, k=10, permutations=1000):
    '''
        Run permuted classification with cross-validation

        Arguments
        ---------
        clf : object
            classifier object
        X : array or pd.DataFrame
            X values
        y : array or pd.Series
            y values
        k : int (optional) 
            How many folds to run within each permutation.
            Default: 10
        permutations : int (optional)
            How many permutations to run. 
            Default: 1000

        Returns
        -------
        array 
            mean accuracies across all permutations

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    import random
    perm_accs = np.zeros(permutations)
    for n in range(permutations): 
        acc = np.zeros(permutations)
        cv  = StratifiedKFold(n_splits=k)  
        for i, (train, test) in enumerate(cv.split(X, y)):
            y_train_ = y[train].copy()
            random.shuffle(y_train_) # break connection between data & labels in training
            clf_clone = clone(clf)
            clf_clone.fit(X[train], y_train_)
            acc[i] = clf_clone.score(X[test], y[test])
        perm_accs[n] = np.mean(acc) # mean acc across folds
    return perm_accs

def run_classifier(clf, X, y, scale=False, cv_splits=6):
    '''
        Run cross-validated classification 

        Arguments
        ---------
        clf : object
            classifier object
        X : array or pd.DataFrame
            X values
        y : array or pd.Series
            y values
        scale : bool (optional)
            Whether to standardize (within each fold) 
            Default: False
        cv_splits : int or 'loo' or iterator (optional)
            int : perform stratified k-fold cv
            'loo' : perform leave one out cv
            iterator : performn cv with specified indices
            Default: 6

        Returns
        -------
        pd.DataFrame 
            dataframe that summarizes accuracies

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    # TODO: add dummy accuracy values... make permutations optional too?
    # dummy_clf = sklearn.dummy.DummyClassifier(strategy='most_frequent')

    # cross-validator
    if isinstance(cv_splits, int):
        # stratified to balance each fold by class (i.e., character) 
        folds = StratifiedKFold(n_splits=cv_splits, random_state=22, shuffle=True).split(X, y)
    elif (isinstance(cv_splits, str)) & (cv_splits=='loo'): 
        folds = LeaveOneOut().split(X, y)
    else:
        folds = cv_splits
        
    X = VarianceThreshold().fit_transform(X) 
    
    acc_df = pd.DataFrame(columns=['acc_type', 'acc'])
    accs = []
    for k, (train_ix, test_ix) in enumerate(folds):
        
        X_train = X[train_ix].copy()
        X_test  = X[test_ix].copy()

        # preprocessing
        if scale:
            scaler  = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test  = scaler.transform(X_test)

        # classify
        # clf = clone(clf)
        clf.fit(X_train, y[train_ix])
        accs.append(clf.score(X_test, y[test_ix]))
        
    acc = np.round(np.mean(accs), 3)
    acc_df.loc[len(acc_df)+1, :] = ['acc', acc]
    
    return acc_df