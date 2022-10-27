import time
import pandas as pd
import numpy as np

import sklearn
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc, roc_auc_score, matthews_corrcoef, silhouette_score


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


def create_pipeline(estimator, steps=[]):
    ''' create a sklearn pipeline, with list of steps followed by estimator
        pl.get_params().keys(): to see which parameters are available
    '''
    
    pl_dict = {'scaler': StandardScaler(), 
               'selector': VarianceThreshold(),
               'pca': PCA(n_components = 25)}
    
    return Pipeline([(step, pl_dict[step]) for step in steps] + [('estimator', estimator)])


def run_perm_clf(clf, X, y, k=10, cv=StratifiedKFold, permutations=1000):
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
        for i, (train, test) in enumerate(cv(n_splits=k).split(X, y)):
            y_train_ = y[train].copy()
            random.shuffle(y_train_) # break connection between data & labels in training
            clf_clone = clone(clf)
            clf_clone.fit(X[train], y_train_)
            acc[i] = clf_clone.score(X[test], y[test])
        perm_accs[n] = np.mean(acc) # mean acc across folds
    return perm_accs


def run_clf(clf, X, y, cv=6, standardize=False):

    # TODO: accept pipelines...
    
    eval_df  = pd.DataFrame()
    pred_dfs = []

    if isinstance(cv, int): # stratified to balance each fold by class (i.e., character) 
        folds = StratifiedKFold(n_splits=cv, random_state=76, shuffle=True).split(X, y)
    elif (isinstance(cv, str)) & (cv=='loo'): # leave one out
        folds = LeaveOneOut().split(X, y)
    else: # iterator
        folds = cv

    # drop voxels with 0/nan values
    X = VarianceThreshold().fit_transform(X) 

    # cross-validated decoding
    for k, (train, test) in enumerate(folds):

        # if standardizing, fit a scaling model on training folds
        if standardize: 
            scaler  = StandardScaler().fit(X[train])     
            X_train = scaler.transform(X[train])
            X_test  = scaler.transform(X[test])
        else:
            X_train = X[train].copy()
            X_test  = X[test].copy()

        # fit classifier on training folds
        decoder = clone(clf)
        decoder.fit(X_train, y[train]) 

        # predict on held out fold
        y_preds = decoder.predict(X_test)
        pred_df = pd.DataFrame(np.vstack([test, y_preds, y[test], (y_preds == y[test]) * 1]).T,
                               columns=['ix', 'predicted', 'actual', 'correct'])
        pred_df.insert(0, 'split', k)

        # evaluate performance
        eval_df.loc[k, 'split']    = k
        eval_df.loc[k, 'accuracy'] = decoder.score(X_test, y[test])
        # eval_df.loc[k, 'balanced_acc'] = balanced_accuracy_score(y[test], y_preds)
        # eval_df.loc[k, 'f1'] = f1_score(y[test], y_preds, average='weighted')
        # eval_df.loc[k, 'phi'] = matthews_corrcoef(y[test], y_preds)
        # eval_df.loc[k, 'dice'] = sp.spatial.distance.dice(y[test], y_preds)

        # get probabilities
        if hasattr(decoder, 'predict_proba'): 
            y_probas = decoder.predict_proba(X_test)
            for p, y_probas_ in enumerate(y_probas.T):
                pred_df[f'probas_class{p+1:02d}'] = y_probas_

        pred_dfs.append(pred_df)
    pred_df = pd.concat(pred_dfs)
    pred_df.reset_index(inplace=True, drop=True)

    # output dict
    clf_dict = {'cross-validation': cv, 
                'predictions': pred_df,
                'evaluation': eval_df} 
    
    return clf_dict
