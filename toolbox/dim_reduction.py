from datetime import date
import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2, pearsonr

from factor_analyzer import ( 
    FactorAnalyzer, ConfirmatoryFactorAnalyzer, 
    ModelSpecificationParser)
from factor_analyzer.factor_analyzer import calculate_kmo

from sklearn.manifold import MDS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import CCA, PLSRegression, PLSCanonical
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, cross_val_predict, 
    StratifiedShuffleSplit, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, pairwise_distances

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# my modules
from general_utils import get_unique

#--------------------------------------------------------------------------------------------
# linear decomposition: e.g., factor analysis, principle components analysis
#--------------------------------------------------------------------------------------------

# TODO: refactor w/ more sklearn like api

class FactorAnalysis(object):

    '''[By Matthew Schafer; github: @matty-gee; 2020ish]'''
        
    def __init__(self, algorithm=None):
        '''
            Initialize a dimensionality reduction object

            Arguments
            ---------
            algorithm : str
                Which dimensionality reduction algorithm to use

            Raises
            ------
            ValueError : 
                If algorithm isn't in accepted algorithm list
        '''
        self.alg = algorithm or 'efa'     
        if self.alg not in ['efa', 'pca']:
            raise ValueError("Select a dimensionality reduction algorithm from this list: 'efa', 'pca'")
        self.fitted = False  
        self.preprocessed = False
        
    #-----------------------------------------------------------------------
    # Preprocessing
    #-----------------------------------------------------------------------
    
    def preprocess(self, df):
        ''' 
            Some simple preprocesssing of dataframe:
                1. Replace nans w/ 0s
                2. Ensure data are floats
                3. Output data shape: (observations/subjects, features/variables matrix)
        '''
        n_features_raw = df.shape[1]
        
        df = df.fillna(value=0) # nans -> 0s 
        df = df.loc[:, np.sum(df.applymap(lambda x: isinstance(x, (int, float))), axis=0) == df.shape[0]] # remove non-numeric features        
        df = df.loc[:, df.var() > 0] # remove features with no variance

        # assign data, observation, feature names
        self.X = df.values
        self.observations = df.index.values
        self.n_observations = len(self.observations)
        self.features = df.columns.values
        self.n_features = len(self.features)
        print(f'Preprocessing removed {n_features_raw-self.n_features} features, leaving {self.n_observations} observations over {self.n_features} features')
        
        if self.corr_matrix is not None:
            self.corr_matrix.index = self.corr_matrix.columns # ensure index is filled in
            self.corr_matrix = self.corr_matrix.loc[self.features, self.features] # reorganize so same column order as X
            
        self.feature_groups, self.feature_groups_bins = self.get_feature_groups(self.features)    

        self.preprocessed = True

    def get_feature_groups(self, features):
        # get groups of features (if they exist) - maybe optionally pass in the feature groupings...
        feature_groups = []
        for feat in features:
            if any((c not in set('0123456789')) for c in feat.split('_')[1]): # so we can have subscales easily
                # ultimately plot w/ flexible coloring - e.g., if share first prefix then make colors in same family...
                feature_groups.append(('_').join(feat.split('_')[:2]))
            else:
                feature_groups.append(feat.split('_')[0])
        feature_groups = pd.unique(feature_groups)
        feature_groups_bins = {}
        for group in feature_groups:
            ixs = [c for c, col in enumerate(features) if col.startswith(group)]
            feature_groups_bins[group] = (np.min(ixs),np.max(ixs))      
        return feature_groups, feature_groups_bins
        
    def impute_missing(self):
        """ Impute missing data with mean of the feature/column """
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X = imp.fit(self.X).transform(self.X)    
        self.auto_preprocess() 
        
    def standardize(self, X):
        return (X - np.nanmean(X)) / np.nanstd(X)
     
    #-----------------------------------------------------------------------
    # Assumptions/Validity
    #-----------------------------------------------------------------------

    def kmo_test(self):
        """  Meyer-Kaiser-Olkin test: is data suitable for factor analysis or not """
        return calculate_kmo(self.X)[1]
        
    def bartletts_test(self):
        """
            ADAPTED FROM FACTOR ANALYZER PACKAGE TO TAKE CORRELATION MATRIX

            Bartlett’s Test of Sphericity: test the hypothesis that the correlation matrix is equal to the identity matrix 

            H0: The matrix of population correlations is equal to I.
            H1: The matrix of population correlations is not equal to I.

            The formula for Bartlett's Sphericity test is:

            .. math:: -1 * (n - 1 - ((2p + 5) / 6)) * ln(det(R))

            Where R det(R) is the determinant of the correlation matrix,
            and p is the number of variables.

            Parameters
            ----------
            x : array-like
                The array from which to calculate sphericity.

            Returns
            -------
            statistic : float
                The chi-square value.
            p_value : float
                The associated p-value for the test.
            """

        if self.is_corr_matrix:
            n = int(input('What is n for this correlation matrix?'))
            p = self.X.shape[1] # num of features (columns)
            x_corr = self.X
        else:
            n, p = self.X.shape # num of obs (rows), num of features (columns)
            x_corr = np.corrcoef(self.X)
        corr_det  = np.linalg.det(x_corr)
        statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
        dof       = p * (p - 1) / 2
        return chi2.sf(statistic, dof) # p-value
    
    def validate(self):
        """ Minimal tests to ensure pca/fa can be performed """
        print('\nSome tests to determine if FA should be performed:')
        if not self.is_corr_matrix:
            # not sure how to adapt this to heterogeneous correlation matrix
            kmo_model = self.kmo_test() 
            if kmo_model > .60:
                print(f'- KMO test: {kmo_model:3f} (>.60) - continue fitting fa/pca') 
            else: 
                print(f'- KMO test: {kmo_model:3f} (<.60) - reconsider fa/pca') 
        p = self.bartletts_test()
        if p < .05:
            print(f'- Reject null for Barletts test of sphericity: p={p:3f} - continue fitting')
        else: 
            print(f'- Cant reject null for Barletts test of sphericity: p={p:3f} - reconsider')
            
    #-----------------------------------------------------------------------
    # Dimensionality reduction
    #-----------------------------------------------------------------------
    
    def fit_transform(self, df, num_comps=3, corr_matrix=None, rotation=None, cfa_dict=None):
        '''
        '''
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
            self.is_corr_matrix = True
        else: 
            self.corr_matrix = None
            self.is_corr_matrix = False
        self.num_comps = num_comps
        self.rotation = rotation
        self.preprocess(df)
        
        if self.alg == 'efa':   self.efa()
        elif self.alg == 'pca': self.pca()
        elif self.alg == 'cfa': self.cfa(cfa_dict)
        self.fitted = True
    
    def pca(self): 
        """ 
            Principle Components Analysis
            Goal: restructure data into reduced rank representation, w/ max variance explained (data compression)
            Algorithm: 
                1 - standardize the data: mean center, scale to unit variance 
                2 - compute pca, which reduces dimensions while preserving covariance in dataset
                        - PCA takes linear combo of original vars/features
                        - each linear combo explains most variance it can of the remaining unexplained variance 
                        - linear combos are orthogonal/uncorrelated 
        """
        from sklearn.decomposition import PCA

        # create, fit & transform 
        self.model = PCA(n_components=self.num_comps) 
        self.X_reduced = self.model.fit_transform(self.standardize(self.X)) 

        # eigvecs/components: directions of maximum variance in data - correlations of features w/ components; org. by explained varianc
        self.eigvecs = self.model.components_ # components x features matrix
        
        # eigenvalues & cumulative variance explained
        self.eigvals = self.model.explained_variance_
        self.exp_var = self.model.explained_variance_ratio_
        self.cum_var = np.cumsum(self.exp_var)
        
        # loadings
        self.loadings = self.eigvecs.T * np.sqrt(self.eigvals) # loading matrix (features x factors): transformation of latent to observed variables 
        self.get_mean_loadings()
        
    def efa(self):
        """ 
            Exploratory Factor Analysis
            Goal: uncover underlying structure
            ....
                - maximum likelihood method
                - optional: orthogonal/oblique rotation
        """
        # create model
        # - maximum likelihood method for getting fit
        # - rotation:
            # orthogonal rotations ensure the factors are uncorrelated - simple, interpretable
                # varimax: Few large and lots of close to 0 loadings
                # quartimax: Forces a given variable to correlate highly with one factor.  Makes large loadings extremely large and small loadings extremely small. 
                # oblimax:  When the assumption of homogeneously distributed error cannot be applied, but may be replaced by a principle of maximum kurtosis
                # equamax: Attempted improvements on varimax.  The rotation adjusts to the number of factors being rotated, resulting in more uniformly distributed set of factors vs. varimax.  Creates less generic factors. 
            # oblique rotations allow factors to correlate - goal is to explain as much data as possible
                # promax, oblimin, quartimin
        # - is_corr_matrix: can pass in already computed corr_matrix as X
        self.model = FactorAnalyzer(n_factors=self.num_comps, method='ml', rotation=self.rotation, is_corr_matrix=self.is_corr_matrix) 
    
        # fit the model
        if self.is_corr_matrix: 
            self.model.fit(self.corr_matrix)
        else: 
            self.model.fit(self.X)
            self.corr_matrix = np.corrcoef(np.transpose(self.X))   

        # eigvecs/components: directions of maximum variance in data - correlations of features w/ components; sorted by explained variance
        # - doesnt output in factorAnalzer so compute it directly using scipy
        _, eigvecs   = scipy.linalg.eigh(self.corr_matrix) 
        self.eigvecs = eigvecs[:, ::-1][:, :self.num_comps] # eigvecs/components x features matrix
        
        # eigenvalues & cumulative variance explained
        self.eigvals_orig, self.eigvals = self.model.get_eigenvalues()
        _, self.exp_var, self.cum_var   = self.model.get_factor_variance()

        # uniqueness
        self.uniqueness = self.model.get_uniquenesses()
        
        # communalities after factor extraction: 1d feature vector
        # - for each feature, sum of the squares of its loading on each factor
        self.communalities = pd.DataFrame(self.model.get_communalities(), index=self.features, columns=['sum-of-squares_loadings']) 
        
        # loadings 
        # - features x factors: transformation of latent to observed variables (correlations of features w/ components)
        self.loadings = self.model.loadings_
        self.get_mean_loadings()
        
        # transform data: coordinates of each observation on components/dimensions
        self.X_reduced, self.weights = self.transform(self.X, return_weights=True)
            
    def transform(self, X, return_weights=False):

        # transform data: coordinates of each observation on components/dimensions
        # - calculation: X_reduced = np.dot(((X-X_mean)/X_std), np.linalg.solve(corrmat, loadings))
        
        if self.alg == 'efa' and self.is_corr_matrix:
            weights   = np.linalg.solve(self.corr_matrix, self.loadings)
            X_reduced = np.dot(self.standardize(X), weights)
        elif self.alg in ['efa', 'pca']:
            X_reduced = self.model.transform(X)
        return (X_reduced, weights) if return_weights else X_reduced
    
    def efa_partial(self, thresh=75, plot=False):
        ''' fit an exploratory factor analysis to a subset of the features '''
        if self.fitted:
            
            # organize each item by top factor it loads onto
            abs_loading = pd.DataFrame(np.abs(self.loadings), 
                                       index=[self.features], 
                                       columns=[f'factor_{f}' for f in np.arange(1, self.num_comps+1)])
            top_factors = abs_loading.idxmax(axis=1).values

            # keep the items that are above some threshold for their top factor
            partial_items, items_incl = [], []
            for f in np.arange(0, self.num_comps):
                items_mask = top_factors == f'factor_{f + 1}'
                factor_items = self.features[items_mask]
                factor_loadings = self.loadings[:,f][items_mask]
                items = list(factor_items[np.abs(factor_loadings) > np.percentile(np.abs(factor_loadings), thresh)]) # threshold
                partial_items.append(items)
                items_incl += items
            incl_mask = np.isin(self.features, items_incl)

            # transform data with this subset of items
            X_ = self.X[:, incl_mask]
            X_scaled_ = (X_ - np.nanmean(X_, axis=0)) / np.nanstd(X_, axis=0)
            corr_matrix_ = pd.DataFrame(self.corr_matrix, columns=self.features).iloc[incl_mask, incl_mask]
            weights_ = np.linalg.solve(corr_matrix_, self.loadings[incl_mask, :])
            self.X_partial_reduced = np.dot(X_scaled_, weights_)

            # plot the correlation between original transform & this reducd one
            if plot:
                fig, axs = plt.subplots(1, self.num_comps, figsize=(5*self.num_comps, 4))
                for f in np.arange(0, self.num_comps):
                    sns.regplot(x=self.X_partial_reduced[:,f], y=self.X_reduced[:,f], ax=axs[f],
                                scatter_kws = {'color': 'purple', 'alpha': 0.3},
                                line_kws = {'color':'purple', 'alpha': 0.3, 'lw':3})
                    axs[f].set_title(f'factor {f + 1}')

            return partial_items
        else:
            print('The EFA model must be fit first')

    def cfa(self, cfa_dict):
        ''' Confirmatory factor analysis
            TODO: needs a lot of work to make functional
        '''
        from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
        
        cfa_spec = ModelSpecificationParser.parse_model_specification_from_dict(self.X, cfa_dict)
        self.model = ConfirmatoryFactorAnalyzer(cfa_spec, disp=False, max_iter=200, tol=None) 
        self.model.fit(self.X)
        self.loadings = self.model.loadings_ 
        self.implied_cov = self.model.get_model_implied_cov()

    # def ica(self):
    #     """
    #         Independent Components Analysis: 
    #         Goal: separate putative source signals from a mixed, observed signal (separate data) 
    #             assumes: observed data are a linear combo of independent components
    #             notes:
    #             - ICA excels at problems PCA will often fail at if the sources are non-gaussian
    #     """
    #     from sklearn.decomposition import FastICA
        
    #     # create ica object
    #     ica = FastICA(n_components=self.num_comps, random_state=0)
        
    #     # fit model & apply unmixing matrix to separate sources
    #     self.X_reduced = ica.fit_transform(self.X) # reconstructed signals
    #     self.mixing_matrix = ica.mixing_ 
        
    #     # reconstruct - how much info is lost?
    #     self.X_restored = ica.inverse_transform(self.X_reduced)     

    #-----------------------------------------------------------------------
    # Plotting
    #-----------------------------------------------------------------------

    def get_mean_loadings(self):
        """ """
        self.mean_loadings = pd.DataFrame([np.mean(self.loadings[ixs[0]:ixs[1], :], axis=0) for fg, ixs in self.feature_groups_bins.items()], 
                                          columns=[f'component {c}' for c in np.arange(1, self.loadings.shape[1]+1)])
        self.mean_loadings.insert(0, 'feature_group', [fg for fg, ixs in self.feature_groups_bins.items()])
    
    def get_corrmats(self, refit=False):
        """
            1. Compute feature x feature correlation matrix
            2. Compute reproduced feature x feature correlation matrix
            3. Compute difference between 1 & 2 (each element should be ~0)
        """
        # check if we have it already
        if self.corr_matrix is None:
            self.corr_matrix = np.corrcoef(np.transpose(self.X))
        self.reproduced_corr_matrix = np.matmul(self.loadings, self.loadings.T)
        self.residuals = self.corr_matrix - self.reproduced_corr_matrix

    def plot_corrmat(self, plot='features', cmap="plasma", refit=False, figsize=(10,10)):
        """ use perceptually uniform sequential cmaps: 'viridis', 'plasma', 'inferno', 'magma', 'cividis' """
        
        if refit: 
            X = self.X[:, self.partial_incl_mask]
            title = 'Correlation matrix - subsetted items'
            corr_matrix = self.corr_matrix_partial
            features = self.features[self.partial_incl_mask]
            feature_groups, feature_groups_bins = get_feature_groups(self, features)  
        else: 
            self.get_corrmats()
            X = self.X
            features = self.features
            feature_groups = self.feature_groups
            feature_groups_bins = self.feature_groups_bins
            if plot == 'features':
                title = 'Correlation matrix'
                corr_matrix = self.corr_matrix
            elif plot == 'reproduced':
                title = 'Reproduced correlation matrix'
                corr_matrix = self.reproduced_corr_matrix
            elif plot == 'residuals':
                title = 'Residuals matrix'
                corr_matrix = self.residuals
            else:
                raise ValueError('What matrix do you want to plot? Options: features, reproduced, residuals')

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        cax = ax.imshow(corr_matrix, cmap=cmap, vmax=1, vmin=-1)

        # x-axis: individual features
        ax.set_xticks(np.arange(X.shape[1])[::4])
        ax.set_xticklabels(features[::4], rotation=90, fontsize=11)

        # y-axis 
        if len(feature_groups) < len(features)/5: # if there are groupings to the features...
            ax.set_yticks([np.mean(ixs) for fg, ixs in feature_groups_bins.items()])
            ax.set_yticklabels(feature_groups, fontsize=20)
            # ax.vlines([ixs[0] for fg, ixs in self.feature_groups_bins.items()], 0, self.X.shape[1]-1)
            # ax.hlines([ixs[0] for fg, ixs in self.feature_groups_bins.items()], 0, self.X.shape[1]-1)
        else:
            ax.set_yticks(np.arange(X.shape[1])[::4])
            ax.set_yticklabels(features[::4], fontsize=11)

        plt.title(title, fontsize=25)
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
        return fig
        plt.show()
         
    def plot_cumvar(self, figsize=(15,7.5)):
        """ Plot the cumulative explained variance of factor loadings """  
        fig, ax = plt.subplots(figsize=figsize)
        ax = plt.plot(self.cum_var*100, linewidth=2)
        plt.grid()
        plt.xticks(range(self.num_comps), range(1, self.num_comps+1))
        plt.xlabel('Factor/Component')
        plt.ylabel('Variance (%)')
        plt.title('Cumulative Variance Explained')
        plt.show()
    
    def plot_scree(self, figsize=(10,7.5)):
        """ Plot the eigenvalues against the factors/components """ 
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(self.eigvals, '-', linewidth=4)
        # plt.xticks(range(self.num_comps), range(1,self.num_comps+1))
        plt.xlabel('Factor/Component')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot')
        plt.show()
        
    def plot_eigvec_3d(self):
        """ Plot first three FA/PCA dimensions """ 
        if self.num_comps < 3:
            raise ValueError('Less than 3 dimensions to plot')
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], self.X_reduced[:, 2], 
                   cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])
        plt.show()
        
    def plot_loadings(self, num_comps=3, colors=None, component_names=None, labels=None, figsize=(20,5)):
        """ plot the component loadings barplot """

        # add errors if its for wrong dim reduction method
        num_comps = max(num_comps, self.num_comps)

        # color different questionnaires
        if colors is None:
            colors = ['red', 'blue', 'purple', 'green', 'lavender', 'grey', 'fuchsia', 'orange', 'dodgerblue', 
                     'yellow', 'orchid', 'indigo', 'aqua','palegreen', 'silver', 'plum', 'fuchsia', 'coral',
                     'gold', 'pink','slategray', 'forestgreen','peachpuff','honeydew','brown','olivedrab',
                     'darkturquoise', 'tan', 'springgreen', 'mintcream','navajowhite','chocolate','lightblue','chartreuse',
                     'lime','yellowgreen','khaki','gold','teal','tomato']
        colors_ = pd.DataFrame([None]*len(self.features))
        for f,(fg, ixs) in enumerate(self.feature_groups_bins.items()):
            colors_[ixs[0]:ixs[1]+1] = colors[f]
        colors_ = [color[0] for color in colors_.values]
        self.item_colors = colors_

        # for legend
        if labels is None: labels = [fg for fg, ixs in self.feature_groups_bins.items()]
        patches = [mpatches.Patch(facecolor=inst[0], edgecolor='black', 
                                  label=inst[1]) for inst in zip(colors, labels)]

        # separate plot for each component
        if component_names is None: component_names = [f'Component {c+1}' for c in range(num_comps)]
        ymax = np.round(np.max(self.loadings),1)+.05
        for c in range(num_comps):
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(np.arange(len(self.loadings[:, c])), self.loadings[:, c], color=colors_, edgecolor='black')
            ax.legend(handles=patches, loc='upper right', 
                      frameon=False, bbox_to_anchor=(1.15, 1), borderaxespad=0, prop={'size': 14})
            plt.xlim(-5,len(self.features)+5)
            plt.ylim(-1,1)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f'{component_names[c]} ({np.round(self.exp_var[c]*100, 2)} % explained)', fontsize=20)
            plt.show()
                

            # ## HEATMAPS
            # # plot by feature
            # fig, ax = plt.subplots(figsize=(30,30))
            # vmax = np.abs(self.loadings).max()
            # ax.imshow(np.array(self.loadings.T), cmap="RdBu_r", vmax=vmax, vmin=-vmax)

            # if len(self.feature_groups) < len(self.features)/5:
            #     ax.set_xticks([np.mean((ixs[0],ixs[1])) for _,ixs in self.feature_groups_bins.items()])
            #     ax.set_xticklabels(self.feature_groups, rotation=90)
            #     ax.vlines([ixs[0]-.5 for fg, ixs in self.feature_groups_bins.items()], 0, self.num_comps)
            # else:
            #     ax.set_xticks(np.arange(len(self.features))[0::4])
            #     ax.set_xticklabels(self.features[0::4], rotation=90)

            # ax.set_yticks(np.arange(self.num_comps))
            # ax.set_yticklabels(np.arange(self.num_comps)+1)
            # ax.set_title('Component loadings plot', fontsize=20)
            # plt.show()

            # # if it makes sense to plot averaged across feature too
            # if len(self.feature_groups) < len(self.features)/5:
            #     fig, ax = plt.subplots(figsize=(10, 10))
            #     vmax = np.abs(self.mean_loadings.iloc[:,1:]).max().max()
            #     ax.imshow(self.mean_loadings.iloc[:,1:].T, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
            #     plt.colorbar(ax.get_children()[-2], ax=ax, orientation='vertical', shrink=.25, label='Loadings')
            #     ax.set_xticks(np.arange(len(self.feature_groups)))
            #     ax.set_xticklabels(self.feature_groups, rotation=90)
            #     ax.set_yticks(np.arange(self.num_comps))
            #     ax.set_yticklabels(np.arange(self.num_comps)+1)
            #     ax.set_title('Loadings averaged across feature groups', fontsize=20)
            #     plt.show()
            
    def plot_components(self, figsize=(20,20)):
        '''
            plot the components/eigenvectors: directions of maxmimum variance
        '''
        # plot by feature

        fig, ax = plt.subplots(figsize=figsize+figsize/2)
        vmax = np.abs(self.eigvecs).max()
        ax.imshow(np.array(self.eigvecs).T, cmap="RdBu_r", vmax=vmax, vmin=-vmax)

        if len(self.feature_groups) < len(self.features)/5:
            ax.set_xticks([np.mean((ixs[0],ixs[1])) for _,ixs in self.feature_groups_bins.items()])
            ax.set_xticklabels(self.feature_groups, rotation=90)
            
            ax.vlines([ixs[0]-.5 for fg, ixs in self.feature_groups_bins.items()], -.5, self.num_comps-.5)
        else:
            ax.set_xticks(np.arange(len(self.features))[0::4])
            ax.set_xticklabels(self.features[0::4], rotation=90)

        ax.set_yticks(np.arange(self.num_comps))
        ax.set_yticklabels(np.arange(self.num_comps)+1)
        ax.set_title('Eigenvectors/components plot', fontsize=20)
        plt.show()

        # averaged plot
        if len(self.feature_groups) < len(self.features)/5:
            
            mean_components = pd.DataFrame([np.mean(self.eigvecs[ixs[0]:ixs[1],:],axis=0) for fg, ixs in self.feature_groups_bins.items()], 
                                            columns=['component ' + str(c) for c in np.arange(1, self.eigvecs.shape[1]+1)])
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(mean_components.T, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
            plt.colorbar(ax.get_children()[-2], ax=ax, orientation='vertical',shrink=.5, label='Eigenvector magnitudes')
            
            ax.set_xticks(np.arange(len(self.feature_groups)))
            ax.set_xticklabels(self.feature_groups, rotation=90)
            ax.set_yticks(np.arange(self.num_comps))
            ax.set_yticklabels(np.arange(self.num_comps)+1)
            ax.set_title('Eigenvectors/components averaged across feature groups', fontsize=20)
            plt.show()

            mean_components.insert(0, 'feature_group', [fg for fg, ixs in self.feature_groups_bins.items()])

        self.mean_components = mean_components
            
    def output(self):
        return {'X': self.X, 'corr_matrix': self.corr_matrix, 'features': self.features, 
                'loadings': self.loadings, 'weights': self.weights, 'eigvals': self.eigvals}


def plot_pca_2d(X):
    pca = PCA(n_components=2).fit(X)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i}",
            linewidth=5,
            color=f"C{i + 4}")
    plt.gca().set(
        aspect="equal",
        title="Principal components",
        xlabel="first feature",
        ylabel="second feature",
    )
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------------------
# linear cross decomposition: e.g., partial least squares, canonical correlation analysis
# TODO add uniqueness scores etc like efa
# TODO add biplots, rotations etc...
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
# nonlinear decomposition
#--------------------------------------------------------------------------------------------

# MDS specific
def calc_rawstress(dsm, embedding):
    ''' 
        Calculate raw stress for MDS
        dsm = orig. dissimilarity matrix
        embedding = embedding from MDS
    '''
    emb_dsm = pairwise_distances(embedding, metric='euclidean') # distances betweeen mds embedding points
    return 0.5 * np.sum((emb_dsm - dsm) ** 2)

def convert_rawstress_to_kruskal(dsm, raw_stress):
    ''' 
        a scaled version of the raw stress
        dm = orig. dissimilarity matrix
        - raw stress can be misleading b/c it depends on normalization of disparities
        - kruskal stress is value in range [0,1], where smaller is better
        - divide demoninator by two to make it only half of the symmetric matrix
    '''
    return np.sqrt(raw_stress / (0.5 * np.sum(dsm ** 2)))

