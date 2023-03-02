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
from utils import get_unique

#--------------------------------------------------------------------------------------------
# linear decomposition: e.g., factor analysis, principle components analysis
#--------------------------------------------------------------------------------------------

# TODO: refactor w/ more sklearn like api
class Decompose(object):

    '''[By Matthew Schafer; github: @matty-gee; 2020ish]'''
        
    def __init__(self, algorithm):
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
        self.alg = algorithm        
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
        num_features_raw = df.shape[1]
        
        # deal with nans/missingness: most of nans will mean something like '0' - since prolific data should be complete... 
        df = df.fillna(value=0) # nans -> 0s 

        # remove non-numeric features
        df = df.loc[:, np.sum(df.applymap(lambda x: isinstance(x, (int, float))), axis=0) == df.shape[0]] 

        # remove features with no variance
        df = df.loc[:, df.var() > 0] 

        # assign data, observation, feature names
        self.X = df.values
        self.observations = df.index.values
        self.features = df.columns.values
        print('Preprocessing removed', num_features_raw-len(self.features), 'features, leaving',\
              len(self.observations), 'observations over', len(self.features), 'features')
        
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
                feature_groups.append('_'.join(feat.split('_')[0:2]))
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
        
    def standardize_data(self):
        """ Standardize data: mean 0, unit variance """
        self.X_rescaled = StandardScaler().fit_transform(self.X)
     
    def kmo_test(self):
        """  Meyer-Kaiser-Olkin test: is data suitable for factor analysis or not """
        _, kmo_model = calculate_kmo(self.X)
        return kmo_model
        
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
        corr_det = np.linalg.det(x_corr)
        statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
        dof = p * (p - 1) / 2
        p_value = chi2.sf(statistic, dof)
        return p_value
    
    def validate_fa_pca(self):
        """ Minimal tests to ensure pca/fa can be performed """
        print('\nSome tests to determine if FA/PCA should be performed:')
        if not self.is_corr_matrix:
            # not sure how to adapt this to heterogeneous correlation matrix
            kmo_model = self.kmo_test() 
            if kmo_model > .60:
                print(f'- KMO test: {np.round(kmo_model,4)} (>.60) - continue fitting fa/pca') 
            else: 
                print(f'- KMO test: {np.round(kmo_model,4)} (<.60) - reconsider fa/pca') 
        p_val = self.bartletts_test()
        if p_val < .05:
            print(f'- Barletts test of sphericity: p={np.round(p_val,4)} - continue fitting fa/pca')
        else: 
            print(f'- Barletts test of sphericity: p={np.round(p_val,4)} - reconsider fa/pca')
            
    #-----------------------------------------------------------------------
    # Unsupervised reduction
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

        # standardize data 
        self.standardize_data()
        
        # create, fit & transform 
        self.model = PCA(n_components=self.num_comps) 
        self.X_reduced = self.model.fit(self.X_rescaled).transform(self.X_rescaled) 

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
        self.loadings = self.model.loadings_ # loading matrix (features x factors): transformation of latent to observed variables (correlations of features w/ components)
        self.get_mean_loadings()
        
        # transform data: coordinates of each observation on components/dimensions
        # - calculation: X_reduced = np.dot(((X-X_mean)/X_std), np.linalg.solve(corrmat, loadings))
        if self.is_corr_matrix:
            X_scaled = (self.X - np.nanmean(self.X)) / np.nanstd(self.X)
            weights = np.linalg.solve(self.corr_matrix, self.loadings)
            self.X_reduced = np.dot(X_scaled, weights)
        else:
            self.X_reduced = self.model.transform(self.X)
            
    def efa_partial(self, thresh=75):
        ''' fit an exploratory factor analysis to a subset of the features '''
        if self.fitted:
            
            # organize each item by top factor it loads onto
            abs_loading = pd.DataFrame(np.abs(self.loadings), 
                                       index=[self.features], 
                                       columns=['factor_' + str(f) for f in np.arange(1, self.num_comps+1)])
            top_factors = abs_loading.idxmax(axis=1).values

            # keep the items that are above some threshold for their top factor
            partial_items = []
            items_incl = []
            for f in np.arange(0, self.num_comps):
                items_mask = top_factors == 'factor_' + str(f+1)
                factor_items = self.features[items_mask]
                factor_loadings = self.loadings[:,f][items_mask]
                items = list(factor_items[np.abs(factor_loadings) > np.percentile(np.abs(factor_loadings), thresh)]) # threshold
                partial_items.append(items)
                items_incl += items
            incl_mask = np.isin(self.features, items_incl)
            
            # transform data with this subset of items
            loadings_  = self.loadings[incl_mask, :]
            corr_matrix_ = pd.DataFrame(self.corr_matrix, columns=self.features).iloc[incl_mask, incl_mask]
            X_ = self.X[:, incl_mask]   
            X_scaled_ = (X_ - np.nanmean(X_, axis=0)) / np.nanstd(X_, axis=0)
            weights_  = np.linalg.solve(corr_matrix_, loadings_)
            self.X_partial_reduced = np.dot(X_scaled_, weights_)

            # plot the correlation between original transform & this reducd one
            fig,axs = plt.subplots(1, self.num_comps, figsize=(5*self.num_comps, 4))
            for f in np.arange(0, self.num_comps):
                sns.regplot(x=self.X_partial_reduced[:,f], y=self.X_reduced[:,f], ax=axs[f],
                            scatter_kws = {'color': 'purple', 'alpha': 0.3},
                            line_kws = {'color':'purple', 'alpha': 0.3, 'lw':3})
                axs[f].set_title('factor ' + str(f+1))
            
            # maybe make this a dictionary instead...
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
    
    # def mca(self):
    #     """
    #         Multiple Correspondence Analysis for categorical data
    #         try this instead: https://github.com/MaxHalford/prince#multiple-correspondence-analysis-mca
    #     """
    #     import prince
    #     mca = prince.MCA()
    #     mca.fit(X)
    #     mca.transform(X)        

    def mds(self): 
        """
            Multidimensional Scaling
            Goal: reduce dimensions while preserving distance between data points
        """        
        # get digitized rdm
        if not self.is_corr_matrix:
            rdm = 1 - np.corrcoef(self.data)
        else:
            rdm = 1 - self.X
        digitized_rdm = self.digitize_rdm(rdm)
        
        # create & fit mds object
        num_comp = 3
        mds = MDS(n_components=num_comp, dissimilarity="precomputed", random_state=0)  
        results = mds.fit(digitized_rdm)
        
        # plot
        self.mds_plot(results, num_comp)

    #-----------------------------------------------------------------------
    # Plotting
    #-----------------------------------------------------------------------

    def get_mean_loadings(self):
        """ """
        self.mean_loadings = pd.DataFrame([np.mean(self.loadings[ixs[0]:ixs[1],:],axis=0) for fg, ixs in self.feature_groups_bins.items()], 
                                     columns=['component ' + str(c) for c in np.arange(1, self.loadings.shape[1]+1)])
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
        
        #         if refit: 
        #             self.reproduced_corr_matrix_partial = np.matmul(self.loadings, self.loadings.T)
        #             self.residuals_partial = self.corr_matrix - self.reproduced_corr_matrix
        
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
        ax.set_xticks(np.arange(X.shape[1])[0::4])
        ax.set_xticklabels(features[0::4], rotation=90, fontsize=11)

        # y-axis 
        if len(feature_groups) < len(features)/5: # if there are groupings to the features...
            ax.set_yticks([np.mean(ixs) for fg, ixs in feature_groups_bins.items()])
            ax.set_yticklabels(feature_groups, fontsize=20)
            # ax.vlines([ixs[0] for fg, ixs in self.feature_groups_bins.items()], 0, self.X.shape[1]-1)
            # ax.hlines([ixs[0] for fg, ixs in self.feature_groups_bins.items()], 0, self.X.shape[1]-1)
        else:
            ax.set_yticks(np.arange(X.shape[1])[0::4])
            ax.set_yticklabels(features[0::4], fontsize=11)

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
        
    def plot_loadings(self, num_comps=3, colors=None, labels=None):
        """ plot the component loadings barplot """

        # add errors if its for wrong dim reduction method
        if num_comps < self.num_comps:
            num_comps = self.num_comps

        # color diff questionnaires diff
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
        if labels is None:
            labels = [fg for fg, ixs in self.feature_groups_bins.items()]
        patches = [mpatches.Patch(facecolor=inst[0], edgecolor='black', 
                                label=inst[1]) for inst in zip(colors, labels)]

        # for ylim
        ymax = np.round(np.max(self.loadings),1)+.05  
        for c in range(num_comps):
            loadings_ = self.loadings[:, c]
            fig, ax = plt.subplots(figsize=(20, 5))
            ax.bar(np.arange(len(loadings_)), loadings_, color=colors_, edgecolor = 'black')
            ax.legend(handles=patches, loc='upper right', 
                    frameon=False, bbox_to_anchor=(1.15, 1), borderaxespad=0,
                    prop={'size': 14})
            
            plt.xlim(-5,len(self.features)+5)
            plt.ylim(-1,1)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f'Component {c+1} ({np.round(self.exp_var[c]*100, 2)} % explained)', fontsize=20)
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
            
    def plot_components(self, figsize=(20, 20)):
        '''
            plot the components/eigenvectors: directions of maxmimum variance
        '''
        # plot by feature
        vmax = np.abs(self.eigvecs).max() # max eigenvector value
        fig, ax = plt.subplots(figsize=figsize+figsize/2)
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
            
    def digitize_rdm(self, rdm_raw, n_bins=10): 
        """
            digitize an input matrix to n bins (10 bins by default)
            rdm_raw: a symmetrical, square matrix 
        """
        rdm_bins = [np.percentile(np.ravel(rdm_raw), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
        rdm_vec_digitized = np.digitize(np.ravel(rdm_raw), bins = rdm_bins) * (100 // n_bins) # Compute the vectorized digitized value 
        rdm_digitized = np.reshape(rdm_vec_digitized, np.shape(rdm_raw)) # Reshape to matrix
        rdm_digitized = (rdm_digitized + rdm_digitized.T) / 2  # Force symmetry in the plot
        return rdm_digitized

    def output(self, output_dir=''):
        if not self.fitted:
            raise Exception('The model is not fitted. Run model.fit_transform first.')
        if self.rotation is not None:
            out_file = f'{output_dir}/{self.alg}_{self.num_comps}comps_{self.rotation}_{len(self.features)}features_n{len(self.observations)}.pkl'
        else:
            out_file = f'{output_dir}/{self.alg}_{self.num_comps}comps_{len(self.features)}features_n{len(self.observations)}.pkl'
        pickle_file(self, out_file)


# PCA specific
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
  
def pca_reconstruction(X, pca, n_comps):
    # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

    pca_embedding = pca.transform(X)
    X_mu = np.mean(X, axis=0)
    X_hat = np.dot(pca_embedding[:,:n_comps], pca.components_[:n_comps,:]) + X_mu
    return X_hat


#--------------------------------------------------------------------------------------------
# linear cross decomposition: e.g., partial least squares, canonical correlation analysis
#--------------------------------------------------------------------------------------------


class CrossDecompose(object):

    '''[By Matthew Schafer; github: @matty-gee; 2020ish]'''
    
    def __init__(self, n_comps=2, scale=True, mode='pls'):

        self.n_comps = n_comps
        self.comp_range = range(1, n_comps+1)
        self.scale = scale
        self.fitted = False
        self.mode = mode
        self.errors = []

        if mode == 'cca':
            # symmetric deflation of X & Y 
            self.model = CCA
        elif mode == 'pls_canonical':
            self.model = PLSCanonical
            # canonical pls is also symmetric but may outperform cca when n_features > n_samples
            # can only find n_components <= min(n_samples, n_features)
        elif mode == 'pls':
            # asymmetric deflation of X & Y
            self.model = PLSRegression

    def check_input(self, X, Y):
        # if not a dataframe, then create list for index
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'X{i}' for i in range(1,X.shape[1]+1)])
        if Y is not None:
            if not isinstance(Y, pd.DataFrame):
                if isinstance(Y, pd.Series): 
                    Y = Y.to_frame() # turn into a dataframe
                else: 
                    if len(Y.shape) == 1: Y = Y.reshape(-1,1)
                    Y = pd.DataFrame(Y, columns=[f'Y{i}' for i in range(1,Y.shape[1]+1)])
        return X, Y

    def fit(self, X, Y, n_comps=None):

        X, Y = self.check_input(X, Y)

        # if n_comps or scal are none use self
        if n_comps is None: 
            n_comps = self.n_comps
        
        # check if n_comps is valid
        if self.mode in ['cca', 'pls_canonical']:
            max_n_comps = min(X.shape[0], X.shape[1], Y.shape[1])
            if not n_comps <= max_n_comps:
                print(f'n_comps must be <= min(n_samples, n_features, n_targets); changing to {max_n_comps}')
                n_comps = max_n_comps

        # come up with as simple & useful a data storage scheme as possible....
        self.model_ = self.model(n_components=n_comps, scale=self.scale)
        self.model_.fit(X, Y)
        self.fitted = True
        self.X_transformed, self.Y_transformed = self.model_.transform(X, Y)

        #---------------------------------------------------------------
        # canonical covariates (i.e., scores)
        # - rotations are the projection matrices for each dataset
        # - project the datasets into the canonical/shared space to get canonical covariates
        #---------------------------------------------------------------

        X_cc, Y_cc = self.model_.x_scores_, self.model_.y_scores_
        
        self.X_scores = pd.DataFrame(X_cc, columns=[f'X_CC0{i}' for i in range(1, n_comps+1)])
        self.Y_scores = pd.DataFrame(Y_cc, columns=[f'Y_CC0{i}' for i in range(1, n_comps+1)])

        self.X_rotations = self.model_.x_rotations_
        self.Y_rotations = self.model_.y_rotations_
        
        # scores should equal the rotations multiplied by the scores
        # try: 
        #     assert np.allclose(X @ self.X_rotations, X_cc, rtol=0.01), 'X @ x_rotations != X_rotated'
        #     assert np.allclose(Y @ self.Y_rotations, Y_cc, rtol=0.01), 'Y @ y_rotations != Y_rotated'
        # except AssertionError as e:
        #     self.errors.append(repr(e))

        #---------------------------------------------------------------
        # canonical correlations: correlations between X & Y canonical covariates
        # - statsmodels: cca.cancorr_
        #---------------------------------------------------------------

        can_corrs = np.corrcoef(X_cc.T, Y_cc.T).diagonal(offset=n_comps)
        self.can_corrs = pd.DataFrame(can_corrs, 
                                      columns=['can_corr'], 
                                      index=[f'CC0{i}' for i in range(1, n_comps+1)])

        #---------------------------------------------------------------
        # weights: translate the [deflated] X & Y matrices to X & Y scores

        # - X_deflated @ X_weights = X_scores
        # - deflated matricces are ones with previous variance accounted for subtracted out
        # - CCA has symmetrical deflation, PLS is asymmetrical (X only)
        # - weights are not very interpretable; better to use loadings
        #---------------------------------------------------------------
        
        #---------------------------------------------------------------
        # loadings: translate the X & Y scores *back* to the original X & Y matrices
        # cross-loadings: the relationship between X & Y to opposite canonical covariates

        # - X_scores @ X_loadings = X
        # - ! more interpretable than weights b/c they are not on the deflated matrices
        # - should be: corr(X, X_c)
        #---------------------------------------------------------------

        # try:
        #     assert np.allclose(X_cc @ self.model_.x_loadings_.T, X, atol=0.01), 'x_scores @ x_loadings.T != X'
        #     assert np.allclose(Y_cc @ self.model_.y_loadings_.T, Y, atol=0.01), 'y_scores @ x_loadings.T != Y'
        # except AssertionError as e:
        #     self.errors.append(repr(e))
        
        # may not always make sense.....
        # ## get all loadings, including cross-loadings
        # # - !! poss. problem is the cca.x_loadings_ is a bit diff!!
        # self.loadings = pd.DataFrame(columns=['X_CC01', 'X_CC02', 'Y_CC01', 'Y_CC02']) # make this dynamically
        # for fname, features in {'X':X, 'Y':Y}.items():
        #     for f in range(features.shape[1]): # assumes same number of features in both and components for each
        #         for c in range(n_comps):
        #             self.loadings.loc[f'{fname}{f+1}', f'Y_CC0{c+1}'] = np.corrcoef(features.iloc[:,f], Y_cc[:,c])[0,1]
        #             self.loadings.loc[f'{fname}{f+1}', f'X_CC0{c+1}'] = np.corrcoef(features.iloc[:,f], X_cc[:,c])[0,1]

        self.X_loadings = pd.DataFrame(self.model_.x_loadings_, index=X.columns, columns=[f'X_loading0{i}' for i in range(1, n_comps+1)])
        self.Y_loadings = pd.DataFrame(self.model_.y_loadings_, index=Y.columns, columns=[f'Y_loading0{i}' for i in range(1, n_comps+1)])

        return self.model_

    def predict(self, X, Y=None, plot=True, colors=None):

        X, Y = self.check_input(X, Y)
        if self.fitted is False:
            print('Fit first...')
            return None

        Y_pred = self.model_.predict(X)
        if plot is True and Y is not None:
            self.plot_predictions(Y, Y_pred, X, colors=colors)

        return Y_pred

    def fit_predict(self, X, Y, n_comps=None, plot=True):
            
        if n_comps is None: n_comps = self.n_comps
        self.fit(X, Y, n_comps)
        Y_pred = self.predict(X, Y, plot)
        return Y_pred

    def optimize_fit(self, X, Y, max_comps=None, folds=5, metric='MSE', \
                    plot=True, colors=None):

        if max_comps is None: comp_range = np.arange(1, self.n_comps+1) # 1,2
        else:                 comp_range = np.arange(1, max_comps+1)

        #------------------------------------------------------------
        # find minimal mse for number of components using cross-validation
        # - options of MSE & MAE (L1 loss)
        #------------------------------------------------------------

        comps_mse, comps_mae = [], []
        for i in comp_range:
            Y_pred = cross_val_predict(PLSRegression(n_components=i), X, Y, cv=folds) 
            comps_mse.append(mean_squared_error(Y, Y_pred))
            comps_mae.append(mean_absolute_error(Y, Y_pred))
            # calcuate mean absolute error too

        # see if there is a difference between mse and mae
        msemin, maemin = np.argmin(comps_mse), np.argmin(comps_mae)
        # if maemin != msemin: 
        #     print('MSE and MAE disagree on optimal number of components')
        
        if metric is 'MSE': 
            self.n_comps = msemin + 1
            metric_min = msemin
            metric_vals = comps_mse
        elif metric is 'MAE':
            self.n_comps = maemin + 1
            metric_min = maemin
            metric_vals = comps_mae

        #------------------------------------------------------------
        # Refit w/ optimal number of components
        #------------------------------------------------------------
        
        # refit 'calibration' w/ entire dataset
        fitted = self.fit(X, Y, n_comps=self.n_comps)
        Y_preds = self.predict(X, Y, plot=plot, colors=colors)
        r2  = r2_score(Y, Y_preds)
        mse = mean_squared_error(Y, Y_preds)
        mae = mean_absolute_error(Y, Y_preds)

        # Cross-validation
        fitted_cv = self.model(n_components=self.n_comps).fit(X, Y)
        Y_preds_cv = cross_val_predict(fitted_cv, X, Y, cv=folds)    
        r2_cv = r2_score(Y, Y_preds_cv)
        mse_cv = mean_squared_error(Y, Y_preds_cv)
        mae_cv = mean_absolute_error(Y, Y_preds_cv)

        #------------------------------------------------------------
        # print/plot out metrics
        #------------------------------------------------------------

        if plot is True:

            print(f"{metric} suggested number of components = {self.n_comps}")

            # plot the mse as a function of component number
            fig = plt.figure(figsize=(3,3))
            plt.plot(comp_range, np.array(metric_vals), '-v', color='blue', mfc='blue')
            plt.plot(comp_range[metric_min], np.array(metric_vals)[metric_min], 'P', ms=10, mfc='red')
            plt.xlabel('Number of components')
            plt.ylabel(f'{folds}-fold CV {metric}')
            plt.xlim(left=1)
            plt.xticks(comp_range)
            plt.show()

            print(f"{'Model':<15} {'R2':<10} {'MSE':<10} {'MAE':<10}")
            print(f"{'Calibration':<15} {np.round(r2, 3):<10} {np.round(mse,3):<10} {np.round(mae,3):<10}")
            print(f"{f'{folds}-fold CV':<15} {np.round(r2_cv, 3):<10} {np.round(mse_cv,3):<10} {np.round(mae_cv,3):<10}")

    def transform(self, X=None, Y=None):
        return self.model_.transform(X, Y)

    def fit_transform(self, X=None, Y=None, n_comps=None):
        self.fit(X, Y, n_comps)
        return self.model_.transform(X, Y)

    def inverse_transform(self, X_tr=None, Y_tr=None):
        # only exact if n_comps = n_features
        return self.model_.inverse_transform(X_tr, Y_tr)

    def score(self, X, Y):
        return self.model_.score(X, Y)

    def plot_predictions(self, Y, Y_pred, X, colors=None):
        
        n_targets = Y.shape[1]
        X_transformed = self.model_.transform(X)
        n_comps = X_transformed.shape[1]

        if not isinstance(Y, pd.DataFrame): Y = pd.DataFrame(Y)

        fig, axs  = plt.subplots(n_targets, n_comps+1, figsize=(3*(n_comps+1), 3*n_targets))
        
        for y in range(n_targets):

            if (n_targets > 1):    ax = axs[y, 0]
            elif (n_targets == 1): ax = axs[0]

            #------------------------------------------------------------
            # plot the predicted Y against real Y 
            #------------------------------------------------------------

            ax.scatter(x=Y_pred[:, y], y=Y.iloc[:, y], alpha=0.75, edgecolor='black', linewidth=0.5, c=colors)
            ax.set_title(f"Y {y + 1}")
            ax.set(xlabel=f"Predicted Y {y + 1}", ylabel=f"True Y {y + 1}")
            
            # annotate plot with correlation
            r = np.corrcoef(Y.iloc[:, y], Y_pred[:, y])[0, 1]
            ax.annotate(f"r  = {np.round(r, 3)}", xy=(0.05, 0.9), fontsize=9, xycoords='axes fraction')           
            
            # make sure the axes are same
            ylim = ax.get_ylim()
            ax.set_xlim(ylim)
            ax.plot(ylim, ylim, ls="--", color='black')
            
            for c in range(n_comps):

                if (n_targets > 1):    ax = axs[y, c+1]
                elif (n_targets == 1): ax = axs[c+1]
            
                # plot the projected X values/scores against the Y values
                ax.scatter(x=X_transformed[:, c], y=Y.iloc[:, y], alpha=0.50, label="observed", 
                           edgecolor='black', linewidth=0.25, c=colors, marker='o')
                ax.scatter(x=X_transformed[:, c], y=Y_pred[:, y], alpha=0.75, label="predicted", 
                           edgecolor='black', linewidth=2, c=colors, marker='o')
                           
                r = np.corrcoef(X_transformed[:, c], Y.iloc[:, y])[0, 1]
                ax.annotate(f"r  = {np.round(r, 3)}", xy=(0.05, 0.9), fontsize=9, xycoords='axes fraction') 
                ax.set(xlabel=f"Projected X {c + 1}", ylabel=f"True y {y + 1}", title=f"Component {c+1}")

                # same y axis
                ax.set_ylabel("")
                # ax.set_ylim(ylim)

            if c == n_comps - 1: ax.legend().set_visible(True)
        
        plt.tight_layout()
        plt.show()

    def plot_X_loadings(self, X_loadings=None, items=True):
  
        colors = ['red', 'blue', 'purple', 'green', 'lavender', 'grey', 'fuchsia', 'orange', 'dodgerblue', 
                    'yellow', 'orchid', 'indigo', 'aqua','palegreen', 'silver', 'plum', 'fuchsia', 'coral',
                    'gold', 'pink','slategray', 'forestgreen','peachpuff','honeydew','brown','olivedrab',
                    'darkturquoise', 'tan', 'springgreen', 'mintcream','navajowhite','chocolate','lightblue','chartreuse',
                    'lime','yellowgreen','khaki','gold','teal','tomato']

        if X_loadings is None: X_loadings = self.X_loadings
        xvars = X_loadings.index

        # organize the coloring scheme
        plot_colors = []
        if items:
            plot_labels = [xvar.split('_')[0] for xvar in xvars]
            plot_labels = list(np.unique(plot_labels))
            plot_colors = [colors[plot_labels.index(xvar.split('_')[0])] for xvar in xvars]
        else:
            plot_labels = xvars
            plot_colors = colors[:len(plot_labels)]
            
        # plot loadings
        fig, axs = plt.subplots(self.n_comps, 1, figsize=(11, 2.5*self.n_comps))
        for i in range(self.n_comps):
            if self.n_comps > 1: ax = axs[i]
            else:                ax = axs
            ax.bar(np.arange(len(X_loadings)), X_loadings.iloc[:,i], 
                   color=plot_colors, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Component {i+1} X loadings', fontsize=14)
            ax.set_xticks([])
            # ax.set_xticks(np.arange(len(X_loadings)))
            # ax.set_xticklabels([X_loadings.index], rotation=90, fontsize=10)

        patches = [mpatches.Patch(color=colors[i], label=plot_labels[i]) for i in range(len(plot_labels))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()
   
    def get_top_features(self, X_loadings=None, n_features=50):
        # maybe eventually I want to do in cross-validation loop?
        if X_loadings is None: X_loadings = self.X_loadings
        X_loadings = np.abs(X_loadings)
        top_features = []
        for i in range(self.n_comps):
            features = X_loadings.iloc[:,i].sort_values(ascending=False).index[:n_features]
            top_features.append(list(features))
        return top_features

# TODO add uniqueness scores etc like efa
# TODO add biplots, rotations etc...


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
    dsm_emb = pairwise_distances(embedding, metric='euclidean') # distances betweeen mds embedding points
    raw_stress = 0.5 * np.sum((dsm_emb - dsm) ** 2)
    return raw_stress

def convert_rawstress_to_kruskal(dm, raw_stress):
    ''' 
        a scaled version of the raw stress
        dm = orig. dissimilarity matrix
        - raw stress can be misleading b/c it depends on normalization of disparities
        - kruskal stress is value in range [0,1], where smaller is better
        - divide demoninator by two to make it only half of the symmetric matrix
    '''
    kruskal_stress = np.sqrt(raw_stress / (0.5 * np.sum(dm ** 2)))
    return kruskal_stress
