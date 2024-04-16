import umap
import kmapper as km
from ripser import ripser, Rips
from persim import plot_diagrams
from gtda.homology import (
    VietorisRipsPersistence, EuclideanCechPersistence, 
    WeightedRipsPersistence, CubicalPersistence
    )
from gtda.diagrams import (
    PersistenceImage, PersistenceLandscape, 
    BettiCurve, HeatKernel, 
    PersistenceEntropy, NumberOfPoints, 
    Amplitude, ComplexPolynomial, Scaler
    )
from gtda.plotting import plot_diagram
from gtda.mapper import (
    CubicalCover, Projection,
    make_mapper_pipeline,
    plot_static_mapper_graph, plot_interactive_mapper_graph,
    MapperInteractivePlotter)
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, spectral_clustering


#----------------------------------------------------------------------------------------
# mapper algorithm
#----------------------------------------------------------------------------------------


def calc_mapper_graph(coords,
                    scaler=MinMaxScaler(), 
                    projection=LocallyLinearEmbedding(n_components=2, n_neighbors=10),
                    cover=km.Cover(n_cubes=5, perc_overlap=0.25), 
                    clusterer=KMeans(n_clusters=5)):
    
    """
        Creates a mapper graph to visualize the topology of high-dimensional data

        Parameters:
        -----------
        coords : numpy.ndarray
            The data points to be visualized, with shape (n_samples, n_features).

        scaler : object, default=MinMaxScaler()
            The scaler object to use to preprocess the data before projection.
        
        projection : object, default=LocallyLinearEmbedding(n_components=2, n_neighbors=10)
            The projection object to use to reduce the dimensionality of the data.

        cover : object, default=km.Cover(n_cubes=5, perc_overlap=0.25)
            The cover object to use to divide the projected space into overlapping hypercubes.

        clusterer : object, default=KMeans(n_clusters=5)
            The clustering object to use to assign data points to nodes in the mapper graph.

        Returns:
        --------
        graph : dict
            The mapper graph, represented as a dictionary with keys "nodes" and "links". 
            The "nodes" value is a list of dictionaries, where each dictionary represents a node in the graph. 
            The "links" value is a list of dictionaries, where each dictionary represents a link between two nodes in the graph.
    """

    mapper = km.KeplerMapper(verbose=0)
    projected = mapper.fit_transform(coords, scaler=scaler, 
                                     projection=projection, 
                                     distance_matrix=False)
    return mapper.map(projected, coords, cover=cover, clusterer=clusterer)


#----------------------------------------------------------------------------------------
# persistent homology
#----------------------------------------------------------------------------------------


def betti_summary(H):
    try: 
        persistence = H[:,1] - H[:,0]
        return [np.max(persistence), len(H)]
    except: 
        return [np.nan, np.nan]

def diagram_amplitude(dgm):
    # amplitudes
    if dgm.ndim == 2: dgm = dgm[None, :, :] # expects 3d input
    n_dims = len(np.unique(dgm[:,:,2]))
    amp_df = pd.DataFrame(columns=['metric'] + [f'amplitude{i}' for i in range(n_dims)])
    for metric in ['betti', 'wasserstein', 'landscape', 'silhouette', 'persistence_image']:
        ampl = Amplitude(metric=metric, order=None).fit_transform(dgm)
        amp_df.loc[len(amp_df),:] = [metric] + ampl.tolist()[0]
    return amp_df

def gtda_to_ripser_diagram(dgm):
    # go from gtda to ripser format
    # note: ripser includes the infinitely persisting feature for dim 0
    return [dgm[dgm[:,2] == d] for d in np.unique(dgm[:,2])]

