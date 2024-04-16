'''
    code to compute trajectory related analyses: spline deocoders, trajectory distances, etc...
'''

# TODO: https://stackoverflow.com/questions/51321100/python-natural-smoothing-splines

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, BSpline
from scipy.spatial import distance
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from scipy.spatial.distance import cdist, euclidean, directed_hausdorff
from scipy.spatial import procrustes
from frechetdist import frdist
from fastdtw import fastdtw


#------------------------------------------------------------
# fitting splines & decoding held out points locations
#------------------------------------------------------------


def parameterize_spline_OLD(decoder_dict, values):

    ''' parameterize spline coordinates with behavioral locations '''

    u_evaluated   = decoder_dict['u_evaluated']
    spl_evaluated = decoder_dict['spl_evaluated']

    # for each trial (u parameter), use euclidean distance to find closest point on evaluated spline 
    dists = np.linalg.norm(spl_evaluated[np.newaxis, :] - u_evaluated[:, np.newaxis, :], axis=2)
    min_ix = np.argmin(dists, axis=1) # index of closest point on spline
    spl_steps = np.diff(min_ix) # how many points between consecutive locations on spline
    assert np.all(spl_steps > 0), f'spline indices are not monotonically increasing: {spl_steps}'
    spl_steps = np.insert(np.diff(min_ix), 0, 0)

    # interpolate between consecutive trials parameter values along spline
    if values.ndim == 1:
        spline_locations = np.hstack([np.linspace(values[i], values[i+1], spl_steps[i+1], endpoint=False) for i in range(len(values)-1)])
        spline_locations = np.hstack([spline_locations, values[-1]])
    elif values.ndim == 2: 
        x_int = np.hstack([np.linspace(values[i,0], values[i+1,0], spl_steps[i+1], endpoint=False) for i in range(len(values)-1)])
        y_int = np.hstack([np.linspace(values[i,1], values[i+1,1], spl_steps[i+1], endpoint=False) for i in range(len(values)-1)])
        spline_locations = np.vstack([np.vstack([x_int, y_int]).T, values[-1,:]]) # add last location

    return spline_locations


#------------------------------------------------------------
# trajecotry similarity measures
#------------------------------------------------------------


def calc_pairwise_frechet_distance(traj_data):
    # calculate pairwise frechet distance between all characters
    # traj_data: (n_characters, n_trials, n_dimensions)
    # distance_matrix: (n_characters, n_characters)
    distance_matrix = np.zeros((traj_data.shape[0], traj_data.shape[0]))
    for i, j in itertools.combinations(range(traj_data.shape[0]), 2):
        distance_matrix[i, j] = distance_matrix[j, i] = frdist(traj_data[i], traj_data[j])
    return distance_matrix

def undirected_hausdorff(U, V):
    # calculate hausdorff distance between two arrays, un-directed 
    # aximum of the directed Hausdorff distance from A to B and the directed Hausdorff distance from B to A
    # symmetric measure of the distance between two sets of points
    return max(directed_hausdorff(U, V)[0], directed_hausdorff(V, U)[0])

def pairwise_trajectory_distances(trajectories, metric='euclidean'):

    # calculate some notion of distance between different trajectories...

    # arguments:
    # - trajectories: (n_trajectories, n_points, n_dimensions)
    # - metric: what to compute
    # returns: distance_matrix with shape = (n_trajectories, n_trajectories)
    
    n_trajs = trajectories.shape[0]
    distance_matrix = np.zeros((n_trajs, n_trajs))
    for i, j in itertools.combinations(range(n_trajs), 2):
        if metric == 'frechet':      d = frdist(trajectories[i], trajectories[j])
        elif metric == 'dtw':        d = fastdtw(trajectories[i], trajectories[j], dist=euclidean)[0]
        elif metric == 'hausdorff':  d = undirected_hausdorff(trajectories[i], trajectories[j])
        elif metric == 'procrustes': d = procrustes(trajectories[i], trajectories[j])[2]
        else:                        d = np.nanmean(cdist(trajectories[i], trajectories[j], metric=metric).diagonal()) # pairwise or elementwise?
        distance_matrix[i,j] = distance_matrix[j,i] = d
    return distance_matrix
