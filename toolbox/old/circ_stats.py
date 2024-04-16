'''
    Functions related to circular statistics and other related problems - e.g., angles, distances, trigonometry, geometry
    Note: all functions that accept angles expect radians
    Wrapped/adapted code from astropy & Matlab's circstat [see docstrings]
'''

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from turtle import degrees
import pycircstat
import astropy.stats
from shapely.geometry import Polygon
from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt

from general_utils import fill_in_upper_tri

# [member[0] for member in list(getmembers(pycircstat, isfunction))] # get all functions within a module

#---------------------------------------------------------------------------------------------------------
# private functions
#---------------------------------------------------------------------------------------------------------


def _components(data, p=1, phi=0.0, axis=None, weights=None):
    ''' 
        computing the generalized rectangular components of the circular data
        [from astropy]
    '''

    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')

    C = np.sum(weights * np.cos(p * (data - phi)), axis)/np.sum(weights, axis)
    S = np.sum(weights * np.sin(p * (data - phi)), axis)/np.sum(weights, axis)

    return C, S

def _angle(data, p=1, phi=0.0, axis=None, weights=None):
    ''' 
        computing the generalized sample mean angle
        [from astropy]
    '''
    
    C, S = _components(data, p, phi, axis, weights)

    # theta will be an angle in the interval [-np.pi, np.pi)
    # [-180, 180)*u.deg in case data is a Quantity
    theta = np.arctan2(S, C)

    if isinstance(data, Quantity):
        theta = theta.to(data.unit)

    return theta

def _length(data, p=1, phi=0.0, axis=None, weights=None):
    ''' 
        computing the generalized sample length 
        [from astropy]
    '''
    C, S = _components(data, p, phi, axis, weights)
    return np.hypot(S, C)

def _corr(x, y, axis=0):
    ''' correlate x & y '''
    return np.sum((x - x.mean(axis=axis, keepdims=True))  * \
                  (y - y.mean(axis=axis, keepdims=True)), axis=axis) \
            / np.std(x, axis=axis) / np.std(y, axis=axis) / x.shape[axis]

def _coincident_vectors(u, v):
    ''' Checks if vectors (u & v) are the same or scalar multiples of each other'''
    return np.dot(u, v) * np.dot(u, v) == np.dot(u, u) * np.dot(v, v)


#---------------------------------------------------------------------------------------------------------
# vector math 
#--------------------------------------------------------------------------------------------------------- 

# probably better to just use the numpy.linalg versions:
def l2_norm(v):
    ''' 
        Returns vectors l2 norm/magnitude/length 
        L2 norm = square root of sum of squared vector values
        (Equivalent to: np.linalg.norm(v))
    '''
    return np.sqrt(np.sum(np.square(v)))

def l1_norm(v):
    '''
        Returns vectors l1 norm/magnitude/length 
        L1 norm = sum of vector absolute values
        (Equivalent to: np.linalg.norm(v, ord=1))
    '''
    return np.sum(np.abs(v))

def l1_normalize(v):
    ''' Returns l1 normalized vector '''
    return v / l2_norm(v)

def l2_normalize(v):
    ''' Returns l2 normalized vector with length of 1 (aka unit vector) '''
    return v / l2_norm(v)

def circ_vectors(theta, r):
    ''' 
        returns x & y components and unit vector values
    '''        
    theta = theta.astype(float)
    xy_comp = np.array([r*np.cos(theta), r*np.sin(theta)]).T # x,y components
    xy_unit = np.array([np.cos(theta), np.sin(theta)]).T # x,y on unit circle
    return xy_comp, xy_unit

def angle_to_circle(theta, ori=None, radius=3):
    '''
        project angles to circle perimeter
        can improve clustering of angles - makes them comparable by distance
        
        theta : float
            angle in radians
        ori : tuple
            origin [x,y]
    '''
    if ori is None: ori = [0,0]
    circ_x = ori[0] + radius + np.cos(theta)
    circ_y = ori[1] + radius + np.sin(theta)
    return np.array([circ_x, circ_y]).T


#---------------------------------------------------------------------------------------------------------
# vector similarity 
#---------------------------------------------------------------------------------------------------------


def angle_between_vectors(u, v, direction=None, verbose=False):
    '''
        Compute elementwise angle between sets of vectors u & v
            
        uses np.arctan2(y,x) which computes counterclockwise angle [-π, π] between origin (0,0) and x, y
        clockwise v. counterclockwise: https://itecnote.com/tecnote/python-calculate-angle-clockwise-between-two-points/  
        included: ADD LINK

        TODO: make pairwise..?

        Arguments
        ---------
        u : array-like
            vector
        v : array-like
            another vector
        direction : None, True or False (optional, default=None)
            None == Included
            True == Clockwise 360
            False == Counterclockwise 360 
        verbose : bool (optional, default=False)
             
        Returns
        -------
        float 
            angle in radians 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    #     if U.shape != V.shape:
    #         if verbose: print(f'Different shape vectors: U={U.shape}, V={V.shape}. Assuming smaller is reference.')
    #         if len(U) < len(V): U = np.repeat(np.expand_dims(U, 0), len(V), axis=0)
    #         else:               V = np.repeat(np.expand_dims(V, 0), len(U), axis=0)
    #     rads = []
    #     for u, v in zip(U, V):        
    # 
    #     
    # if one of vectors is at origin, the angle is undefined but could be considered as orthogonal (90 degrees)
    if ((u==0).all()) or ((v==0).all()): 
        if verbose: print(u, v, 'at least 1 vector at origin; treating as orthogonal')
        rad = np.pi/2

    # if same vectors (or scalar multiples of each other) being compared, no angle between (0 degrees)
    # -- b/c 0-360 degrees, direction matters: make sure the signs are the same too
    elif (_coincident_vectors(u, v)) & all(np.sign(u) == np.sign(v)):
        if verbose: print(u, v, 'same vectors, no angle in between')
        rad = 0 # 0 degrees == 360 degrees == 2*pi radians 

    else:

        if direction is None: 

            # "included" angle from [0,180], [0, π] 
            rad = np.arccos(np.dot(u, v) / (l2_norm(u) * l2_norm(v)))
            # equivalent: np.arctan2(l2_norm(np.cross(u, v)), np.dot(u, v))

        elif direction is True: 

            # clockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*v[::-1]) - np.arctan2(*u[::-1])) % (2 * np.pi)  
        
        elif direction is False:

            # counterclockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*u[::-1]) - np.arctan2(*v[::-1])) % (2 * np.pi)
            
    return rad

def calculate_angle(U, V=None, direction=None, force_pairwise=True, verbose=False):
    '''
        Calculate angles between n-dim vectors 
        If V == None, calculate U pairwise
        Else, calculate elementwise
        
        TODO: more explanation; find more elegant ways to do this; also adapt other pairwise like functions to have structure

        Arguments
        ---------
        U : array-like
            shape (n_vectors, n_dims)
        V : array-like
            shape (n_vectors, n_dims)
        direction : optional (default=None)
            None : included 180
            False : counterclockwise 360 (wont give a symmetrical matrix)
            True : clockwise 360
            
        Returns
        -------
        numeric 
            pairwise angles in radians

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # # testing (10-12-22)
    # U = np.random.randint(100, size=(4,2))
    # for V in [None, np.random.randint(100, size=U.shape), 
    # np.random.randint(100, size=(1, U.shape[1])), np.random.randint(100, size=(7, U.shape[1]))]:

    messages = []

    # check/fix shapes
    if U.ndim == 1: 
        U = np.expand_dims(U, 0)
        messages.append('Added a dimension to U')
        
    if V is not None:
        if V.ndim == 1: 
            V = np.expand_dims(V, 0)
            messages.append('Added a dimension to V')

            
    # determine output shape 
    # - 1 set of vectors: pw, square, symm
    if V is None: 
        default = 'pairwise'
        V = U 
        
    # - 2 vectors of same shape
    # -- pw, square, non-symm
    # -- ew, vector
    elif U.shape == V.shape: 
        default = 'elementwise' 

    # - 2 vectors, 1 w/ length==1 & is reference
    # -- pw, vector shape (1,u)
    elif (U.shape[0] > 1) & (V.shape[0] == 1): 
        V = np.repeat(V, len(U), 0) 
        default = 'reference'  
    
    # -- pw, vector shape (v,1)
    elif (U.shape[0] == 1) & (V.shape[0] > 1): 
        U = np.repeat(U, len(V), 0) 
        default = 'reference' 
        
    # - 2 vectors, different lengths
    # -- pw, rectangle 
    else: 
        default = 'pairwise' 
        
    messages.append(f'Calculated {default}')
    
    
    # calculate angles
    radians = np.zeros((U.shape[0], V.shape[0]))
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            radians[i, j] = angle_between_vectors(U[i,:], V[j,:], direction=direction)

            
    # output
    if default == 'pairwise': cols = 'U'
    else:                     cols = 'V'
    radians = pd.DataFrame(radians, index=[f'U{i+1:02d}' for i in range(len(U))], columns=[f'{cols}{i+1:02d}' for i in range(len(V))])

    if not force_pairwise:
        if default == 'reference':
            radians = radians.iloc[:,0].values
        elif default == 'elementwise':
            radians = np.diag(radians)
    if verbose: [print(m) for m in messages]
        
    return radians

def cosine_distance(u, v=None):
    ''' 
        cosine distance of (u, v) = 1 - (dot(u,v) / dot(l2_norm(u), l2_norm(v)))
        returns similarity measure [0,2]
    '''
    return pairwise_distances(u, v, metric='cosine')

def cosine_similarity(u, v=None):
    ''' 
        cosine similarity of (u, v) = dot(u,v) / dot(l2_norm(u), l2_norm(v))
        returns similarity measure [-1,1], equivalent to [0 degrees,18 0degrees]
        maybe issue: small angles tend to get very similar values(https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance)

    '''
    return 1 - pairwise_distances(u, v, metric='cosine')

def angular_distance(u, v=None):
    ''' 
        angular distance of (u, v) = arccos(cosine_similarity(u, v)) / pi
        returns dissimilarity measure [0,2]
    '''
    return np.arccos(cosine_similarity(u, v))/np.pi

def angular_similarity(u, v=None):
    ''' 
        angular similarity between two vectors = 1 - (arccos(cosine_similarity(u, v)) / pi)
        returns similarity measure [-1,1]
    '''
    return 1 - (np.arccos(cosine_similarity(u, v))/np.pi)

def binary_distances(arr, metric='jaccard'):
    ''' returns binary distances '''
    return squareform(pdist(arr, metric=metric))


#-----------------------------------------------------------------------------------------------------------
# functions for diff. shapes & waves etc 
# metric coordinate systems & conversions: cartesian, spherical, cylindrical, polar
#-----------------------------------------------------------------------------------------------------------

def magnitude_phasor(magnitude, angle, offset=0):
    ''' 
    Calculate a vectors phasor representation given its magnitude, angle, and offset
    x = magnitude * cos(angle + offset)
    - magnitude: magnitude
    - angle: angle (in degrees)
    - offset: offset (in degrees); default=0
    '''
    return magnitude * np.cos(np.deg2rad(angle + offset))

def calc_circularity(coords):
    ''' 
        calculate the circularity score of a set of coordinates
        circularity = (4 * np.pi * area) / perimeter ** 2
        bounded between 0 and 1, where 1 is a perfect circle
    '''
    # NOT SURE ABT THIS....
    # coordinates -> convex hull -> perimeter & area
    shape = scipy.spatial.ConvexHull(coords)
    perim = shape.area # when 2d: perimeter...
    area  = shape.volume # when 2d: area...
    return (4 * np.pi * area) / perim ** 2

def fit_circle(coords):
    ''' 
        fits a circle to the points that minimizes squared loss 
        returns center, radius, and sum of squared residuals
    '''
    # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((coords[:,0] - xc) ** 2 + (coords[:,1] - yc) ** 2)

    def obj(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        dists_ = calc_R(*c)
        return dists_ - dists_.mean()

    # an initial center estimate to start the optimization
    center_guess = np.mean(coords, axis=0)

    # minimize distances of points to center of mean circle
    center, ier = optimize.leastsq(obj, center_guess) 

    # figure out the fit
    xc_2, yc_2 = center
    dists = calc_R(xc_2, yc_2)
    radius = np.mean(dists) # radius of ideal circle
    resids = np.sum((dists - radius) ** 2) #  sum of squares of the diffs between dist of each point to center and mean distance

    return center, radius, resids

def circle_gof(coords, center, radius):
    ''' 
        Calculate goodness of fit for a circle
        Test angle uniformity, distance diff from radius
        If a circle, expect both tests to be non-significant
    '''

    # find the polar coordinates of the coords to the center 
    distances = pairwise_distances(coords, np.array(center)[np.newaxis])
    degrees = np.rad2deg(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))
    degrees[degrees < 0] += 360
    radians = np.deg2rad(degrees)

    # null hypothesis: angles are uniformly distributed from 0-360
    z, zp = circ_vtest(radians)

    # null hypothesis: distances are constant around the circle
    t, tp = scipy.stats.ttest_1samp(distances.flatten(), popmean=radius)

    return {'angles': [z, zp], 'distances': [t, tp]}

def calc_sphericity(points):
    # calculate the center of mass of the points
    center = np.mean(points, axis=0)
    # calculate the distances of each point from the center
    distances = np.linalg.norm(points - center, axis=1)
    # calculate the sphericity score as 1 - (standard deviation / average distance)
    # ie, spherical points should have low variance in distance from center
    score = 1 - (np.std(distances) / np.mean(distances))
    return score

def fit_sphere(X, Y, Z):
    # https://jekel.me/2015/Least-Squares-Sphere-Fit/
    # fit a sphere to X, Y, and Z data points
    # returns the radius and center points of best fit sphere

    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    # make A matrix
    A = np.zeros((len(X), 4))
    A[:,0] = X*2
    A[:,1] = Y*2
    A[:,2] = Z*2
    A[:,3] = 1

    # make f matrix
    f = np.zeros((len(X), 1))
    f[:,0] = X**2 + Y**2 + Z**2
    C, resids, rank, singval = np.linalg.lstsq(A, f) # C = Af

    # solve radius
    radius = np.sqrt(C[0]**2 + C[1]**2 + C[2]**2 + C[3])

    return radius, C[0], C[1], C[2]

def calc_polar_coords(coords, center):

    # find the polar coordinates of the coords to the center 
    distances = pairwise_distances(coords, np.array(center)[np.newaxis])
    degrees = np.rad2deg(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))
    degrees[degrees < 0] += 360
    radians = np.deg2rad(degrees)

    # stack the distances and radians together into a single array
    # polar_coords = np.hstack((distances, radians))

    return np.hstack((distances, radians[:,np.newaxis]))

def cart_to_cylindrical(x, y, z):
    ''' 
        Convert 3D cartesian coordinates to cylindrical coordinates.

        Parameters
        ----------
        x : float
            x-coordinate in the Cartesian coordinate system.
        y : float
            y-coordinate in the Cartesian coordinate system.
        z : float
            z-coordinate in the Cartesian coordinate system.

        Returns
        -------
        r : float
            Distance from the origin to the point in the cylindrical coordinate system.
        theta : float
            Angle in the x-y plane, in radians, from the positive x-axis to the projection of the point onto the x-y plane.
        z : float
            z-coordinate in the cylindrical coordinate system, unchanged from the input.
    '''
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z

def cart_to_spherical(x, y, z):
    # gotta make sure the convention is correct!
    """
        Convert Cartesian coordinates to spherical coordinates.
        https://en.wikipedia.org/wiki/Spherical_coordinate_system

        Parameters
        ----------
        x (float or ndarray): x-coordinate(s)
        y (float or ndarray): y-coordinate(s)
        z (float or ndarray): z-coordinate(s)
        
        Returns
        -------
        tuple: radial coordinate `r`, polar angle `theta`, and azimuthal angle `phi` for each point
        
        Notes:
        Expressed in physics ISO notation (math switches theta & phi: see wikipedia link above)
        The spherical coordinates of a point (x, y, z) in 3D space are given by (r, theta, phi), where:
        - `rho`: radial distance to origin
        - `theta`: polar/inclination angle, angle wrt polar axis (z-axis)
        - `phi`: azimuthal angle, angle of rotation from initial xy/meridian plane
    """

    rho   = np.sqrt(x**2 + y**2 + z**2) 
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi   = np.arctan2(y, x) 
    return rho, theta, phi

def latlong_to_cart(lat, lon):
    # https://skeptric.com/calculate-centroid-on-sphere/#geodesic-distance
    # returns in radians# returns in radians
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def cart_to_latlong(x, y, z):
    # https://skeptric.com/calculate-centroid-on-sphere/#geodesic-distance
    # returns in radians
    assert np.all(np.abs((x*x +y*y +z*z) - 1) < 1e-5)
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return np.array([lat, lon])

def geo_dist_sphere(x, y, eps=1e-6):
    # geodesic distance between two points on a sphere:
    # https://skeptric.com/calculate-centroid-on-sphere/#geodesic-distance
    # x & y should be in 3d cartesian coordinates
    dotprod = y.T @ x
    assert ((-1 - eps) <= dotprod).all() and (dotprod <= (1 + eps)).all()
    dotprod = dotprod.clip(-1, 1)
    return np.arccos(dotprod)

def fit_plane(data, order=1):

    # TODO: add higher-order polynomial fits, evaluate quality etc...
    # ref: http://inversionlabs.com/2016/03/21/best-fit-surfaces-for-3-dimensional-data.html

    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()
        
    # best-fit linear plane (1st-order)
    if order == 1:
        
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2]) # coefficients

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    # best-fit quadratic curve (2nd-order)
    elif order == 2:
        
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
            
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    return X, Y, Z
    

#---------------------------------------------------------------------------------------------------------
# circular statistics 
#---------------------------------------------------------------------------------------------------------

# TODO: not sure of the difference between "circular" & "angular" standard deviation
# astropy.stats.circstd gives same answer as pycircstat.astd, NOT pycircstat's allegeldy "circular" std function
# but sp.stats.circstd does give the same answer as pycircstat.std

# descriptive
circ_mean     = pycircstat.mean
circ_confmean = pycircstat.mean_ci_limits
circ_median   = pycircstat.median
circ_cstd     = pycircstat.std # "circular"
circ_cvar     = pycircstat.var 
circ_astd     = pycircstat.astd # "angular"
circ_avar     = pycircstat.avar 
circ_cluster  = pycircstat.clustering.AggCluster1D
circ_skewness = pycircstat.skewness
circ_kurtosis = pycircstat.kurtosis
circ_pdiff    = pycircstat.pairwise_cdiff
circ_r        = pycircstat.resultant_vector_length

# distributional tests
circ_medtest = pycircstat.medtest # only p-value
circ_hktest  = pycircstat.hktest
circ_raotest = pycircstat.raospacing
circ_wwtest  = pycircstat.watson_williams # anova


def circ_rtest(angles, axis=None, weights=None):
    """ 
        [EDITS: 
            - astropy function edited to output z-value in addition to p-value
            - tested against matlab's circtat version]

        Performs the Rayleigh test of uniformity.

        This test is  used to identify a non-uniform distribution, i.e. it is
        designed for detecting an unimodal deviation from uniformity. More
        precisely, it assumes the following hypotheses:
        - H0 (null hypothesis): The population is distributed uniformly around the
        circle.
        - H1 (alternative hypothesis): The population is not distributed uniformly
        around the circle.
        Small p-values suggest to reject the null hypothesis.

        Parameters
        ----------
        angles : numpy.ndarray or Quantity
            Array of circular (directional) data, which is assumed to be in
            radians whenever ``angles`` is ``numpy.ndarray``.
        axis : int, optional
            Axis along which the Rayleigh test will be performed.
        weights : numpy.ndarray, optional
            In case of grouped data, the i-th element of ``weights`` represents a
            weighting factor for each group such that ``np.sum(weights, axis)``
            equals the number of observations.
            See [1]_, remark 1.4, page 22, for detailed explanation.

        Returns
        -------
        z       : value of z-test
        p-value : float or dimensionless Quantity (gives different value than matlab circstat... p/2?)
    """
    # z-stat
    n = np.size(angles, axis=axis)
    Rbar = _length(angles, 1, 0.0, axis, weights)
    z = n*Rbar*Rbar

    # p-values
    tmp = 1.0
    if(n < 50):
        tmp = 1.0 + (2.0*z - z*z)/(4.0*n) - (24.0*z - 132.0*z**2.0 +
                                             76.0*z**3.0 - 9.0*z**4.0)/(288.0 *
                                                                        n * n)
    pval = np.exp(-z)*tmp
    
    return [z, pval]

def circ_vtest(angles, mu=0.0, axis=None, weights=None):
    """ 
    [EDITS: 
        - astropy function edited to output z-value in addition to p-value
        - tested against matlab's circtat version]
    
    Performs the Rayleigh test of uniformity where the alternative
    hypothesis H1 is assumed to have a known mean angle ``mu``.

    Parameters
    ----------
    angles : numpy.ndarray or Quantity
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``angles`` is ``numpy.ndarray``.
    mu : float or Quantity, optional
        Mean angle. Assumed to be known.
    axis : int, optional
        Axis along which the V test will be performed.
    weights : numpy.ndarray, optional
        In case of grouped data, the i-th element of ``weights`` represents a
        weighting factor for each group such that ``sum(weights, axis)``
        equals the number of observations. See [1]_, remark 1.4, page 22,
        for detailed explanation.

    Returns
    -------
    z       : value of z-test
    p-value : float or dimensionless Quantity

    """

    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, angles.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')

    n = np.size(angles, axis=axis)
    R0bar = np.sum(weights * np.cos(angles - mu), axis)/np.sum(weights, axis)
    z = np.sqrt(2.0 * n) * R0bar
    pz = sp.stats.norm.cdf(z)
    fz = sp.stats.norm.pdf(z)
    # see reference [3]
    pval = 1 - pz + fz*((3*z - z**3)/(16.0*n) +
                           (15*z + 305*z**3 - 125*z**5 + 9*z**7)/(4608.0*n*n))
    return [z, pval]

def circ_symtest(angles, axis=None):
    """
    [EDITED from pycircstat slightly
        - output [T, pval]
        - ]
    Non-parametric test for symmetry around the median. Works by performing a
    Wilcoxon sign rank test on the differences to the median.

    H0: the population is symmetrical around the median
    HA: the population is not symmetrical around the median

    :param angles: sample of angles in radian (prev. 'alpha')
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return T:    test statistics of underlying wilcoxon test

    References: [Zar2009]_
    """

    m = circ_median(angles, axis=axis)
    d = np.angle(np.exp(1j * m[np.newaxis]) / np.exp(1j * angles))

    if axis is not None:
        oshape = d.shape[1:]
        d2 = d.reshape((d.shape[0], int(np.prod(d.shape[1:]))))
        T, pval = map(lambda x: np.asarray(x).reshape(
            oshape), zip(*[sp.stats.wilcoxon(dd) for dd in d2.T]))
    else:
        T, pval = sp.stats.wilcoxon(d)

    return [T, pval]


#---------------------------------------------------------------------------------------------------------
# cicrular correlations & regressions
#---------------------------------------------------------------------------------------------------------


def circ_corrcc(angles1, angles2):
    '''
        rho computation from astropy library
        pvalue computation adapted froms matlab toolbox circstats
        input:
         - angles1 (in radians!)
         - angles2 (in radians!)
        output:
         - rho: correlation coefficient
         - pval based on normal distribution 
    '''
    # compute mean directions
    n = len(angles1)
    angles1_mean = circ_mean(angles1)
    angles2_mean = circ_mean(angles2)

    # compute correlation coeffcient
    rho = astropy.stats.circstats.circcorrcoef(angles1, angles2)

    # compute pvalue
    l20 = np.mean(np.sin(angles1 - angles1_mean)**2)
    l02 = np.mean(np.sin(angles2 - angles2_mean)**2)
    l22 = np.mean((np.sin(angles1 - angles1_mean)**2) * (np.sin(angles2 - angles2_mean)**2))

    # avoid division by 0 (?)
    if l22 != 0:
        ts = np.sqrt((n * l20 * l02)/l22) * rho
        pval = 2 * (1 - sp.stats.norm.cdf(np.abs(ts)))
    else: 
        pval = np.nan
    
    return [rho, pval]

def circ_corrcc_matrix(df_deg):
    '''
    TODO: speed up
    computes circular correlations between columns of dataframe 
    
    -input-
    :df: angular data (in degrees), with n observations in columns
    
    -output-
    :coefs: upper triangle of n x n correlation coefficient matrix
    :pvals: upper triangle of n x n correlation p-values matrix
    
    basic algorithm:
        1 convert degrees to radians
        2 for each column c in matrix
            2a compute correlation(c,c+1:end) (e.g., if c=0, then [0,1:end])
            2b transpose & store
        3 symmetrize 
    
    '''
    df_rad = np.deg2rad(df_deg)
    n = len(df_rad.columns)
    coefs = np.zeros((n,n))
    pvals = np.zeros((n,n))
    for c in range(n-1):
        corrs = [circ_corrcc(df_rad.iloc[:,c], df_rad.iloc[:,r]) for r in np.arange(c+1,n)] # maybe can be faster...
        coefs[c,c+1:n] = np.array(corrs).T[0] # get rid of transpose...?
        pvals[c,c+1:n] = np.array(corrs).T[1]
    return coefs, pvals

def circ_corrcl(angles, x, axis=None):
    """
    [EDITED from pycircstats
        - added p-value computation]
        
    Correlation coefficient between one circular and one linear random variable.

    :param angles: sample of angles in radians (prev 'alpha')
    :param x: sample of linear random variable
    :param axis:  compute along this dimension,
                  default is None (across all dimensions)
    :param bootstrap_iter: number of bootstrap iterations
                           (number of samples if None)
    :param ci: if not None, confidence level is bootstrapped
    :return: correlation coefficient
    """

    assert angles.shape == x.shape, "Dimensions of alpha and x must match"

    if axis is None:
        angles = angles.ravel()
        x = x.ravel()
        axis = 0

    # compute correlation coefficient for sin and cos independently
    rxs = _corr(x, np.sin(angles), axis=axis)
    rxc = _corr(x, np.cos(angles), axis=axis)
    rcs = _corr(np.sin(angles), np.cos(angles), axis=axis)

    # compute angular-linear correlation (equ. 27.47) & p-value
    rho = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))
    pval = 1 - sp.stats.chi2.cdf(len(angles)*rho**2, 2)
    return [rho, pval]

class BaseRegressor(object):
    """
    Basic regressor object. Mother class to all other regressors.
    Regressors support indexing which is passed to the coefficients.
    Regressors also support calling. In this case the prediction function is called.
    """

    def __init__(self):
        self._coef = None

    def isfit(self):
        """
        Returns whether the regressor is trained of not.

        :return: True if trained
        """
        return self._coef is not None

    def fit(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.fit not implemented".format(self.__class__.__name__))

    def test(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.test not implemented".format(self.__class__.__name__))

    def loss(self, x, y, lossfunc, ci=None, bootstrap_iter=1000):
        """
        Computes loss function between the predictions f(x) and the true y.

        :param x: inputs in radians. If multidimensional, each row must
                  be a specimen and each column a feature.
        :param y: desired outputs in radians. If multidimensional, each
                  row must be a specimen and each column a feature.
        :param lossfunc: loss function, must take an array of input and outputs and compute the loss.
        :param ci: confidence interval in [0,1]. If not None, bootstrapping is performed.
        :param bootstrap_iter: number of bootstrap iterations if
        :return: loss as computed by the loss function.
        """

        if ci is not None:
            yhat = self.predict(x)
            l = [lossfunc(y[idx], yhat[idx]) for idx in index_bootstrap(x.shape[0], bootstrap_iter)]
            mu = np.mean(l)
            q = 1 - ci
            return mu, CI(np.percentile(l, q / 2. * 100), np.percentile(l, 1 - q / 2. * 100))
        return lossfunc(y, self.predict(x))

    def predict(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.predict not implemented".format(self.__class__.__name__))

    def __getitem__(self, item):
        return self._coef.__getitem__(item)

    def __setitem__(self, key, value):
        return self._coef.__getitem__(key, value)

    def __call__(self, *args, **kwargs):
        assert self.isfit(), "Regressor must be trained first."
        return self.predict(*args, **kwargs)

class circ_linear_regression(BaseRegressor):
    """
    [EDITED lightly from pycircstat]
    Implements a circular linear regression model of the form
    .. math::
        y = m + a \\cos(\\alpha - \\alpha_0)
    The actual model is equivalently implemented as
    .. math::
        y = c_1 \\cos(\\alpha) + c_2 \\sin(\\alpha) + m

    References: [Jammalamadaka2001]_
    """

    def __init__(self):
        super(circ_linear_regression, self).__init__()

    def fit(self, angles, y):
        """
        [EDITED: moved constant to 1st column]
        Estimates the regression coefficients. Only works for 1D data.

        :param angles: independent variable, angles in radians
        :param y: dependent variable
        """
        assert angles.shape == y.shape, "y and alpha need to have the same shape"
        assert len(angles.shape) == 1, "regression only implemented for 1D data"
        assert len(y.shape) == 1, "regression only implemented for 1D data"

        X = np.c_[np.ones_like(angles), np.cos(angles), np.sin(angles)]
        self.coef_ = np.dot(np.linalg.pinv(X), y)

    def predict(self, angles):
        """
        Predicts linear values from the angles.

        :param angles: inputs, angles in radians
        :return: predictions
        """
        X = np.c_[np.ones_like(angles), np.cos(angles), np.sin(angles)]
        return np.dot(X, self.coef_)

    def test(self, angles, y):
        """
        Tests whether angles and y are significantly correlated.
        The test assumes that y is normally distributed. The test
        function uses a Shapiro-Wilk test to test this assumption.

        :param angles: independent variable, angles in radians
        :param y: dependent variable
        :return: test results of Shapiro-Wilk and Liddell-Ord test
        :rtype: pandas.DataFrame

        References: [Jammalamadaka2001]_
        """
        w, psw = sp.stats.shapiro(y)
        rxc, rxs, rcs = np.corrcoef(y, np.cos(angles))[0,1], np.corrcoef(y, np.sin(angles))[0,1], \
                        np.corrcoef(np.cos(angles), np.sin(angles))[0,1]
        n = len(angles)
        r2 = (rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1 - rcs**2)
        f = (n-3)*r2/(1-r2)
        p = sp.stats.f.sf(f, 2, n-3)

        df = pd.DataFrame(dict(
            test = ['Shapiro-Wilk','Liddell-Ord'],
            statistics = [w, f],
            p = [psw, p],
            dof = [None, (2, n-3)]
        )).set_index('test')
        return df 

class circ_circ_regression(BaseRegressor):
    """
    [EDITED lightly from pycircstat]
    Implements a circular circular regression model of the form
    .. math::
        \\cos(\\beta) = a_0 + \\sum_{k=1}^d a_k \\cos(k\\alpha) + b_k \\sin(k\\alpha)
        \\sin(\\beta) = c_0 + \\sum_{k=1}^d c_k \\cos(k\\alpha) + d_k \\sin(k\\alpha)

    The angles :math:`\\beta` are estimated via :math:`\\hat\\beta = atan2(\\sin(\\beta), \\cos(\\beta))`
    :param degree: degree d of the trigonometric polynomials
    References: [Jammalamadaka2001]_
    """

    def __init__(self, degree=3):
        super(circ_circ_regression, self).__init__()
        self.degree = degree

    def fit(self, angles_x, angles_y):
        """
        Estimates the regression coefficients. Only works for 1D data.

        :param angles_x: independent variable, angles in radians (prev. alpha)
        :param angles_y: dependent variable, angles in radians (prev. beta)
        """
        X = np.vstack([np.ones_like(angles_x)] + [np.cos(angles_x*k) for k in np.arange(1., self.degree+1)] \
                                  + [np.sin(angles_x*k) for k in np.arange(1., self.degree+1)]).T
        self.coef_ = np.c_[np.dot(np.linalg.pinv(X), np.cos(angles_y)),
                           np.dot(np.linalg.pinv(X), np.sin(angles_y))]

    def predict(self, angles_x):
        """
        Predicts linear values from the angles.

        :param angles_x: inputs, angles in radians
        :return: predictions, angles in radians
        """
        X = np.vstack([np.ones_like(angles_x)] + [np.cos(angles_x*k) for k in np.arange(1., self.degree+1)] \
                                  + [np.sin(angles_x*k) for k in np.arange(1., self.degree+1)]).T
        preds = np.dot(X, self.coef_)
        angle_preds = np.arctan2(preds[:,1], preds[:,0]) # angularize the linear predictions
        return angle_preds


#---------------------------------------------------------------------------------------------------------
# plotting
#---------------------------------------------------------------------------------------------------------


def vector_plot(ax, xy_comp, xy_unit, cluster_ids=None):
    '''
        plot the x & y components and unit circle values 
        uses matplotlibs quiver()
        Example
        -------
        fig, ax = plt.subplots() 
        n,bins,patches = vector_plot(ax, xy_comp, xy_unit)
    '''
    
    if cluster_ids is None:
        colors_ = "black"
    else:
        colors = random_colors(num_colors=len(np.unique(cluster_ids)))
        colors_ = [colors[c] for c in cluster_ids]
    ax.quiver(xy_comp[:,0], xy_comp[:,1], 
              xy_unit[:,0], xy_unit[:,1],
              color=colors_) 
    ax.axis([-6, 6, -6, 6]) 
    b = ax.get_xgridlines()[3]
    b.set_color('black')
    b.set_linewidth(1)
    b = ax.get_ygridlines()[3]
    b.set_color('black')
    b.set_linewidth(1)   

def random_colors(num_colors=10):
    from random import randint
    colors = ['#%06X' % randint(0, 0xFFFFFF) for i in range(num_colors)]
    return colors

def cluster_vector_plot(a, r, cluster_ids=None):
    '''
        plot the x & y components and unit circle values, colored by cluster id
    '''
    if cluster_ids is None:
        colors_ = "black"
    else:
        colors = random_colors(num_colors=len(np.unique(cluster_ids)))
        colors_ = [colors[c] for c in cluster_ids]
    xy_comp, xy_unit = circ_vectors(a, r)
    vector_plot(xy_comp, xy_unit, color=colors_)

def circular_hist(ax, x, color='blue', n_bins=16, density=True, offset=0, gaps=True):
    """
    TO DO: add mean circular angle...
    
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.


    Example
    -------
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar')) 
        n,bins,patches = circular_hist(ax, radians)
    """

    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        n_bins = np.linspace(-np.pi, np.pi, num=n_bins+1)

    # Bin data and record counts
    n, n_bins = np.histogram(x, bins=n_bins)

    # Compute width of each bin
    widths = np.diff(n_bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(n_bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, n_bins, patches