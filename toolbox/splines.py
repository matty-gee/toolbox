# import all libraries used in this file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, BSpline
from scipy.spatial import distance
from scipy import stats



#------------------------------------------------------------
# fitting splines
#------------------------------------------------------------

# TODO: https://stackoverflow.com/questions/51321100/python-natural-smoothing-splines


class Spline:

    def __init__(self, s=0, k=3, w=None, t=None, 
                 extrapolate=False, periodic=False, sort=False):

        self.k = k # degree
        self.s = s # smoothing factor
            # Positive smoothing factor used to choose the number of knots. Number of 
            # knots will be increased until the smoothing condition is satisfied:
            # sum((w[i]*(y[i]-s(x[i])))**2,axis=0) <= s
        self.w = w # weights for weighted least-squares spline fit, length of the data; defaults to 1s        
        self.t = t # knots
        if t is None:
            self.task = 0 # find t & c for a given s 
        else:
            self.task = -1 # find weighted least sqr spline for t
            assert len(t) >= 2*k+2, \
                f'Not enough knots, need at least {2*k+2}'
        self.extrapolate = extrapolate
        self.periodic = periodic
        self.sort = sort 
    
    def fit(self, points):

        # transpose if necessary: the function expects (n_dims, n_points)
        # if points.shape[0] > points.shape[1]: 
        #     print('Transposing input')
        #     points = points.T

        self.points = points
        n_points = self.points.shape[1]

        # if periodic spline, add a point at the end that is the same as the first
        if self.periodic:
            self.points = np.hstack((self.points, 
                                     self.points[:, 0].reshape(-1, 1)))
        
        if self.sort: # sort points smallest to largest x, else leave as is
            self.points = self.points[:, np.argsort(self.points[0, :])]
        ndim = self.points.shape[0] # dimensionality of input space

        #-------------------------------------------------------------------------------
        # fit a parameterized cubic B-spline ("basis spline") to a set of data points
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html

        # B-spline: a piecewise polynomial function of degree k
        # - fitted w/ least squares: min. sum of squared residuals of spline approx. to data
        # - 'knots' connect the pieces, define 'control points' that control curves

        # inputs:
        # - points: array of data points (up to 10D)
        # - k : degree of the spline (3 = cubic)
        # - s : smoothing factor, balances fit against smoothness
        # - w : weights for weighted least-squares spline fit

        # returns:
        # - tck : tuple of knots, coefficients, and degree k of the spline
        # -- knots : points at which pieces of curve meet 
        # -- coefs : coefficients of the spline pieces 
        # - u : parameter values corresponding to x, y, [z]
        #-------------------------------------------------------------------------------

        tck, self.u_params = splprep(self.points, 
                                     s=self.s, 
                                     k=self.k, 
                                     w=self.w, 
                                     t=self.t,
                                     per=self.periodic,
                                     task=self.task,
                                     full_output=False)
        self.knots, self.coefs, _ = tck 

        #-------------------------------------------------------------------------------------------
        # knots: breakpoints
        #-------------------------------------------------------------------------------------------

        self.m = len(self.knots)
        if (self.s == 0) & (not self.periodic): # only true if s=0
            assert self.m == n_points + self.k + 1, \
                f'm != len(x) + k + 1: {self.m} != {n_points + self.k + 1}'

        # evaluate the spline for the knots to get x,y,z coordinates of knots
        knots_coords = splev(self.knots, tck)
        self.knots_coords = np.vstack(knots_coords).T
        if (self.s == 0) & (not self.periodic):
            assert all([len(knots_coords[i]) == len(self.knots) == n_points + self.k + 1 for i in range(ndim)]), \
                f'knots != len(x) + k + 1: {len(knots_coords)} != {n_points + self.k + 1}'

        # internal & clamped knots (not sure if this is true for periodic too?)
        # - internal knots: not at the beginning or end of the spline
        # - clamped knots: of number k at the beginning or end of the spline
        self.knots_internal = self.knots_coords[self.k+1 : self.m-self.k-1]
        self.knots_clamped  = self.knots_coords[0: self.k+1], self.knots_coords[-self.k-1:]

        #-------------------------------------------------------------------------------------------
        # coefficients for control points
        # - num of control points should equal number of coefficients
        #-------------------------------------------------------------------------------------------

        self.coefs = np.asarray(self.coefs)
        assert len(self.coefs) == ndim, \
            f'coefs != ndim: {len(self.coefs)} != {ndim}'

        for i in range(len(self.coefs)):
            ni = len(self.coefs[i])
            assert ni == len(self.knots) - 1 - self.k, \
                f'coefs != len(knots) - 1 - k: {ni} != {len(self.knots) - 1 - self.k}'
        self.n = len(self.coefs[0]) 

        #----------------------------------
        # u parameter values
        #----------------------------------

        if not self.periodic:
            assert len(self.u_params) == n_points, \
                f'u != len(x): {len(self.u_params)} != {n_points}'

        #-------------------------------------------------------------------------------------------
        # p = degree of spline = m [len(knots)] - n [len(coefs)] - 1
        #-------------------------------------------------------------------------------------------

        assert self.m - self.n - 1 == self.k, \
            f'p != k: {self.m - self.n - 1} != {self.k}'

        #-------------------------------------------------------------------------------------------
        # create BSpline object
        # - BSpline by default extrapolates the first and last polynomial pieces of B-spline functions active on base interval
        # - note: backwards compatability issue between splprep & BSpline means have to transpose coefs - https://github.com/scipy/scipy/issues/10389
        #-------------------------------------------------------------------------------------------

        self.spline_f = BSpline(self.knots, self.coefs.T, self.k, extrapolate=self.extrapolate, axis=0) 
        check = np.sum(self.spline_f(self.u_params) - self.points.T) # should be ~equal to inputted points
        assert np.allclose(check, 0, atol=0.01), f'check != 0: {check} != 0'

        return self.spline_f, self.u_params
        
    def evaluate_spline(self, n=100):
        # find a set of locations [0,1] of the spline function 
        # self.spline_f.__call__(x)
        self.eval_domain = np.linspace(0, 1, n) # 0-1, n points
        self.eval = self.spline_f(self.eval_domain) # generate points in domain to evaluate the spline
        return self.eval

    def evaluate_point(self, x):
        # find the location of a single point on the spline function
        # self.spline_f.__call__(x)
        return self.spline_f(x)

    def get_control_points(self):
        
        #-------------------------------------------------------------------------------------------
        # define coordinates of B-spline control points in knot space
        # http://vadym-pasko.com/examples/spline-approx-scipy/example1.html
        #-------------------------------------------------------------------------------------------

        n  = len(self.knots) - 1 - self.k
        cx = np.zeros(n) # control points
        for i in range(n):
            tsum = 0
            for j in range(1, self.k+1):
                tsum += self.knots[i+j]
            cx[i] = float(tsum) / self.k
        return cx
    
    @staticmethod
    def distance_to_origin(points):

        #-------------------------------------------------------------------------------------------
        # distance between points along curve/trajectory
        # TODO: integrate between all points in points?
        # approximation: sum distances between all consecutive points
        #-------------------------------------------------------------------------------------------
        
        points = np.vstack(points).T
        dists = [np.linalg.norm(points[i] - points[i+1]) for i in range(len(points)-1)]
        cum_dists = np.cumsum(dists)
        return dists, cum_dists


def fit_splines(Xs, s=0, k=3, sort=False, n=1000):

    ''' fit a list of splines, evaluate to a list of coordinates '''

    out = {} # dictionary output
    Xs = np.array(Xs)
    n_spls = Xs.shape[0] # expects list of coordinates

    # fit & evaluate each spline in list
    for i in range(n_spls):
        spl = Spline(s=s, k=k, sort=sort)
        _, u_params = spl.fit(Xs[i].T) 

        out[f'spline_{i}'] = {'X': Xs[i], 'spline_obj': spl, 
                              'u_evaluated': spl.evaluate_point(u_params), # shape = (n u params, n dims)
                              'spl_evaluated': spl.evaluate_spline(n=n)} # shape = (n points, n dims)
    return out


#------------------------------------------------------------
# parameterize splines
#------------------------------------------------------------


def parameterize_spline(decoder_dict, values):

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
