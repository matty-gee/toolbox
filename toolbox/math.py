import numpy as np

def exponential_decay(init, decay, time_steps):
    # f(t) = i(1 - r) ** t
    return np.array([init * (1 - decay) ** step for step in range(time_steps)])


def linear_decay(init, decay, time_steps):
    # f(t) = i - r * t
    # dont let it decay past 0
    return np.array([max((0, init - decay * step)) for step in range(time_steps)])


def running_mean(X, step=2):
    ''' 
        a mean over a period of time
    '''
    if type(X) not in [list, np.ndarray]: 
        X = X.values
    cumsum = np.cumsum(np.insert(X, 0, 0)) 
    return (cumsum[step:] - cumsum[:-step]) / float(step)


def cumulative_mean(X):
    cumsum = np.cumsum(X, 0)
    cumnum = np.cumsum(np.arange(1,len(X)+1, 0))
    return cumsum / cumnum
                       