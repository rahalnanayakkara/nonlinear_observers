import numpy as np
from lib.moving_poly import PolyEstimator
from lib.moving_gauss import GaussEstimator
from lib.func import generate_lagrange, deriv_bound
import matplotlib.pyplot as plt

def get_poly_estimates(t_samples, y_samples, nderivs, d, Y_max, window_size, delay=1, integration_dt=0, E=0):
    '''
    Args:
        t_samples       : Array of sampling times
        y_samples       : Array of sampled measurements
        nderivs         : Max derivative to be estimated
        d               : Degree of polynomial to fit (Must be greater than or equal to nderivs)
        Y_max           : Array of bounds for derivatives of y, upto d+1. Only Y_max[d+1] is used in this case
        window_size     : Size of sliding window for polynomial fitting
        delay           : Delay (measured in number of samples) to estimate derivatives
        integration_dt  : If continuous time approximation is needed
        E               : Maximum noise magnitude
    '''
    d = nderivs if d==0 else d
    sampling_dt = t_samples[1]-t_samples[0]
    integration_dt = sampling_dt if integration_dt==0 else integration_dt

    estimator = PolyEstimator(d=d, N=window_size, dt=sampling_dt)
    
    lagrange_pols, l_indices = generate_lagrange(d=d, N=window_size, sampling_dt=sampling_dt)
    lagrange_pols = np.array(lagrange_pols)

    num_samples = len(y_samples)
    num_steps_per_sample = int(sampling_dt/integration_dt)
    num_steps = (num_samples-1)*num_steps_per_sample

    y_hat = np.zeros((nderivs+1, num_steps+1))
    y_bound = np.zeros((nderivs+1, num_steps+1))

    # Sliding window of above size
    for i in range(window_size-1, num_samples):
        estimator.fit(y_samples[i-window_size+1:i+1])
        res_pol = lagrange_pols @ estimator.residuals[l_indices]
        t = np.linspace(window_size-2-delay, window_size-1-delay, num_steps_per_sample+1)*sampling_dt
        for j in range(nderivs+1):
            y_hat[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1] = estimator.differentiate(t, j)
            y_bound[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1]  = np.abs(res_pol.deriv(j)(t))+deriv_bound(k=j, d=d, M=Y_max[d+1], delta_s=sampling_dt)+np.sum(np.abs(np.array([pol.deriv(j)(t) for pol in lagrange_pols])), axis=0)*E

    return y_hat, y_bound


def get_gauss_estimates(t_samples, y_samples, nderivs, n, Y_max, window_size, delay=1, integration_dt=0, E=0, d0=-1, sigma=1, reg=0):
    '''
    Args:
        t_samples   : Array of sampling times
        y_samples   : Array of sampled measurements
        nderivs     : Max derivative to be estimated
        n           : Number of Guassian functions to fit
        Y_max       : Array of bounds for derivatives of y, upto nderivs+1.
        window_size : Size of sliding window
        delay       : Delay (measured in number of samples) to estimate derivatives
        integration_dt : If continuous time approximation is needed
        E           : Maximum noise magnitude
        d0          : Use fixed values for d
        sigma       : Relative std deviation
        reg         : Regularization weight to use in fitting
    '''
    sampling_dt = t_samples[1]-t_samples[0]
    integration_dt = sampling_dt if integration_dt==0 else integration_dt
    num_samples = len(y_samples)
    num_steps_per_sample = int(sampling_dt/integration_dt)
    num_steps = (num_samples-1)*num_steps_per_sample

    estimator = GaussEstimator(n=n, N=window_size, dt=sampling_dt, sigma=sigma)
    G_max = np.zeros(max(d0+2, nderivs+2))
    
    d_list = list(range(1, nderivs+2))   # to estimate kth derivative, we need d=k+1
    if d0>0:    # if d is specified use it for all derivatives
        for i in range(len(d_list)):
            d_list[i] = d0

    lagrange_pols = []  # store lagrange polynomials for each case of d
    l_indices = []      # store indices of set D for each case of d
    for d_pol in d_list:
        lp, li = generate_lagrange(d=d_pol, N=window_size, sampling_dt=sampling_dt)
        lagrange_pols.append(lp)
        l_indices.append(li)
    
    y_hat = np.zeros((nderivs+1, num_steps+1))
    y_bound = np.zeros((nderivs+1, num_steps+1))

    # Sliding window of above size
    for i in range(window_size-1, num_samples):
        t0 = t_samples[i-window_size+1]
        estimator.fit(y_samples[i-window_size+1:i+1], reg=reg)

        t = np.linspace(window_size-2-delay, window_size-1-delay, num_steps_per_sample+1)*sampling_dt
        for j in range(len(G_max)):
            G_max[j] = np.max(np.abs(estimator.differentiate(t, j)))
        # Y_max[3] = np.minimum(1.6*np.max(estimator.estimate(t))+0.001, 0.5)
        for j in range(nderivs+1):
            res_pol = np.array(lagrange_pols[j])@estimator.residuals[l_indices[j]]
            y_hat[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1] = estimator.differentiate(t, j)
            d = max(j,1) if d0<1 else d0
            Y_d = np.max(np.abs(Y_max[d+1][i-window_size+1+l_indices[j][0]:i-window_size+1+l_indices[j][-1]])) if isinstance(Y_max[d+1], np.ndarray) else Y_max[d+1]
            y_bound[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1]  = np.abs(res_pol.deriv(j)(t))+deriv_bound(k=j, d=d, M=Y_d+G_max[d+1], delta_s=sampling_dt)+np.sum(np.abs(np.array([pol.deriv(j)(t) for pol in lagrange_pols[j]])), axis=0)*E
            # print(j, d, Y_max[d+1]+G_max[d+1], deriv_bound(k=j, d=d, M=Y_max[d+1]+G_max[d+1], delta_s=sampling_dt))

    return y_hat, y_bound