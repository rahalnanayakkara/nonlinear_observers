import numpy as np
from moving_polyfit.moving_ls import PolyEstimator
from moving_gauss import GaussEstimator
from lib.func import generate_lagrange, deriv_bound
import matplotlib.pyplot as plt

def get_poly_estimates(t_samples, y_samples, Y_max, window_size, sampling_dt, d, delay, nderivs, integration_dt):

    estimator = PolyEstimator(d=d, N=window_size, dt=sampling_dt)
    lagrange_pols, l_indices = generate_lagrange(d=d, N=window_size, sampling_dt=sampling_dt)
    num_samples = len(y_samples)
    num_steps_per_sample = int(sampling_dt/integration_dt)
    num_steps = (num_samples-1)*num_steps_per_sample

    y_hat = np.zeros((nderivs, num_steps+1))
    y_bound = np.zeros((nderivs, num_steps+1))

    # Sliding window of above size
    for i in range(window_size-1, num_samples):
        t0 = t_samples[i-window_size+1]
        estimator.fit(y_samples[i-window_size+1:i+1])
        res_pol = np.array(lagrange_pols)@estimator.residuals[l_indices]
        t = np.linspace(window_size-2-delay, window_size-1-delay, num_steps_per_sample+1)*sampling_dt
        for j in range(nderivs):
            y_hat[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1] = estimator.differentiate(t, j)
            y_bound[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1]  = np.abs(res_pol.deriv(j)(t))+deriv_bound(k=j, d=d, M=Y_max[d+1], delta_s=sampling_dt)
            # print(j, d, Y_max[d+1], deriv_bound(k=j, d=d, M=Y_max[d+1], delta_s=sampling_dt))

    return y_hat, y_bound

def get_gauss_estimates(t_samples, y_samples, Y_max, window_size, sampling_dt, n, delay, nderivs, integration_dt, d=-1):

    num_samples = len(y_samples)
    num_steps_per_sample = int(sampling_dt/integration_dt)
    num_steps = (num_samples-1)*num_steps_per_sample

    estimator = GaussEstimator(n=n, N=window_size, dt=sampling_dt)
    G_max = np.zeros(max(d+2,nderivs+1))
    
    d_list = list(range(nderivs))   # to estimate kth derivative, we need d=k
    d_list[0] = 1   # for 0th derivative we still need d=1

    lagrange_pols = []  # store lagrange polynomials for each case of d
    l_indices = []      # store indices of set D for each case of d
    for d_pol in d_list:
        lp, li = generate_lagrange(d=d_pol, N=window_size, sampling_dt=sampling_dt)
        lagrange_pols.append(lp)
        l_indices.append(li)
    
    y_hat = np.zeros((nderivs, num_steps+1))
    y_bound = np.zeros((nderivs, num_steps+1))

    # Sliding window of above size
    for i in range(window_size-1, num_samples):
        t0 = t_samples[i-window_size+1]
        estimator.fit(y_samples[i-window_size+1:i+1])

        t = np.linspace(window_size-2-delay, window_size-1-delay, num_steps_per_sample+1)*sampling_dt
        for j in range(max(d+2,nderivs+1)):
            G_max[j] = np.max(np.abs(estimator.differentiate(t, j)))

        for j in range(nderivs):
            res_pol = np.array(lagrange_pols[j])@estimator.residuals[l_indices[j]]
            y_hat[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1] = estimator.differentiate(t, j)
            d = max(j,1) if d<1 else d
            y_bound[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1]  = np.abs(res_pol.deriv(j)(t))+deriv_bound(k=j, d=d, M=Y_max[d+1]+G_max[d+1], delta_s=sampling_dt)
            # print(j, d, Y_max[d+1]+G_max[d+1], deriv_bound(k=j, d=d, M=Y_max[d+1]+G_max[d+1], delta_s=sampling_dt))

    return y_hat, y_bound