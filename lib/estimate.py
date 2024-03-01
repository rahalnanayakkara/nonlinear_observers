import numpy as np
from moving_polyfit.moving_ls import PolyEstimator
from func import generate_lagrange, deriv_bound

def get_poly_estimates(t_samples, y_samples, Y_max, window_size, sampling_dt, d, delay, nderivs, num_steps_per_sample):

    estimator = PolyEstimator(d=d, N=window_size, dt=sampling_dt)
    lagrange_pols, l_indices = generate_lagrange(d=d, N=window_size, sampling_dt=sampling_dt)
    num_samples = len(y_samples)

    y_hat = np.zeros((nderivs, num_samples+1))
    y_bound = np.zeros((nderivs, num_samples+1))

    # Sliding window of above size
    for i in range(window_size-1, num_samples):
        t0 = t_samples[i-window_size+1]
        estimator.fit(y_samples[i-window_size+1:i+1])
        res_pol = np.array(lagrange_pols)@estimator.residuals[l_indices]
        t = np.linspace(window_size-2-delay, window_size-1-delay, num_steps_per_sample+1)
        for j in range(nderivs):
            y_hat[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1] = estimator.differentiate(t, j)
            y_bound[j, (i-delay-1)*num_steps_per_sample : (i-delay)*num_steps_per_sample+1]  = np.abs(res_pol.deriv(j)(t))+deriv_bound(k=j, d=d, M=Y_max[d+1], delta_s=sampling_dt)

    return y_hat, y_bound