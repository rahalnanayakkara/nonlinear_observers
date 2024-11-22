import numpy as np
from numpy.polynomial import Polynomial as Pol
import math

def quad_roots(a, b, c):
    delta = np.sqrt(b**2-4*a*c)
    sol1, sol2 = (-b+delta)/(2*a), (-b-delta)/(2*a)
    sol1[a==0] = -c[a==0] / b[a==0]
    sol2[a==0] = -c[a==0] / b[a==0]
    return sol1, sol2


def generate_lagrange(d, N, sampling_dt, verbose_lagrange=False):
    '''
    Generate lagrange polynomial to estimate dth derivative error
    Input:
        d - number of points to consider for residual polynomial
        N - window size
        sampling_dt - sampling time
    Returns:
        lagrange_pols - list of Lagrange polynomials as np.Polynomial objects
        l_indices - indices of the window corresponding to the selected points
    '''
    num_t_points = d + 1 # number of points for the residual polynomial
    window_times = np.linspace(0., N*sampling_dt, N, endpoint=False)
    l_indices = np.arange(N-1-d, N, 1) # indices of the window which we pick for D
    l_times = window_times[l_indices] # times corresponding to D (s0, ..., sd)
    lagrange_pols = []
    for i in range(num_t_points):
        # build the lagrange polynomial, which is zero at all evaluation samples except one
        evals = np.zeros(num_t_points)
        evals[i] = 1.0  # we are choosing the data points that are closest to our evaluation point
        l_i = Pol.fit(l_times, evals, d)
        lagrange_pols.append(l_i)

        # to checking that you built the right lagrange polynomial, evaluate it at the relevant points
        if verbose_lagrange:
            for j in range(num_t_points):
                print(f't = {l_times[j]:.3f}, l_{i}(t) = {l_i(l_times[j])}')

        # t = np.linspace(l_times[0], l_times[-1], (l_indices[-1]-l_indices[0])*num_integration_steps)
        # for i in range(num_t_points):            ## Plotting Lagrange Polynomials
        #     plt.plot(t, lagrange_pols[i](t))
        # plt.grid()
        # plt.show()
                
    return lagrange_pols, l_indices

def deriv_bound(k, d, M, delta_s=1):
    '''
    Function to compute the offline term of the error bound
    Inputs:
        k - derivative being estimated
        d - 
    '''
    if k==0:
        return 0.25*np.power(delta_s, d+1)*M/(d+1)
    return np.power(delta_s, d-k+1)*M*math.comb(d, k-1)