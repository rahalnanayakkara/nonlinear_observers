import sys
sys.path.append('./')

from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.epidem_odes import UIV
from moving_gauss import GaussEstimator
from lib.simulate import *
from lib.estimate import *
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.polynomial import Polynomial as Pol
from matplotlib.ticker import ScalarFormatter


uiv_ode = UIV(beta=1, p=2)
params = {}
params["beta"] = uiv_ode.beta
params["p"] = uiv_ode.p_p
params["c"] = uiv_ode.c
params['delta'] = uiv_ode.delta


n = uiv_ode.n
m = uiv_ode.m
p = uiv_ode.p

def get_some(sampling_dt=0.25):
    num_days = 30
    integration_dt = 0.01

    U0 = 4
    V0 = 50e-8
    # x0 = [U0, 0, 0]
    x0 = [U0, 0, V0]

    nderivs = uiv_ode.nderivs

    N = 6           # Number of samples in a window
    n_gauss = 5     # Number of Gaussian Functions

    delay = 1


    time, x, y_d, t_samples, y_samples = simulate(uiv_ode, x0, integration_dt, sampling_dt, num_days)

    MAX_DERIVS = 5

    Y_max = [0.8, 0.4, 0.4, 0.5, 0.9, 1.7]

    # Polynomial Bounds
    d=4
    y_hat_p, y_bound_p = get_poly_estimates(t_samples, y_samples, Y_max, N, sampling_dt, d, delay, nderivs, integration_dt)

    # Gaussian Bounds
    if sampling_dt>0.5:
        d=-1
    y_hat_g, y_bound_g = get_gauss_estimates(t_samples, y_samples, Y_max, N, sampling_dt, n_gauss, delay, nderivs, integration_dt, d0=d)


    x2 = uiv_ode.invert_output2(t=0, y_d=y_d)
    x2_hat_g = uiv_ode.invert_output2(t=0, y_d=y_hat_g)
    x2_hat_p = uiv_ode.invert_output2(t=0, y_d=y_hat_p)

    x2_bound_g = uiv_ode.bound_tf(y_bound_g)
    x2_bound_p = uiv_ode.bound_tf(y_bound_p)


    x_hat_g = uiv_ode.state_tf(x2_hat_g)
    x_hat_p = uiv_ode.state_tf(x2_hat_p)

    x_g_min, x_g_max = uiv_ode.state_tf_bound(x2_hat_g, x2_bound_g)
    x_p_min, x_p_max = uiv_ode.state_tf_bound(x2_hat_p, x2_bound_p)

    return time, x, x_hat_g, x_g_min, x_g_max


sampling_dts = [0.25, 0.5]
sampling_Ts = [int(sampling_dt*24) for sampling_dt in sampling_dts]
x_hat_g_list = []
x_g_min_list = []
x_g_max_list = []
for sampling_dt in sampling_dts:
    time, x, x_hat_g, x_g_min, x_g_max = get_some(sampling_dt=sampling_dt)
    x_hat_g_list.append(x_hat_g)
    x_g_min_list.append(x_g_min)
    x_g_max_list.append(x_g_max)

states = ["U", "I", "V"]

ti=1000
tf=2000

plt.figure(figsize=(12,6))
# plt.suptitle(f"Sampling time = {sampling_dt}")

time_p = time-time[ti]

for i in range(n):
    plt.subplot(3,1,i+1)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    formatter = ScalarFormatter(useMathText=True)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.get_offset_text().set_position((-0.05, 0))
    # plt.plot(time_p[ti:tf], 1e8*x[i][ti:tf], label="True State")
    # plt.plot(time_p, 1e8*x[i], label="True State")
    for j in range(len(sampling_dts)):
        plt.plot(time_p, np.abs(1e8*x_hat_g_list[j][i]-1e8*x[i]), label="Estimation error for T="+str(sampling_Ts[j])+"hrs")
        plt.fill_between(time_p, 0, 1e8*x_g_max_list[j][i]-1e8*x[i], alpha=0.3, label='Error Bound for T='+str(sampling_Ts[j])+"hrs")
        # plt.fill_between(time_p, 1e8*x_g_min_list[j][i], 1e8*x_g_max_list[j][i], alpha=0.3, color='blue', label='Gaussian Bound '+str(sampling_dts[j]))
    # plt.plot(time, x_hat_p[i], label="Polynomial estimate")
    # plt.fill_between(time, x_p_min[i], x_p_max[i], alpha=0.3, color='green', label='Polynomial Bound')
    # if i==2:
    #     plt.scatter(t_samples_p, y_samples*1e8)
    if i==0:
        plt.ylim([0.0, min((np.max(x[i])*1.2),6)*1e8])
    plt.xlim([time_p[ti], time_p[tf]])
    plt.grid()
    plt.ylabel(states[i])
    plt.legend()
plt.tight_layout()
plt.show()
