import sys
sys.path.append('./')

from nonlinear_system.epidem_odes import UIV
from lib.simulate import *
from lib.estimate import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

num_days = 30
sampling_dt = 0.25
integration_dt = 0.01
uiv_ode = UIV(beta=1, p=2)

n = uiv_ode.n
U0 = 4
V0 = 50e-8
x0 = [U0, 0, V0]
nderivs = uiv_ode.nderivs
N = 6           # Number of samples in a window
n_gauss = 5     # Number of Gaussian Functions
delay = 1

time, x, y_d, t_samples, y_samples = simulate(uiv_ode, x0, integration_dt, sampling_dt, num_days)

Y_max = [0.8, 0.4, 0.4, 0.5, 0.9, 1.7]
d=4
y_hat_p, y_bound_p = get_poly_estimates(t_samples, y_samples, Y_max, N, sampling_dt, d, delay, nderivs, integration_dt)
y_hat_g, y_bound_g = get_gauss_estimates(t_samples, y_samples, Y_max, N, sampling_dt, n_gauss, delay, nderivs, integration_dt, d0=d)

states2 = ["V", "I", "UV"]
x2 = uiv_ode.invert_output2(t=0, y_d=y_d)
x2_hat_g = uiv_ode.invert_output2(t=0, y_d=y_hat_g)
x2_hat_p = uiv_ode.invert_output2(t=0, y_d=y_hat_p)
x2_bound_g = uiv_ode.bound_tf(y_bound_g)
x2_bound_p = uiv_ode.bound_tf(y_bound_p)

states = ["U", "I", "V"]
x_hat_g = uiv_ode.state_tf(x2_hat_g)
x_hat_p = uiv_ode.state_tf(x2_hat_p)
x_g_min, x_g_max = uiv_ode.state_tf_bound(x2_hat_g, x2_bound_g)
x_p_min, x_p_max = uiv_ode.state_tf_bound(x2_hat_p, x2_bound_p)

ti=1000
tf=2000
plt.figure(figsize=(12,6))
time_p = time-time[ti]
t_samples_p = t_samples-time[ti]
for i in range(n):
    plt.subplot(3,1,i+1)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    plt.gca().yaxis.get_offset_text().set_position((-0.03, 0))
    plt.plot(time_p, 1e8*x[i], label="True State")
    plt.plot(time_p, 1e8*x_hat_g[i], label="Gaussian Estimate")
    plt.fill_between(time_p, 1e8*x_g_min[i], 1e8*x_g_max[i], alpha=0.3, color='blue', label='Gaussian Bound')
    plt.ylim([0.0, min((np.max(x[i])*1.2),6)*1e8])
    plt.xlim([time_p[ti], time_p[tf]])
    plt.grid()
    plt.ylabel(states[i])
    plt.legend()

plt.tight_layout()
plt.show()
