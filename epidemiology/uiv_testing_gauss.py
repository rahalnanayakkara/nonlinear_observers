import sys
sys.path.append('./')

from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.epidem_odes import UIV
from moving_gauss import GaussEstimator
import numpy as np
import matplotlib.pyplot as plt

'''
Simulate infection spread in a single host using UIV model.
Obtain an estimate of the internal states by using a Gaussian observer on V.

delay - selects the delay to evaluate the gaussian observer
infection_step - step in which to initiate viral load. 
                Non zero allows observer to run before infection begins.
'''

verbose = False

sampling_dt = 1
integration_per_sample = 100
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 30
num_integration_steps = num_sampling_steps*integration_per_sample

uiv_ode = UIV(beta=1)
beta = uiv_ode.beta
p_p = uiv_ode.p_p
n = uiv_ode.n
m = uiv_ode.m
p = uiv_ode.p

U0 = 4
V0 = 50e-8
x0 = [U0, 0, 0]

nderivs = uiv_ode.nderivs

d = 6
N = 6

delay = 1
infection_step = 0  # default is 0
estimator = GaussEstimator(d, N, sampling_dt)

x = np.zeros((n, num_integration_steps))
y_d = np.zeros((nderivs, num_integration_steps))

y_samples = np.zeros((nderivs, num_sampling_steps))
y_hat = np.zeros((nderivs, num_sampling_steps))
x_samples = np.zeros((n, num_sampling_steps))
x_hat = np.zeros((n, num_sampling_steps))

integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))

x[:, 0] = x0
x_samples[:, 0] = x0

sys = ContinuousTimeSystem(uiv_ode, x0=x0, dt=integration_dt)

y_d[:, 0] = sys.y
y_samples[:, 0] = sys.y

for t in range(0, num_sampling_steps):

    if t==infection_step:
        sys.reset(x0= [U0, 0, V0],t=sys.t)

    for i in range(integration_per_sample):
        idx = t*integration_per_sample + i
        x[:, idx], y_d[:, idx] = sys.step(0)
        integration_time[idx] = sys.t
    
    sampling_time[t] = sys.t
    y_samples[:,t] = sys.y
    x_samples[:,t] = sys.x

    if t >= N-1:
        # fit gaussian observer
        estimator.fit(y_samples[0, t-N+1:t+1])

        # estimate with polynomial derivatives with delay
        for i in range(nderivs):
            y_hat[i, t-delay] = estimator.differentiate((N-delay-1)*sampling_dt, i)[0]
        
        x_hat[:, t-delay] = uiv_ode.invert_output(t=t-delay, y_d=y_hat[:, t-delay])
        
        if verbose:
            print(f"On day {t} we estimate day {t-delay}")


states = ["U", "I", "V"]

f1 = plt.figure("State Evolution", figsize=(12,8))
for i in range(n):
    ax = f1.add_subplot(1,n,i+1)
    ax.plot(integration_time, x[i,:], label=states[i])
    if i==2:
        ax.scatter(sampling_time, x_samples[i,:], label=states[i]+" Sampled")
    ax.plot(sampling_time[N-1:], x_hat[i, N-1:], label=states[i]+" Estimate")
    ax.grid()
    ax.legend()
f1.suptitle(f"beta = {beta}, p={p_p}, delay={delay}")
f1.tight_layout()

f2 = plt.figure("Output Derivatives", figsize=(12,8))
for derivs in range(nderivs):
    ax2 = f2.add_subplot(1, nderivs, derivs+1)
    ax2.plot(integration_time, y_d[derivs,:], label="Actual")
    ax2.scatter(sampling_time, y_samples[derivs,:], label="Value at Sample Time")
    ax2.plot(sampling_time[N-1:], y_hat[derivs, N-1:], label="Estimated")
    ax2.grid()
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel(f"$y^{derivs}(t)$")
    ax2.legend()
f2.suptitle(f"beta = {beta}, p={p_p}, delay={delay}")
f2.tight_layout()

plt.show()