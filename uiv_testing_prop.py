from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.epidem_odes import UIV
from moving_polyfit.moving_ls import PolyEstimator
import numpy as np
import matplotlib.pyplot as plt
import copy

'''
Simulate infection spread in a single host using UIV model.
Obtain an estimate of the internal states by using a polynomial observer on V with a delay.
Propagate the estimate forward using the differential equation to obtain current states

delay - selects the delay to evaluate the polynomial observer
infection_step - step in which to initiate viral load. 
                Non zero allows observer to run before infection begins.
'''

verbose = True

sampling_dt = 1
integration_per_sample = 100
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 30
num_integration_steps = num_sampling_steps*integration_per_sample

uiv_ode = UIV(beta=1, p=2)
beta = uiv_ode.beta
p_p = uiv_ode.p_p
n = uiv_ode.n
m = uiv_ode.m
p = uiv_ode.p

U0 = 4
V0 = 50e-8
x0 = [U0, 0, 0]

nderivs = uiv_ode.nderivs

d = 4
N = 6

delay = 1
infection_step = 0  # default is 0
estimator = PolyEstimator(d, N, sampling_dt)

x = np.zeros((n, num_integration_steps))
y_d = np.zeros((nderivs, num_integration_steps))
x_hat = np.zeros((n, num_integration_steps))
y_hat = np.zeros((nderivs, num_integration_steps))

y_samples = np.zeros((nderivs, num_sampling_steps))
y_hat_samples = np.zeros((nderivs, num_sampling_steps))
x_samples = np.zeros((n, num_sampling_steps))
x_hat_samples = np.zeros((n, num_sampling_steps))
x_hat_prop = np.zeros((n, num_sampling_steps))

theta = np.empty((d+1, num_sampling_steps)) # coefficients of fitted polynomial

integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))

x[:, 0] = x0
x_samples[:, 0] = x0

sys = ContinuousTimeSystem(uiv_ode, x0=x0, dt=integration_dt)

y_d[:, 0] = sys.y
y_samples[:, 0] = sys.y

sys_copy = copy.deepcopy(sys)

print(f"delay = {delay}")
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
        # fit polynomial
        theta[:, t] = estimator.fit(y_samples[0, t-N+1:t+1])    # store polynomial coefficients

        # estimate with polynomial derivatives with delay
        for j in range(nderivs):
            y_hat_samples[j, t-delay] = estimator.differentiate((N-delay-1)*sampling_dt, j)

        for i in range(integration_per_sample):
            idx = (t-delay)*integration_per_sample + i
            for j in range(nderivs):
                y_hat[j, idx] = estimator.differentiate((N-delay-1-1)*sampling_dt+i*integration_dt, j)
            x_hat[:, idx] = uiv_ode.invert_output(t=t-delay, y_d=y_hat[:, idx])

        x_hat_samples[:, t-delay] = uiv_ode.invert_output(t=t-delay, y_d=y_hat_samples[:, t-delay])
        
        if verbose:
            print(f"On day {t} we estimate day {t-delay}")

        sys_copy.reset(x0=x_hat_samples[:, t-delay], t=sampling_time[t-delay])

        for i in range(integration_per_sample*delay):
            sys_copy.step(0)
        x_hat_prop[:, t] = sys_copy.x

    else:
        theta[:, t] = 0.0
        y_hat_samples[:, t] = 0.0
        # x_hat[0, t] = U0

states = ["U", "I", "V"]

f1 = plt.figure("State Evolution", figsize=(12,8))
for i in range(n):
    ax = f1.add_subplot(1,n,i+1)
    ax.plot(integration_time, x[i,:], label=states[i])
    ax.scatter(sampling_time, x_samples[i,:], label=states[i]+" at Sample")
    ax.plot(integration_time, x_hat[i,:], label=states[i]+" Estimate")
    ax.scatter(sampling_time[N-1:], x_hat_samples[i, N-1:], label=states[i]+" Sampled Estimate")
    ax.scatter(sampling_time[N-1:], x_hat_prop[i, N-1:], label=states[i]+" Estimate Prop", c='red', marker='x')
    ax.grid()
    ax.legend()
f1.suptitle(f"beta = {beta}, p={p_p}, delay={delay}")
f1.tight_layout()

f2 = plt.figure("Output Derivatives", figsize=(12,8))
for derivs in range(nderivs):
    ax2 = f2.add_subplot(1, nderivs, derivs+1)
    ax2.plot(integration_time, y_d[derivs,:], label="Actual")
    ax2.scatter(sampling_time, y_samples[derivs,:], label="Value at Sample Time")
    ax2.plot(integration_time, y_hat[derivs,:], label="Estimate")
    ax2.scatter(sampling_time[N-1:], y_hat_samples[derivs, N-1:], label="Estimated Sampled")
    ax2.grid()
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel(f"$y^{derivs}(t)$")
    ax2.legend()
f2.suptitle(f"beta = {beta}, p={p_p}, delay={delay}")
f2.tight_layout()

plt.show()