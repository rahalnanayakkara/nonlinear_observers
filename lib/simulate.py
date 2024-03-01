import numpy as np
from nonlinear_system.ct_system import ContinuousTimeSystem

def simulate(ode, x0, integration_dt, sampling_dt, sim_time):

    num_steps = int(sim_time/integration_dt)
    num_steps_per_sample = int(sampling_dt/integration_dt)
    x = np.zeros((ode.n, num_steps+1))
    time = np.zeros(num_steps+1)
    y_d = np.zeros((ode.nderivs, num_steps+1))
    
    x[:,0] = x0
    sys = ContinuousTimeSystem(ode, x0=x0, dt=integration_dt)
    y_d[:,0] = sys.y

    for i in range(1, num_steps+1):
        sys.step(0)
        x[:,i] = sys.x
        y_d[:,i] = sys.y
        time[i] = sys.t
    y_samples = y_d[0, ::num_steps_per_sample]
    t_samples = time[::num_steps_per_sample]

    return time, x, y_d, t_samples, y_samples