import numpy as np

def gauss_func(x, sigma=1, deriv=0):
    if deriv==1:
        return -x*np.exp(-(x)**2/(2*sigma))/sigma
    if deriv==2:
        return np.exp(-(x)**2/(2*sigma))*(x*x-sigma)/(sigma*sigma)
    return np.exp(-(x)**2/(2*sigma))


class GaussEstimator:

    def __init__(self, d, N, dt):
        self.d = d
        self.N = N
        self.dt = dt
        self.theta = np.zeros(N)
        self.t0 = 0.0
        self.sigma =  dt*N/2
        self.residuals = np.empty((self.N,))
        self.di = np.linspace(self.t0, self.t0+self.d*self.dt, self.d, endpoint=False).reshape(-1,1)

    def fit(self, y, t0=0.0):
        self.t0 = t0
        self.di = np.linspace(self.t0, self.t0+self.d*self.dt, self.d, endpoint=False).reshape(-1,1)
        t = np.linspace(t0, t0+self.N*self.dt, self.N, endpoint=False)
        A = gauss_func((t-self.di).T, sigma=self.sigma)
        self.theta, self.residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return self.theta
    
    def estimate(self, t):
        return self.theta @ gauss_func(t-self.di, sigma=self.sigma)
    
    def differentiate(self, t, q):
        return self.theta @ gauss_func(t-self.di, sigma=self.sigma, deriv=q)
