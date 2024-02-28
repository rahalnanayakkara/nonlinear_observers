import numpy as np

def gauss_func(x, sigma=1, deriv=0):
    if deriv==0:
        return np.exp(-(x)**2/(2*sigma))
    if deriv==1:
        return -x*np.exp(-(x)**2/(2*sigma))/sigma
    if deriv==2:
        return np.exp(-(x)**2/(2*sigma))*(x*x-sigma)/(sigma*sigma)
    if deriv==3:
        return -np.exp(-(x)**2/(2*sigma))*(x*x*x-3*sigma*x)/(sigma*sigma*sigma)
    return np.exp(-(x)**2/(2*sigma))


class GaussEstimator:
    '''
    Gaussian fit to a moving window
    '''
    def __init__(self, n, N, dt):
        '''
        n - Number of Gaussians to fit (n<=N)
        N - Number of samples in window
        '''
        self.n = n
        self.N = N
        self.dt = dt
        self.theta = np.zeros(N)
        self.t0 = 0.0
        self.sigma =  dt*N/2
        self.residuals = np.empty((self.N,))
        self.ti = np.linspace(self.t0, self.t0+self.n*self.dt, self.n, endpoint=False).reshape(-1,1) # centers for the gaussians

    def fit(self, y, t0=0.0):
        '''
        Fit n Gaussian Functions to N samples y
        '''
        self.t0 = t0
        self.ti = np.linspace(self.t0, self.t0+self.n*self.dt, self.n, endpoint=False).reshape(-1,1)
        t = np.linspace(t0, t0+self.N*self.dt, self.N, endpoint=False)
        A = gauss_func((t-self.ti).T, sigma=self.sigma)
        self.theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.residuals = y - A @ self.theta
        return self.theta
    
    def estimate(self, t):
        return self.theta @ gauss_func(t-self.ti, sigma=self.sigma)
    
    def differentiate(self, t, q):
        return self.theta @ gauss_func(t-self.ti, sigma=self.sigma, deriv=q)
