import numpy as np

def gauss_func(x, s=1, deriv=0):
    '''
    Evaluate the k^(th) derivative of a Gaussian function with std dev 's' at a value x
    
    Inputs:
        x - value at which to evaluate
        s - standard deviation (defaults to 1)
        deriv - derivative number (defaults to 0)
    '''
    if deriv==0:
        return np.exp(-(x)**2/(2*s))
    if deriv==1:
        return -x*np.exp(-(x)**2/(2*s))/s
    if deriv==2:
        return np.exp(-(x)**2/(2*s))*(x*x-s)/(s*s)
    if deriv==3:
        return -np.exp(-(x)**2/(2*s))*(x*x*x-3*s*x)/(s*s*s)
    if deriv==4:
        return (np.power(x,4) - 6*s*np.power(x,2)+3*s*s)*gauss_func(x,s,0)/np.power(s,4)
    if deriv==5:
        return -(np.power(x,5) - 10*s*np.power(x,3) + 15*s*s*x)*gauss_func(x,s,0)/np.power(s,5)
    if deriv==6:
        return (np.power(x,6) - 15*s*np.power(x,4) + 45*s*s*x*x-15*np.power(s,3))*gauss_func(x,s,0)/np.power(s,6)
    return 0


class GaussEstimator:
    '''
    Gaussian fit to a moving window
    '''
    def __init__(self, n, N, dt, sigma=1):
        '''
        n - Number of Gaussians to fit (n<=N)
        N - Number of samples in window
        '''
        self.n = n
        self.N = N
        self.dt = dt
        self.theta = np.zeros(self.n)
        self.t0 = 0.0
        self.sigma =  dt*N*sigma
        self.residuals = np.empty((self.N,))
        self.ti = np.linspace(self.t0, self.t0+self.n*self.dt, self.n, endpoint=False).reshape(-1,1) # centers for the gaussians

    def fit(self, y, t0=0.0, reg=0.0):
        '''
        Fit n Gaussian Functions to N samples y
        '''
        self.t0 = t0
        self.ti = np.linspace(self.t0, self.t0+self.n*self.dt, self.n, endpoint=False).reshape(-1,1)
        t = np.linspace(t0, t0+self.N*self.dt, self.N, endpoint=False)
        A = gauss_func((t-self.ti).T, s=self.sigma)
        # self.theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.theta = np.linalg.inv((A.T @ A) + reg*np.eye(self.n)) @ A.T @ y
        self.residuals = y - A @ self.theta
        return self.theta
    
    def estimate(self, t):
        '''
        input: t - a time to evaluate the current gaussian fit
        output: the current gaussian fit
        '''
        return self.theta @ gauss_func(t-self.ti, s=self.sigma)
    
    def differentiate(self, t, q):
        '''
        q-th derivative of gaussian fit
        inputs:
            t - time to evaluate
            q - derivative number
        '''
        return self.theta @ gauss_func(t-self.ti, s=self.sigma, deriv=q)
