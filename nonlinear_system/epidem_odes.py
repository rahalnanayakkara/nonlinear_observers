import numpy as np
from nonlinear_system.sample_odes import ControlAffineODE

NDERIVS = 3

class UIV(ControlAffineODE):

    def __init__(self,  beta=4.71, delta=1.07, p=3.07, c=2.3):
        self.beta = beta
        self.delta = delta
        self.p_p = p
        self.c = c
        self.nderivs = NDERIVS
        super().__init__(state_dim=3, input_dim=1, output_dim=3, f=self.uiv_f, h=self.output_derivative)
    
    def uiv_f(self, t: float, x: np.ndarray):
        '''
        RHS for UIV system:

        d_x0 = - beta*x0*x2
        d_x1 = beta*x0*x2 - delta*x1
        d_x2 = p*x1 - c*x2

        '''
        rhs = np.array([
            -self.beta*x[0]*x[2],
            self.beta*x[0]*x[2] - self.delta*x[1],
            self.p_p*x[1] - self.c*x[2]
        ])
        return rhs
    
    def output_fn(self, t: float, x: np.ndarray, u: np.ndarray):
        return x[2]
    
    def output_derivative(self, t: float, x: np.ndarray, u: np.ndarray):
        '''
        Computes the output of the system and it's derivatives
        '''
        y_d = np.empty((NDERIVS,))
        y_d[0] = x[2]
        y_d[1] = self.rhs(t, x, u)[2]
        y_d[2] = self.p_p*self.beta*x[0]*x[2] - (self.c+self.delta)*y_d[1] - self.delta*self.c*x[2]
        return y_d
    
    def invert_output(self, t: float, y_d: np.ndarray, u: np.ndarray = None):
        '''
        Function that maps the output and it's derivatives to the system states
        '''
        xhat = np.array([
            (y_d[2]+(self.delta+self.c)*y_d[1]+self.delta*self.c*y_d[0])/(self.p_p*self.beta*y_d[0]),
            (y_d[1]+self.c*y_d[0])/self.p_p,
            y_d[0]
        ])
        return xhat
    
    def invert_output2(self, t: float, y_d: np.ndarray, u: np.ndarray = None):
        '''
        Function that maps the output and it's derivatives to the states [V, I, UV]
        '''
        xhat = np.array([
            y_d[0],
            (y_d[1]+self.c*y_d[0])/self.p_p,
            (y_d[2]+(self.delta+self.c)*y_d[1]+self.delta*self.c*y_d[0])/(self.p_p*self.beta)
        ])
        return xhat