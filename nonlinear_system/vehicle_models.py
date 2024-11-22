import numpy as np


class UnicycleModel:
    '''
    Unicycle ODE of the form:
    
    model inputs - u, w
    states - x, y, theta
    measured outputs - x, y

    \dot{x} = u*cos(theta)
    \dot{y} = u*sin(theta)
    \dot{theta} = w
    '''

    def __init__(self):
        pass

    def atan(self, y, x, y_b, x_b):
        theta_hat = np.arctan2(y, x)
        theta_low = np.zeros_like(theta_hat)
        theta_high = np.zeros_like(theta_hat)

        x_l = x - x_b
        x_h = x + x_b
        y_l = y - y_b
        y_h = y + y_b

        theta_low[(x_h>0) & (y_l>0)] = np.arctan2(y_l, x_h)[(x_h>0) & (y_l>0)]
        theta_low[(x_h<0) & (y_h>0)] = np.arctan2(y_h, x_h)[(x_h<0) & (y_h>0)]
        theta_low[(x_l<0) & (y_h<0)] = np.arctan2(y_h, x_l)[(x_l<0) & (y_h<0)]
        theta_low[(x_l>0) & (y_l<0)] = np.arctan2(y_l, x_l)[(x_l>0) & (y_l<0)]

        theta_high[(x_l>0) & (y_h>0)] = np.arctan2(y_h, x_l)[(x_l>0) & (y_h>0)]
        theta_high[(x_l<0) & (y_l>0)] = np.arctan2(y_l, x_l)[(x_l<0) & (y_l>0)]
        theta_high[(x_h<0) & (y_l<0)] = np.arctan2(y_l, x_h)[(x_h<0) & (y_l<0)]
        theta_high[(x_h>0) & (y_h<0)] = np.arctan2(y_h, x_h)[(x_h>0) & (y_h<0)]
        
        theta_low[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = -10*np.pi
        theta_high[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = 10*np.pi
        theta_high[theta_high<theta_hat] += 2*np.pi
        theta_low[theta_low>theta_hat] -= 2*np.pi

        return theta_hat, theta_low, theta_high


    def ode(self, x, v, w):
        '''
        RHS of ode
            x - state
            v - forward velocity
            w - angular velocity
        '''
        dxdt = np.zeros(3)
        dxdt[0] = v*np.cos(x[2])
        dxdt[1] = v*np.sin(x[2])
        dxdt[2] = w
        return dxdt
    
    def invert_outputs(self, x_hat, y_hat, x_bound, y_bound):
        '''
        Return estimated states and state bounds from estimated outputs and output bounds
        for N time steps
        
        Inputs:
            x_hat - (2,N) - x output and it's first derivative
            y_hat - (2,N) - y output and it's first derivative
            x_bound - (2,N) - bound for x output and it's first derivative
            y_bound - (2,N) - bound for y output and it's first derivative

        Outputs:
            state_hat - (3,N) - estimated states
            state_low - (3,N) - lower bound for state estimate
            state_low - (3,N) - upper bound for state estimate
        '''

        theta_hat, theta_low, theta_high = self.atan(y_hat[1], x_hat[1], y_bound[1], x_bound[1])
        state_hat = np.array([x_hat[0], y_hat[0], theta_hat])
        state_low = np.array([x_hat[0] - x_bound[0], y_hat[0] - y_bound[0], theta_low])
        state_high = np.array([x_hat[0] + x_bound[0], y_hat[0] + y_bound[0], theta_high])

        return state_hat, state_low, state_high

