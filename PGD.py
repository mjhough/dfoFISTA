import numpy as np
import pandas as pd
from collections import deque

from Dykstra import Dykstra
from FISTA import FISTA

import pdb

"""
Accelerated projected gradient descent.
- f is convex and differentiable.
- The problem is to minimize f(x) subject to x in C.
- Where C is the intersection of convex sets.

Input: (grad_f, P, L, x0, max_iter)
- (function handle) f is the objective function
- (function handle) grad_f = grad(f), where f is the objective function
- P is an array of projection function handles that take/return a vector in R^n
- L = Lf
- x0 is the initial iterate in R^n
- max_itr is the maxiumum number of FISTA steps to take

Output:
- Iterate x after the max number of iterations or after converged within tolerance.
"""
def PGD(f,grad_f,P,L,x0,max_iter=100000,callback_func=None,tol=1e-13):
    # Init
    y = x = x0; t = 1
    proj = lambda w: Dykstra(P,w)
    m = 10 # EMA history size

    # Accelerated projected GD
    k = 0
    stat = float('inf')
    ma_stat = float('inf')
    grad_Fd = float('inf')
    f_hist = deque(maxlen=m) # EMA history
    while k < max_iter and ma_stat >= tol:
       # a) take GD step and project
       w = y - (1/L)*grad_f(y)
       prev_x = x
       x = proj(w)

       # b) compute next t
       prev_t = t
       t = (1+np.sqrt(1+4*(t**2)))/2

       # c) compute next y
       y = x + ((prev_t - 1)/(t))*(x - prev_x)

       # d) compute stopping condition
       #  grad_Fd = compute_grad_Fd(grad_f,proj,x)
       grad_FD = None
       f_hist.append(abs(f(prev_x) - f(x)))
       ma_stat = ma(f_hist)
       stat = abs(f(prev_x) - f(x))

       print('k =',k,'=>','diff:',stat,'ma:',ma_stat,'grad:',grad_Fd)

       if callback_func is not None:
           callback_func(grad_f,k,x,proj,stat,ma_stat,grad_Fd)

       k += 1

    return x

# Moving average of last m items
def ma(f_hist):
    m = f_hist.maxlen
    l = len(f_hist)
    d = pd.Series(f_hist)
    return d.rolling(window=min([l,m])).mean().iloc[-1]

def compute_grad_Fd(grad_f,proj,xk):
    delta = 1

    # Problem is to min grad_f(xk)^T * d w.r.t d
    def F(d):
        return grad_f(xk).T @ d
    def grad_F(d):
        return grad_f(xk)

    d_delta_ball = lambda dk: dk - dk*(np.absolute((dk.T @ dk) - delta)/(dk.T @ dk))
    inner_p = lambda dk: proj(xk+dk)-xk
    p = lambda dk: Dykstra([inner_p,d_delta_ball],dk)

    L = 1
    d0 = np.ones(xk.shape[0])
    d = FISTA(F,grad_F,p,L,d0)

    return abs(F(d))
