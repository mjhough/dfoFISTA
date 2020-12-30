import numpy as np

"""
Projected gradient descent FISTA. Replaces prox operator with a projection
onto the desired set
f is convex, differentiable, and dom(g) = R^n, and h is
convex but not necessarily differentiable

Input: (grad_f, p, L, x0, max_iter)
- (function handle) grad_f = grad(f), where f is the objective function
- (function handle) p is a projection function that returns a vector in R^n
- L = Lf
- x0 is the initial iterate in R^n
- max_itr is the maxiumum number of FISTA steps to take

Output:
- Iterate x at k=max_iter
"""
def FISTA(f,grad_f,p,L,x0,max_iter=100,tol=1e-6,callback=None):
    # Init
    y = x = x0; t = 1
    stat = float('inf')
    
    k = 0
    while k < max_iter and stat >= tol:
       # a) take GD step and project
       w = y - (1/L)*grad_f(y)
       prev_x = x
       x = p(w)

       # b) compute next t
       prev_t = t
       t = (1+np.sqrt(1+4*(t**2)))/2

       # c) compute next y
       y = x + ((prev_t - 1)/(t))*(x - prev_x)

       # d) compute stationarity
       stat = abs(f(prev_x) - f(x))

       if callback is not None:
           callback(grad_f,k,x,p)

       k += 1

    return x
