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
def FISTA(grad_f,p,L,x0,num_iter,callback=None):
    # Init
    y = x = x0; t = 1

    for k in range(num_iter):
       # a) take GD step and project
       w = y - (1/L)*grad_f(y)
       prev_x = x
       x = p(w)

       # b) compute next t
       prev_t = t
       t = (1+np.sqrt(1+4*(t**2)))/2

       # c) compute next y
       y = x + ((prev_t - 1)/(t))*(x - prev_x)

       if callback is not None:
           callback(grad_f,k,x,p)

    return x
