import numpy as np

"""
Projected gradient descent FISTA. Replaces prox operator with a projection
onto the desired set
f = g + h, g is convex, differentiable, and dom(g) = R^n, and h is
convex but not necessarily differentiable

Input: (grad_g, h, p, x0, L, max_iter)
- (function handle) grad_g = grad(g), where g is given by f = g + h
- (function handle) p is a projection function that returns a vector in R^n
- x0 is the initial iterate in R^n
- L = Lf
- max_itr is the maxiumum number of FISTA steps to take

Output:
- Iterate x at k=max_iter
"""
def FISTA(grad_g,p,x0,L,num_iter):
    # Init
    y = x = x0; t = 1

   for k in range(max_iter):
       # a) take GD step and project
       w = y - (1/L)*grad_g(y)
       prev_x = x
       x = p(w)

       # b) compute next t
       prev_t = t
       t = (1+np.sqrt(1+4*(t**2)))/2

       # c) compute next y
       y = x + ((prev_t - 1)/(t))*(x - prev_x)

    return x
