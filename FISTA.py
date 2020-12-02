import numpy as np

"""
Modified FISTA. Replaces prox operator with a projection onto the desired set

Input: (h, gh, p, x0, L, max_iter)
- (function handle) h = f + g, where g: E -> (-infty,infty] is proper closed and convex,
  and f: E -> R is Lf-smooth and convex
- (function handle) gh = grad(h) is convex and Hessian(h) is symmetric PSD
- (function handle) p is a projection function
- L = Lf = np.linalg.norm(Hessian(h))
- max_itr is the maxiumum number of FISTA steps to take

Output:
- Iterate x at k=max_iter
"""
def FISTA(h,gh,p,x0,L,num_iter):
    # Init
    y = x0; t = 1

   # Steps 
   for k in range(max_iter):
       # a) compute next x by projection
       w = y - (1/L)*gf(y)
       prev_x = x
       x = p(w)

       # b) compute next t
       prev_t = t
       t = (1+np.sqrt(1+4*(t**2)))/2

       # c) compute next y
       y = x + ((prev_t - 1)/(t))*(x - prev_x)

    return x
