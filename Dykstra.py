import numpy as np
import matplotlib.pyplot as plt

'''
Dykstra's algorithm for the projection onto the
intersection of closed convex sets, K [1].

Input:
- P: An array of projection function handles onto
 closed convex sets, where P[i] projections onto
 a subset of the Hilbert space X for all i.
- x0: A point in the Hilbert space X.
- max_iter: Maximum number of outer iterations before
stopping.
- tol: The tolerance for the stopping condition c_I in paper [2].

Output:
- x: The point in the intersection of sets, K
s.t. the distance between x and x0 is minimized.

References:
[1] R.L. Dykstra, 1983. An algorithm for restricted least squares regression.
[2] E.G. Birgin & M. Raydan, 2004. Robust stopping criteria for Dykstra's algorithm.
'''
def Dykstra(P,x0,max_iter=1000,tol=1e-6):
    x = x0
    p = len(P)
    y = np.zeros((p,x0.shape[0]))

    n = 0
    cI = float('inf')
    while n < max_iter and cI >= tol:
        cI = 0
        for i in range(0,p):
            # Update iterate
            prev_x = x.copy()
            x = P[i](prev_x - y[i,:])

            # Update increment
            prev_y = y[i,:].copy()
            y[i,:] = x - (prev_x - prev_y)

            # Stop condition
            cI += np.linalg.norm(prev_y - y[i,:])**2

            n += 1
    return x
