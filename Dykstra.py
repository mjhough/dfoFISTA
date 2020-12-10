import numpy as np
import matplotlib.pyplot as plt

'''
Dykstra's algorithm for the projection onto the
intersection of closed convex sets, K.

Input:
- P: An array of projection function handles onto
 closed convex sets, where P[i] projections onto
 a subset of the Hilbert space X for all i.
- x0: A point in the Hilbert space X.

Output:
- x: The point in the intersection of sets, K
s.t. the distance between x and x0 is minimized.
'''
def Dykstra(P,x0,max_iter=10):
    # a) initialize starting params
    x = x0
    r = len(P)
    M = max_iter*r # since r cycles per iteration
    I = np.zeros((len(P),x0.shape[0]))
    tol = 1e-6
    mov = float('inf')

    # b) iteratively compute the projection
    # until reach tolerance level
    n = 0
    while mov > tol and n < M:
        # b.1) project x - increment
        prev_x = x
        x = P[n % r](prev_x - I[n % r,:])

        # b.2) update increment
        I[n % r,:] = x - (prev_x - I[n % r,:])
        
        mov = np.linalg.norm(x - prev_x)
        n += 1
    return x
