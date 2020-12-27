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
    stop_cond = float('inf')
    while n < M and ((n % r != 0) or (stop_cond >= tol)):
        # If a new cycle
        if n % r == 0:
            # reset stopping conditions
            stop_cond = 0

        # b.1) project (x - increment)
        prev_x = x
        x = P[n % r](prev_x - I[n % r,:])

        # b.2) update increment
        prev_I = I[n % r,:]
        I[n % r,:] = x - (prev_x - prev_I)
        
        # Old stop condition
        mov = np.linalg.norm(x - prev_x)

        # Stopping condition for this cycle
        stop_cond += np.linalg.norm(prev_I - I[n % r,:]) + 2*(prev_I.T @ (x - prev_x))

        print(stop_cond)

        n += 1
    return x
