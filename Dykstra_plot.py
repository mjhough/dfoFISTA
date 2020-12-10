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
iters = []
def Dykstra(P,x0,max_iter=10):
    # a) initialize starting params
    x = x0
    r = len(P)
    M = max_iter*r # since r cycles per iteration
    I = np.zeros((len(P),x0.shape[0]))
    tol = 1e-6
    mov = float('inf')

    # Plotting
    iters.append(x0)

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

        # Plotting
        iters.append(x)
        print("Mov:",mov)

    return x


def p_box(x,l,u):
    return np.minimum(np.maximum(x,l), u)

def p_ball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

ball_c = [1,1]
ball_r = 5
pball = lambda x: p_ball(x,np.array(ball_c),ball_r)

ball_c2 = [4,-5]
ball_r2 = 3
pball2 = lambda x: p_ball(x,np.array(ball_c2),ball_r2)

box_l = np.array([-5,-5])
box_u = np.array([5,5])
pbox = lambda x: p_box(x,box_l,box_u)

Dykstra([pbox,pball,pball2], np.array([0,-8]))

fig,ax = plt.subplots()
ax.axis('equal')
iters = np.array(iters)
ax.plot(iters[0,0],iters[0,1],marker='.',color='k',zorder=5)
ax.plot(iters[0:2,0],iters[0:2,1],color='r')
ax.plot(iters[1:len(iters)-1,0],iters[1:len(iters)-1,1],marker='.',color='r')
ax.plot(iters[-2:-1,0],iters[-2:-1,1],color='r')
ax.plot(iters[-1,0],iters[-1,1],marker='.',color='k')
rec = plt.Rectangle((box_l[0], box_l[1]), box_u[0] - box_l[0],
        box_u[1] - box_l[1], fill=False, color='b')
ax.add_patch(rec)
cir = plt.Circle((1,1), ball_r, fill=False, color='g')
ax.add_patch(cir)
cir2 = plt.Circle((4,-5), ball_r2, fill=False, color='m')
ax.add_patch(cir2)

plt.show()
