import numpy as np
import matplotlib.pyplot as plt

from FISTA import FISTA
from Dykstra import Dykstra

from PGD import PGD

import pdb

# Unconstrained projection
def p_uc(x):
    return x

'''
Box constrained projection
Input:
- x = y - (1/L)*grad_g(y) in R^n
- l: lower bound in R^n s.t. x >= l (element-wise)
- u: upper bound in R^n s.t. x <= u (element-wise)
- we require that l <= u
Output: x projected onto the box constraint
'''
def p_box(x,l,u):
    return np.minimum(np.maximum(x,l), u)

'''
Ball constrained projection
Input:
- x = y - (1/L)*grad_g(y) in R^n
- c: center of the ball in R^n
- r: radius of the ball s.t. r > 0
Output: x projected onto the ball constraint
'''
def p_ball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

'''
Linear least squares function in R^n
(1/2)*||Ax-b||_2^2
Input:
- n: size of n x n matrix A
- b: solution
'''
def gen_func(A,b):
    def f(x):
        return (np.linalg.norm(A@x - b)**2)/2
    def grad(x):
        return A.T @ (A@x - b)
    def H():
        return A.T @ A
    return f,grad,H







if __name__ == "__main__":
    n=2
    A = np.random.uniform(low=-5, high=5, size=(n,n))
    x = np.random.uniform(low=-5, high=5, size=(n,)) # soln
    b = A@x
    delta = 1

    # Num iterations and starting point
    num_iter = 1000
    start = np.array([-5,-4])

    # Change constraint here
    constraint = 'box'

    count = 0
    iters = np.empty((num_iter+1,2))
    iters[0,:] = start
    status = []
    ma_status = []
    grads = []
    # Callback function
    def callback_func(grad_f, curr_iter,xk,p,stat,ma_stat,grad_Fd):
        global count
        count += 1
        iters[curr_iter+1,:] = xk

        # Call FISTA with F = grad_f^T*d to minimize d
        def F(d):
            return grad_f(xk).T @ d
        def grad_F(d):
            return grad_f(xk)

        #  d_delta_ball = lambda d: p_ball(xk+d,xk,delta)
        #  d_delta_ball = lambda dk: dk - dk*(np.absolute((dk.T @ dk) - delta)/(dk.T @ dk))
        #  p2 = lambda dk: p(xk+dk)-xk
        #  proj = lambda dk: Dykstra([p2,d_delta_ball],dk)

        #  eLl = 1 # np.linalg.norm(np.identity(xk.shape[0]))
        #  d0 = np.ones(xk.shape[0])
        #  d = FISTA(F,grad_F,proj,eLl,d0,max_iter=20,callback=None)
        #  grads2.append(abs(F(d)))
        #  grads.append(np.linalg.norm(grad_f(xk)))
        
        status.append(stat)
        ma_status.append(ma_stat)
        #  grads.append(grad_Fd)

    # generate function
    f,grad,H = gen_func(A,b)
    L = np.linalg.norm(H())

    # generate projection functions
    ball_c = [1,1]
    ball_r = 2
    pball = lambda x: p_ball(x,np.array(ball_c),ball_r)

    box_l = np.array([-1,-1])
    box_u = np.array([1,1])
    pbox = lambda x: p_box(x,box_l,box_u)

    #  x_pred = PGD(f,grad,[pball],L,start,num_iter)
    #  print(x)
    #  print(x_pred)

    # Plotting the function contour and iterates with no constraint
    if constraint == None:
        #  x_pred = FISTA(grad,p_uc,L,start,num_iter,callback_func)
        x_pred = PGD(f,grad,[p_uc],L,start,num_iter,callback_func)
        print(x)
        print(x_pred)
    
        if n == 2:
            fig,ax = plt.subplots()
            xx = yy = np.arange(-6,6,0.5)
            X,Y = np.meshgrid(xx,yy,sparse=True)
            def f2(x1,x2):
                return f(np.array([x1,x2]))
            f2_vec = np.vectorize(f2)
            zz = f2_vec(X,Y)
            ax.contourf(xx,yy,zz)
            ax.plot(iters[:count,0],iters[:count,1],marker='x',color='r')

    # Plotting the function contour and iterates with ball constraint
    if constraint == 'ball':
        #  x_pred = FISTA(grad,pball,L,start,num_iter,callback_func)
        x_pred = PGD(f,grad,[pball],L,start,num_iter,callback_func)
        print(x)
        print(x_pred)

        if n == 2:
            fig,ax = plt.subplots()
            xx = yy = np.arange(-6,6,0.5)
            X,Y = np.meshgrid(xx,yy,sparse=True)
            def f2(x1,x2):
                return f(np.array([x1,x2]))
            f2_vec = np.vectorize(f2)
            zz = f2_vec(X,Y)
            ax.contourf(xx,yy,zz)
            ax.plot(iters[:count,0],iters[:count,1],marker='x',color='r')
            cir = plt.Circle((1,1), ball_r, fill=False, color='w')
            ax.add_patch(cir)

    # Plotting the function contour and iterates with box constraint
    if constraint == 'box':
        #  x_pred = FISTA(grad,pbox,L,start,num_iter,callback_func)
        x_pred = PGD(f,grad,[pbox],L,start,num_iter,callback_func)
        print(x)
        print(x_pred)

        if n == 2:
            fig,ax = plt.subplots()
            xx = yy = np.arange(-6,6,0.5)
            X,Y = np.meshgrid(xx,yy,sparse=True)
            def f2(x1,x2):
                return f(np.array([x1,x2]))
            f2_vec = np.vectorize(f2)
            zz = f2_vec(X,Y)
            ax.contourf(xx,yy,zz)
            ax.plot(iters[:count,0],iters[:count,1],marker='x',color='r')
            rec = plt.Rectangle((box_l[0], box_l[1]), box_u[0] - box_l[0],
                    box_u[1] - box_l[1], fill=False, color='w')
            ax.add_patch(rec)

    plt.plot(x[0], x[1], color='w', marker='.')

    # Plot residuals
    plt.figure()
    plt.semilogy(range(0,len(status)),status, 'b-', label='diff')
    plt.semilogy(range(0,len(ma_status)),ma_status, 'r-.', label='ma')
    #  plt.semilogy(range(0,len(grads)),grads, 'k--', label='grad')
    plt.legend()
    plt.show()
