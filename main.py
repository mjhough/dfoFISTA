import numpy as np
from FISTA import FISTA

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
    return np.minimum([np.maximum([x,l]), u])

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
Random l1-regularized least squares function in R^n
(1/2)*||Ax-b||_2^2 + lmbda*||x||_1
Input:
- n: size of n x n matrix A
- b: solution
- lmbda: lambda constant in front of regularizer
'''
def gen_func(A,b,lmbda):
    def f(x):
        return (np.linalg.norm(A@x - b)**2)/2 + lmbda*np.linalg.norm(x, ord=1)
    def grad(x):
        return A.T @ (A@x - b)
    def H(x):
        return A.T @ A
    return f,grad,H


def callback_func(k,x):
    print(f(np.array(x)))

n=2
A = np.random.rand(n,n)
x = np.random.rand(2) # soln
b = A@x
lmbda = 0.01

# generate function
f,grad,H = gen_func(A,b,lmbda)

# generate projection function
p = lambda x: p_ball(x,np.array([1,1]),0.2)

start = np.array([3,-2])
# print(FISTA(grad,H,p_uc,start,20,callback_func))
print(FISTA(grad,H,p,start,2000))
print(x)


# f,grad,H = gen_func(3,np.zeros(3))
# x = np.ones(3)
# print(f(x), grad(x), H(x), H(x).T)
