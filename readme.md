#### Modified FISTA with projection

This repo implements the `FISTA(h,gh,p,x0,L,max_iter)` function, where `h,gh,p` are function handles, and
- `h = f + g` where g is a function from a Euclidean space to `(-infty,infty]` that is proper closed and convex, and `f` is a function from the same Euclidean space to the reals s.t. it is `Lf`-smooth and convex.
- `gh = grad(h)` is convex and the Hessian of h is symmetric positive semi-definite.
- `p` is a function that projects its input onto the desired set.
- `L = Lf`.
- `max_itr` is the maximum number of iterations allowed.

In `main.py` FISTA is run on quadratic functions on R^n with symmetric PSD Hessian and varying projection functions.
