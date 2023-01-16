import numpy as np
from numba import njit
kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}


@njit(**kwd)
def der(func,x,args=[], eps=1e-7):
    '''
    Forward estimate of derivative
    '''
    return ((func(x+eps,*args)-func(x,*args))/eps)

@njit(**kwd)
def newton(func,x0,tol=1e-7,nstep_max = 100, args = [], verbose=False):
    '''
    Newton method for zero finding.
    Uses forward estimate of derivative
    '''
    rel_error = np.infty
    xold = x0
    nstep = 0
    while ((rel_error > tol) and (nstep<nstep_max)):
        x = xold - func(xold,*args)/der(func,xold,args=args)
        nstep += 1
        if verbose==True:
            print ("x at iteration",nstep, 'is ',x)
        rel_error = np.abs((x-xold)/xold)
        xold = x
#    if (nstep == nstep_max):
#        print ("Convergence not achieved in %d iterations"%nstep_max)
    return(x)

def sqr(x):
    return (x**2-1)

def main():
    newton(sqr,4.0,verbose=True)

if __name__ == '__main__':
    main()

