import sys
import os
import sympy as sy
import numpy as np
import dill as pickle
from time import time
import imc
from imc_utils import *


def initialize_functions(ce_k, h2):
    """
        Must be run for first time ce_k and h2 values.
        After ran once, created functions are serialized.
    """
    dir = './DER/grads_hessian_functions/'

    if not os.path.exists(dir): os.makedirs(dir)

    if os.path.isfile(dir + 'first_grad'):
        print('Functions have already been generated previously for a different ce_k, h2 pair. '
              'Redundant functions will be skipped.')
        print('If all functions wish to be remade, delete grads_hessian_functions directory')
        f1 = []
    else:
        print('Functions are being generated for the first time. All functions will be created.')
        dd, f_g, f_ch, f_h = first_grads_and_hessian()
        ffr = ffr_jacobian()

        f1 = [(dd, 'dd'), (f_g, 'first_grad'), (f_ch, 'constant_hess'), (f_h, 'first_hess'), (ffr, 'friction_jacobian')]

    s_g, s_h = second_grads_and_hessian(ce_k, h2)

    cekr = '_cek_' + str(ce_k) + '_h2_' + str(h2)

    f2 = [(s_g, 'second_grad' + cekr), (s_h, 'second_hess' + cekr)]

    pickle.settings['recurse'] = True
    all_functions = f1 + f2
    for func, name in all_functions:
        with open(dir + name, 'wb') as f:
            pickle.dump(func, f)


def first_grads_and_hessian(k1=50.0):
    """
        x is a (1x12) vector
        Gradients: dd1/dx | dd2/dx | dd12/dx are (3x12), there hessians are all zero
                   dD1/dx | dD2/dx | dR/dx   | dS1/dx | dS2/dx | dt2/dx are (1x12)
        Hessian :  exist only for the (1x12) gradients and are (12x12)
    """
    start = time()
    print('Starting first gradients and hessians')

    # Declare symbolic variables
    x1s = sy.symarray('x1s', 3)
    x1e = sy.symarray('x1e', 3)
    x2s = sy.symarray('x2s', 3)
    x2e = sy.symarray('x2e', 3)

    # First half of the min-distance algorithm
    d1 = x1e - x1s
    d2 = x2e - x2s
    d12 = x2s - x1s
    D1 = (d1 ** 2).sum()
    D2 = (d2 ** 2).sum()
    S1 = (d1 * d12).sum()
    S2 = (d2 * d12).sum()
    R = (d1 * d2).sum()
    den = D1 * D2 - R ** 2
    t1 = sy.Piecewise((((S1 * D2 - S2 * R) / den), sy.Ne(den, 0.0)), (0.0, True))  # avoid division by zero
    t2 = approx_fixbound_sy(t1, k=k1)

    wrt = [*x1s, *x1e, *x2s, *x2e]

    # No need to create a function for dd1/dx | dd2/dx | dd12/dx because they are constant.
    # These are constant gradients
    dd = np.zeros((9, 12), dtype=np.float64)
    dd[:3] += np.array(sy.Matrix(d1).jacobian(wrt), dtype=np.float64)
    dd[3:6] += np.array(sy.Matrix(d2).jacobian(wrt), dtype=np.float64)
    dd[6:] += np.array(sy.Matrix(d12).jacobian(wrt), dtype=np.float64)

    # Exclude d1 d2 and d12 since their hessians are all zero
    ele = [D1, D2, R, S1, S2, t2]

    # Compute dD1/dx | dD2/dx | dR/dx | dS1/dx | dS2/dx | dt2/dx
    gv = [sy.Matrix([e]).jacobian(wrt) for e in ele]
    hv = [g.jacobian(wrt) for g in gv]

    # Compute functions to compute gradients with nodal coordinates as input
    grads = [create_function(g, wrt) for g in gv]

    # These are constant matrices
    constant_hess = np.array([h for h in hv[:-1]], dtype=np.float64).reshape((5, 12, 12))

    # Compute function gradient of dt2/dx
    hess = create_function(hv[-1], wrt)  # only t2 Hessian is non-constant

    print('Completed first contact gradient and hessian: {:.3f} seconds'.format(time() - start))

    return dd, grads, constant_hess, hess


def second_grads_and_hessian(ce_k, h2, k1=50.0, k2=50.0):
    """
        Required inputs: d1, d2, d12, D1, D2, R, S1, S2, t2
        refer to these values as secondary input vals

        Gradients: dE/dd1 | dE/dd2 | dE/dd12 are all (1x3)
                   dE/dD1 | dE/dD2 | dE/dR   | dE/S1 | dE/S2 | dE/t2 are all scalars
        Hessian : hessians wrt d1, d2, d12 not needed
                  hessians of scalar gradients are also gradients
    """
    start = time()
    print('Starting second gradients and hessians')

    # Declare symbolic variables
    d1 = sy.symarray('d1', 3)
    d2 = sy.symarray('d2', 3)
    d12 = sy.symarray('d12', 3)
    D1, D2, R, S1, S2, t2 = sy.symbols('D1, D2, R, S1, S2, t2')

    # Second half of min-distance algorithm
    u1 = (t2 * R - S2) / D2
    u2 = approx_fixbound_sy(u1, k=k1)
    t3 = (1 - boxcar_func_sy(u1, k=k2)) * approx_fixbound_sy(((u2 * R + S1) / D1), k=k1) + boxcar_func_sy(u1, k=k2) * t2
    dist1 = (d1 * t3 - d2 * u2 - d12)
    dist = sy.sqrt((dist1 ** 2).sum())
    E = (1 / ce_k) * sy.log(1 + sy.exp(ce_k * (h2 - dist)))

    # Create min-distance function with secondary input vals as input
    inputs = wrt = [*d1, *d2, *d12, D1, D2, R, S1, S2, t2]

    # Compute dE/dd1 | dE/dd2 | dE/dd12 are all (1x3)
    # dE/dD1 | dE/dD2 | dE/dR | dE/S1 | dE/S2 | dE/t2 are all scalars
    gv = [E.diff(w) for w in wrt]

    # Create functions for each gradient listed above
    grads = [create_function(g, inputs) for g in gv]

    # Create functions for each gradient of ...
    hess = [create_function(sy.Matrix([g]).jacobian(wrt), inputs) for g in gv]

    print('Completed second contact gradient and hessian: {:.3f} seconds'.format(time() - start))

    return grads, hess


def ffr_jacobian():
    """
        Obtain Jacobian of friction force.
        Returns 3x24 Jacobian. dFfr/dy where y = [velocity; Fn]
    """
    v1s = sy.symarray('v1s', 3)
    v1e = sy.symarray('v1e', 3)
    v2s = sy.symarray('v2s', 3)
    v2e = sy.symarray('v2e', 3)
    f1s = sy.symarray('f1s', 3)
    f1e = sy.symarray('f1e', 3)
    f2s = sy.symarray('f2s', 3)
    f2e = sy.symarray('f2e', 3)
    mu_k = sy.symbols('mu_k')

    fn1 = sy.sqrt(((f1s + f1e)**2).sum())
    fn2 = sy.sqrt(((f2s + f2e)**2).sum())

    fn1_u = (f1s + f1e) / fn1
    fn2_u = (f2s + f2e) / fn2

    v1 = 0.5 * (v1s + v1e)
    v2 = 0.5 * (v2s + v2e)
    v_rel1 = v1 - v2
    tv_rel1 = v_rel1 - (v_rel1.dot(fn1_u) * fn1_u)
    tv_rel1_n = sy.sqrt((tv_rel1 ** 2).sum())
    tv_rel1_u = tv_rel1 / sy.sqrt((tv_rel1 ** 2).sum())

    v_rel2 = -v_rel1
    tv_rel2 = v_rel2 - (v_rel2.dot(fn2_u) * fn2_u)
    tv_rel2_n = sy.sqrt((tv_rel2 ** 2).sum())
    tv_rel2_u = tv_rel2 / tv_rel2_n

    heaviside1 = 1 / (1 + sy.exp(-50.0 * (tv_rel1_n - 0.15)))
    heaviside2 = 1 / (1 + sy.exp(-50.0 * (tv_rel2_n - 0.15)))

    ffr_e1 = heaviside1 * mu_k * tv_rel1_u * fn1
    ffr_e2 = heaviside2 * mu_k * tv_rel2_u * fn2

    inputs = [*v1s, *v1e, *v2s, *v2e, *f1s, *f1e, *f2s, *f2e, mu_k]

    ffr = sy.Matrix([*ffr_e1, *ffr_e2])

    wrt = [*v1s, *v1e, *v2s, *v2e, *f1s, *f1e, *f2s, *f2e]
    ffr_grad = create_function(sy.Matrix(ffr).jacobian(wrt), inputs)

    return ffr_grad


def main():
    if len(sys.argv) != 3: raise ValueError("Expects two arguments, ce_k and h2")
    ce_k = float(sys.argv[1])
    h2   = float(sys.argv[2])
    initialize_functions(ce_k, h2)


if __name__  == '__main__':
    main()