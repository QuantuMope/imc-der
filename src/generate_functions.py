import os
import sys
import dill as pickle
import numpy as np
import symengine as se
from time import time


def create_function(ele, wrt):
    return se.Lambdify(wrt, ele, cse=True, backend="llvm")


def approx_fixbound_se(x, k):
    """ H(x) """
    return (1 / k) * (se.log(1 + se.exp(k * x)) - se.log(1 + se.exp(k * (x - 1.0))))


def log_func_se(x, k, c=0.5):
    return 1 / (1 + se.exp(-k*(x-c)))


def boxcar_func_se(x, k):
    """ B(x) """
    step1 = log_func_se(x, k, 0.00)
    step2 = log_func_se(x, k, 1.00)
    return step1 - step2


def generate_functions(ce_k, h2):
    """
        Must be run for first time ce_k and h2 values.
        After ran once, created functions are serialized.
    """
    dir = './DER/grads_hessian_functions/'

    if not os.path.exists(dir): os.makedirs(dir)

    de, f_g, f_ch, f_h = first_grads_and_hessian()
    s_d, s_op = second_derivatives_and_second_order_partials(ce_k, h2)
    ffr = ffr_jacobian()

    cekr = '_cek_' + str(ce_k) + '_h2_' + str(h2)

    functions = [(de, 'de'), (f_g, 'first_grad'), (f_ch, 'constant_hess'), (f_h, 'first_hess'),
                 (ffr, 'friction_jacobian'), (s_d, 'second_derivative' + cekr),
                 (s_op, 'second_order_partials' + cekr)]

    pickle.settings['recurse'] = True
    for func, name in functions:
        with open(dir + name, 'wb') as f:
            pickle.dump(func, f)


def first_grads_and_hessian(k1=50.0):
    """
        x is a (1x12) vector
        Gradients: de1/dx | de2/dx | de12/dx are (3x12), there hessians are all zero
                   dD1/dx | dD2/dx | dR/dx   | dS1/dx | dS2/dx | dt2/dx are (1x12)
        Hessian :  0 for e1, e2, and e12. Others are (12x12)
    """
    start = time()
    print('Starting first gradients and hessians...')

    # Declare symbolic variables
    x1s = se.symarray('x1s', 3)
    x1e = se.symarray('x1e', 3)
    x2s = se.symarray('x2s', 3)
    x2e = se.symarray('x2e', 3)

    # First half of the min-distance algorithm
    e1 = x1e - x1s
    e2 = x2e - x2s
    e12 = x2s - x1s
    D1 = (e1 ** 2).sum()
    D2 = (e2 ** 2).sum()
    S1 = (e1 * e12).sum()
    S2 = (e2 * e12).sum()
    R = (e1 * e2).sum()
    den = D1 * D2 - R ** 2
    t1 = se.Piecewise((((S1 * D2 - S2 * R) / den), se.Ne(den, 0.0)), (0.0, True))  # avoid division by zero
    t2 = approx_fixbound_se(t1, k=k1)

    wrt = se.Matrix([*x1s, *x1e, *x2s, *x2e])

    # No need to create a function for de1/dx | de2/dx | de12/dx because they are constant.
    # These are constant gradients
    de = np.zeros((9, 12), dtype=np.float64)
    de[:3]  = np.array(se.Matrix(e1).jacobian(wrt)).astype(np.float64)
    de[3:6] = np.array(se.Matrix(e2).jacobian(wrt)).astype(np.float64)
    de[6:]  = np.array(se.Matrix(e12).jacobian(wrt)).astype(np.float64)

    # Exclude e1 e2 and e12 since their hessians are all zero
    ele = se.Matrix([D1, D2, R, S1, S2, t2])

    # Compute dD1/dx | dD2/dx | dR/dx | dS1/dx | dS2/dx | dt2/dx
    gv = [se.Matrix([e]).jacobian(wrt) for e in ele]

    # Compute d^D1/dx^2 | d^2D2/dx^2 | d^2R/dx^2 | d^2S1/dx^2 | d^S2/dx^2 | d^2t2/dx^2
    hv = [g.T.jacobian(wrt) for g in gv]

    # Create functions for the gradients
    grads = [create_function(g, wrt) for g in gv]

    # These are constant hessian matrices
    constant_hess = np.array([h for h in hv[:-1]]).reshape((5, 12, 12)).astype(np.float64)

    # Create hessian function for d^2t2/dx^2
    hess = create_function(hv[-1], wrt)  # only t2 Hessian is non-constant

    print('Completed first contact gradient and hessian: {:.3f} seconds'.format(time() - start))

    return de, grads, constant_hess, hess


def second_derivatives_and_second_order_partials(ce_k, h2, k1=50.0, k2=50.0):
    """
        Required inputs: e1, e2, e12, D1, D2, R, S1, S2, t2

        Derivatives: dE/de1 | dE/de2 | dE/de12 are all (1x3)
                     dE/dD1 | dE/dD2 | dE/dR   | dE/S1 | dE/S2 | dE/t2 are all scalars
        Second order partials: To compute d/dx(dE/dD1) via chain rule,
                               we need d^2E/dD1v where v is each of the required inputs
    """
    start = time()
    print('Starting second derivatives and second order partials...')

    # Declare symbolic variables
    e1 = se.symarray('e1', 3)
    e2 = se.symarray('e2', 3)
    e12 = se.symarray('e12', 3)
    D1, D2, R, S1, S2, t2 = se.symbols('D1, D2, R, S1, S2, t2')

    # Second half of min-distance algorithm
    u1 = (t2 * R - S2) / D2
    u2 = approx_fixbound_se(u1, k=k1)
    conditional = boxcar_func_se(u1, k=k2)
    t3 = (1 - conditional) * (u2 * R + S1) / D1 + conditional * t2
    dist = se.sqrt(((e1 * t3 - e2 * u2 - e12)**2).sum())
    E = (1 / ce_k) * se.log(1 + se.exp(ce_k * (h2 - dist)))

    inputs = wrt = se.Matrix([*e1, *e2, *e12, D1, D2, R, S1, S2, t2])

    # Compute dE/dd1 | dE/dd2 | dE/dd12 are all (1x3)
    # dE/dD1 | dE/dD2 | dE/dR | dE/S1 | dE/S2 | dE/t2 are all scalars
    dv = [E.diff(w) for w in wrt]

    # Compute second order partials
    # For example (for D1): d^2E/dD1^2, d^2E/dD1D2, d^2E/dD1S1, and so on...
    sop = [se.Matrix([d]).jacobian(wrt) for d in dv]

    # Create functions for each derivative listed above
    derivatives = [create_function(d, inputs) for d in dv]

    # Create functions all second order partials
    second_order_partials = [create_function(s, inputs) for s in sop]

    print('Completed second contact derivative and second order partials: {:.3f} seconds'.format(time() - start))

    return derivatives, second_order_partials


def ffr_jacobian():
    """
        Obtain Jacobian of friction force.
        Returns 3x24 Jacobian. dFfr/dy where y = [velocity; Fn]
    """
    start = time()
    print("Starting friction jacobian...")
    v1s = se.symarray('v1s', 3)
    v1e = se.symarray('v1e', 3)
    v2s = se.symarray('v2s', 3)
    v2e = se.symarray('v2e', 3)
    f1s = se.symarray('f1s', 3)
    f1e = se.symarray('f1e', 3)
    f2s = se.symarray('f2s', 3)
    f2e = se.symarray('f2e', 3)
    mu_k = se.symbols('mu_k')

    fn1 = se.sqrt(((f1s + f1e)**2).sum())
    fn2 = se.sqrt(((f2s + f2e)**2).sum())

    fn1_u = (f1s + f1e) / fn1
    fn2_u = (f2s + f2e) / fn2

    v1 = 0.5 * (v1s + v1e)
    v2 = 0.5 * (v2s + v2e)
    v_rel1 = v1 - v2
    tv_rel1 = v_rel1 - (v_rel1.dot(fn1_u) * fn1_u)
    tv_rel1_n = se.sqrt((tv_rel1 ** 2).sum())
    tv_rel1_u = tv_rel1 / se.sqrt((tv_rel1 ** 2).sum())

    v_rel2 = -v_rel1
    tv_rel2 = v_rel2 - (v_rel2.dot(fn2_u) * fn2_u)
    tv_rel2_n = se.sqrt((tv_rel2 ** 2).sum())
    tv_rel2_u = tv_rel2 / tv_rel2_n

    heaviside1 = 1 / (1 + se.exp(-50.0 * (tv_rel1_n - 0.15)))
    heaviside2 = 1 / (1 + se.exp(-50.0 * (tv_rel2_n - 0.15)))

    ffr_e1 = heaviside1 * mu_k * tv_rel1_u * fn1
    ffr_e2 = heaviside2 * mu_k * tv_rel2_u * fn2

    inputs = se.Matrix([*v1s, *v1e, *v2s, *v2e, *f1s, *f1e, *f2s, *f2e, mu_k])

    ffr = se.Matrix([*ffr_e1, *ffr_e2])

    wrt = se.Matrix([*f1s, *f1e, *f2s, *f2e])

    ffr_grad = create_function(ffr.jacobian(wrt), inputs)

    print('Completed friction jacobian: {:.3f} seconds'.format(time() - start))

    return ffr_grad


def main():
    if len(sys.argv) != 3: raise ValueError("Expects two arguments, ce_k and h2")
    ce_k = float(sys.argv[1])
    h2   = float(sys.argv[2])
    generate_functions(ce_k, h2)


if __name__  == '__main__':
    main()
