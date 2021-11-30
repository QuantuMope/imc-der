import os
import sys
import dill as pickle
import symengine as se
from time import time


def create_function(ele, wrt):
    return se.Lambdify(wrt, ele, cse=True, backend="llvm")


def approx_fixbound(x, k):
    """ H(x) """
    return (1 / k) * (se.log(1 + se.exp(k * x)) - se.log(1 + se.exp(k * (x - 1.0))))


def log_func(x, k, c):
    return 1 / (1 + se.exp(-k*(x-c)))


def approx_boxcar(x, k):
    """ B(x) """
    step1 = log_func(x, k, 0.00)
    step2 = log_func(x, k, 1.00)
    return step1 - step2


def generate_functions(ce_k, h2):
    dir = './DER/grads_hessian_functions/'

    if not os.path.exists(dir): os.makedirs(dir)

    ce_grad_func, ce_hess_func = generate_contact_energy_functions(ce_k, h2)
    fr_jaco_func = generate_friction_jacobian_function()

    ce_params = '_cek_' + str(ce_k) + '_h2_' + str(h2)

    functions = [(ce_grad_func, 'ce_grad' + ce_params),
                 (ce_hess_func, 'ce_hess' + ce_params),
                 (fr_jaco_func, 'fr_jaco')]
    #
    pickle.settings['recurse'] = True
    for func, name in functions:
        with open(dir + name, 'wb') as f:
            pickle.dump(func, f)


def generate_contact_energy_functions(ce_k, h2, k1=50.0, k2=50.0):
    """
        Obtain gradient and Hessian of contact energy.
        The gradient can be thought of as our contact forces.
        The Hessian can be thought of as our contact force Jacobian.
    """
    start = time()
    print('Starting contact energy gradient and Hessian...')

    # Declare symbolic variables.
    x1s = se.symarray('x1s', 3)
    x1e = se.symarray('x1e', 3)
    x2s = se.symarray('x2s', 3)
    x2e = se.symarray('x2e', 3)

    # Smoothly approximated contact energy formulation.
    e1 = x1e - x1s
    e2 = x2e - x2s
    e12 = x2s - x1s
    D1 = (e1 ** 2).sum()
    D2 = (e2 ** 2).sum()
    S1 = (e1 * e12).sum()
    S2 = (e2 * e12).sum()
    R = (e1 * e2).sum()
    den = D1 * D2 - R ** 2
    t1 = se.Piecewise(((S1 * D2 - S2 * R) / den, den > 1e-6), (0.0, True))  # avoid division by zero
    t2 = approx_fixbound(t1, k=k1)
    u1 = (t2 * R - S2) / D2
    u2 = approx_fixbound(u1, k=k1)
    conditional = approx_boxcar(u1, k=k2)
    t3 = (1 - conditional) * (u2 * R + S1) / D1 + conditional * t2
    dist = se.sqrt(((e1 * t3 - e2 * u2 - e12)**2).sum())
    E = (1 / ce_k) * se.log(1 + se.exp(ce_k * (h2 - dist)))

    inputs = wrt = se.Matrix([*x1s, *x1e, *x2s, *x2e])

    # Generate contact energy gradient and Hessian symbolic solutions.
    ce_grad = se.Matrix([E]).jacobian(wrt)
    ce_hess = ce_grad.T.jacobian(wrt)

    # Generate callable functions using symengine and LLVM backend.
    ce_grad_func = create_function(ce_grad, inputs)
    ce_hess_func = create_function(ce_hess, inputs)

    print('Completed first contact gradient and Hessian in {:.3f} seconds.'.format(time() - start))

    return ce_grad_func, ce_hess_func


def generate_friction_jacobian_function():
    """
        Obtain Jacobian of friction force.
        Treat friction direction explicitly by using previous time step's velocity.
        The contact force magnitude is treated implicitly using our contact energy Hessian.
    """
    start = time()
    print("Starting friction Jacobian...")
    x1s = se.symarray('x1s', 3)
    x1e = se.symarray('x1e', 3)
    x2s = se.symarray('x2s', 3)
    x2e = se.symarray('x2e', 3)
    x1s0 = se.symarray('x1s0', 3)
    x1e0 = se.symarray('x1e0', 3)
    x2s0 = se.symarray('x2s0', 3)
    x2e0 = se.symarray('x2e0', 3)
    f1s = se.symarray('f1s', 3)
    f1e = se.symarray('f1e', 3)
    f2s = se.symarray('f2s', 3)
    f2e = se.symarray('f2e', 3)
    mu_k, dt, vel_tol = se.symbols('mu_k, dt, vel_tol')

    # We can take advantage of the fact that force vectors are always parallel
    # and that their magnitudes vary accordingly with t and u.
    # This allows us to avoid computing dt/dx and du/dx and instead depend entirely
    # on dFc/dx (d2E/dx2) for the friction force jacobian computation.
    f1s_n = se.sqrt((f1s**2).sum())
    f1e_n = se.sqrt((f1e**2).sum())
    f2s_n = se.sqrt((f2s**2).sum())
    f2e_n = se.sqrt((f2e**2).sum())
    fn1 = se.sqrt(((f1s + f1e)**2).sum())
    fn2 = se.sqrt(((f2s + f2e)**2).sum())

    # Compute contact point ratios
    t1 = f1s_n / fn1
    t2 = f1e_n / fn1
    u1 = f2s_n / fn2
    u2 = f2e_n / fn2

    fn1_u = (f1s + f1e) / fn1
    fn2_u = (f2s + f2e) / fn2

    v1s = (x1s - x1s0)
    v1e = (x1e - x1e0)
    v2s = (x2s - x2s0)
    v2e = (x2e - x2e0)

    v1 = t1 * v1s + t2 * v1e
    v2 = u1 * v2s + u2 * v2e
    v_rel1 = v1 - v2
    tv_rel1 = v_rel1 - (v_rel1.dot(fn1_u) * fn1_u)
    tv_rel1_n = se.sqrt((tv_rel1 ** 2).sum())
    tv_rel1_u = tv_rel1 / tv_rel1_n

    v_rel2 = -v_rel1
    tv_rel2 = v_rel2 - (v_rel2.dot(fn2_u) * fn2_u)
    tv_rel2_n = se.sqrt((tv_rel2 ** 2).sum())
    tv_rel2_u = tv_rel2 / tv_rel2_n

    tv_rel1_n *= 1 / dt * vel_tol
    tv_rel2_n *= 1 / dt * vel_tol
    heaviside1 = 2 / (1 + se.exp(-tv_rel1_n)) - 1
    heaviside2 = 2 / (1 + se.exp(-tv_rel2_n)) - 1

    ffr_e1 = heaviside1 * mu_k * tv_rel1_u * fn1
    ffr_e2 = heaviside2 * mu_k * tv_rel2_u * fn2

    ffr1s = t1 * ffr_e1
    ffr1e = t2 * ffr_e1
    ffr2s = u1 * ffr_e2
    ffr2e = u2 * ffr_e2

    inputs = se.Matrix([*x1s, *x1e, *x2s, *x2e, *x1s0, *x1e0, *x2s0, *x2e0, *f1s, *f1e, *f2s, *f2e, mu_k, dt, vel_tol])

    wrt = se.Matrix([*x1s, *x1e, *x2s, *x2e, *f1s, *f1e, *f2s, *f2e])

    ffr = se.Matrix([*ffr1s, *ffr1e, *ffr2s, *ffr2e])

    fr_jaco = ffr.jacobian(wrt)

    fr_jaco_func = create_function(fr_jaco, inputs)

    print('Completed friction Jacobian in {:.3f} seconds.'.format(time() - start))

    return fr_jaco_func


def main():
    if len(sys.argv) != 3: raise ValueError("Expects two arguments, ce_k and h2")
    ce_k = float(sys.argv[1])
    h2 = float(sys.argv[2])
    generate_functions(ce_k, h2)


if __name__  == '__main__':
    main()
