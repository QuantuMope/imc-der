import numpy as np
import symengine as se
from numba import njit

"""
    Contains a wide assortment of utility code for IMC.
"""


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


@njit
def min_dist_vectorized(edges):
    """
        This function is capable of auto-parallelization by numba jit compiler.
        Time comparisons are done on single-cpu for fair comparisons.
        Switch to @njit(parallel=True) for speed.
    """
    assert edges.shape[1] == 12
    num_inputs = edges.shape[0]
    x1s, x1e, x2s, x2e = edges[:, :3], edges[:, 3:6], edges[:, 6:9], edges[:, 9:]
    d1 = x1e - x1s
    d2 = x2e - x2s
    d12 = x2s - x1s
    D1 = np.sum(d1 ** 2, axis=1)  # D1
    D2 = np.sum(d2 ** 2, axis=1)  # D2
    R = np.sum(np.multiply(d1, d2), axis=1)  # R
    S1 = np.sum(np.multiply(d1, d12), axis=1)  # S1
    S2 = np.sum(np.multiply(d2, d12), axis=1)  # S2
    den = D1 * D2 - R ** 2
    t = np.zeros((num_inputs,))
    non_parallels = den != 0
    t[non_parallels] = ((S1 * D2 - S2 * R) / den)[non_parallels]
    t[t > 1.] = 1.
    t[t < 0.] = 0.
    u = (t * R - S2) / D2
    uf = u.copy()
    uf[uf > 1.] = 1.
    uf[uf < 0.] = 0.
    t[uf != u] = ((uf * R + S1) / D1)[uf != u]
    t[t > 1.] = 1.
    t[t < 0.] = 0.
    u[uf != u] = uf[uf != u]
    t = t.reshape((num_inputs, 1))
    u = u.reshape((num_inputs, 1))

    # Avoid np.linalg.norm due to lack of support for axis keyword in numba jit
    dist = np.sqrt(np.sum((d1*t - d2*u - d12)**2, axis=1))
    return dist


@njit
def min_dist_f_out_vectorized(edges, k=50.0):
    assert edges.shape[1] == 12
    num_inputs = edges.shape[0]
    f_out_vals = np.zeros((num_inputs, 15), dtype=np.float64)
    x1s, x1e, x2s, x2e = edges[:, :3], edges[:, 3:6], edges[:, 6:9], edges[:, 9:]
    f_out_vals[:, :3] = d1 = x1e - x1s
    f_out_vals[:, 3:6] = d2 = x2e - x2s
    f_out_vals[:, 6:9] = d12 = x2s - x1s
    f_out_vals[:, 9] = D1 = np.sum(d1 ** 2, axis=1)  # D1
    f_out_vals[:, 10] = D2 = np.sum(d2 ** 2, axis=1)  # D2
    f_out_vals[:, 11] = R = np.sum(np.multiply(d1, d2), axis=1)  # R
    f_out_vals[:, 12] = S1 = np.sum(np.multiply(d1, d12), axis=1)  # S1
    f_out_vals[:, 13] = S2 = np.sum(np.multiply(d2, d12), axis=1)  # S2
    den = D1 * D2 - R ** 2
    t = np.zeros((num_inputs,))
    non_parallels = den != 0
    t[non_parallels] = ((S1 * D2 - S2 * R) / den)[non_parallels]
    f_out_vals[:, 14] = (1 / k) * (np.log(1 + np.exp(k * t)) - np.log(1 + np.exp(k * (t - 1.0))))
    t[t > 1.] = 1.
    t[t < 0.] = 0.
    u = (t * R - S2) / D2
    uf = u.copy()
    uf[uf > 1.] = 1.
    uf[uf < 0.] = 0.
    t[uf != u] = ((uf * R + S1) / D1)[uf != u]
    t[t > 1.] = 1.
    t[t < 0.] = 0.
    u[uf != u] = uf[uf != u]
    t = t.reshape((num_inputs, 1))
    u = u.reshape((num_inputs, 1))

    con_dir = d1*t - d2*u - d12

    # Avoid np.linalg.norm due to lack of support for axis keyword in numba jit
    dist = np.sqrt((con_dir**2).sum(axis=1))

    return dist, f_out_vals
