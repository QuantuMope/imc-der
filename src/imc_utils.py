import numpy as np
from numba import njit

"""
    Contains a wide assortment of utility code for IMC.
"""


@njit(cache=True)
def min_dist_vectorized(edges):
    edges = edges[:, :12]
    assert edges.shape[1] == 12
    num_inputs = edges.shape[0]
    x1s, x1e, x2s, x2e = edges[:, :3], edges[:, 3:6], edges[:, 6:9], edges[:, 9:]
    e1 = x1e - x1s
    e2 = x2e - x2s
    e12 = x2s - x1s
    D1 = np.sum(e1 ** 2, axis=1)  # D1
    D2 = np.sum(e2 ** 2, axis=1)  # D2
    R = np.sum(np.multiply(e1, e2), axis=1)  # R
    S1 = np.sum(np.multiply(e1, e12), axis=1)  # S1
    S2 = np.sum(np.multiply(e2, e12), axis=1)  # S2
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
    dist = np.sqrt(np.sum((e1*t - e2*u - e12)**2, axis=1))
    return dist


@njit(cache=True)
def min_dist_f_out_vectorized(edges, k=50.0):
    assert edges.shape[1] == 12
    num_inputs = edges.shape[0]
    f_out_vals = np.zeros((num_inputs, 15), dtype=np.float64)
    x1s, x1e, x2s, x2e = edges[:, :3], edges[:, 3:6], edges[:, 6:9], edges[:, 9:]
    f_out_vals[:, :3] = e1 = x1e - x1s
    f_out_vals[:, 3:6] = e2 = x2e - x2s
    f_out_vals[:, 6:9] = e12 = x2s - x1s
    f_out_vals[:, 9] = D1 = np.sum(e1 ** 2, axis=1)  # D1
    f_out_vals[:, 10] = D2 = np.sum(e2 ** 2, axis=1)  # D2
    f_out_vals[:, 11] = R = np.sum(np.multiply(e1, e2), axis=1)  # R
    f_out_vals[:, 12] = S1 = np.sum(np.multiply(e1, e12), axis=1)  # S1
    f_out_vals[:, 13] = S2 = np.sum(np.multiply(e2, e12), axis=1)  # S2
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

    con_dir = e1*t - e2*u - e12

    # Avoid np.linalg.norm due to lack of support for axis keyword in numba jit
    dist = np.sqrt((con_dir**2).sum(axis=1))

    return dist, f_out_vals


@njit(cache=True)
def compute_friction(data, forces, mu_k, dt, vel_tol):
    x1s, x1e = data[:, :3], data[:, 3:6]
    x2s, x2e = data[:, 6:9], data[:, 9:12]
    x1s0, x1e0 = data[:, 12:15], data[:, 15:18]
    x2s0, x2e0 = data[:, 18:21], data[:, 21:24]
    f1s, f1e = forces[:, :3], forces[:, 3:6]
    f2s, f2e = forces[:, 6:9], forces[:, 9:]

    num_inputs = data.shape[0]

    f1s_n = np.sqrt((f1s**2).sum(axis=1))
    f2s_n = np.sqrt((f2s**2).sum(axis=1))

    # fn should be the same whether we compute using fc1 or fc2
    fn = np.sqrt(((f1s + f1e) ** 2).sum(axis=1))
    ffr_val = mu_k * fn

    # Compute contact point ratios
    t1 = f1s_n / fn
    u1 = f2s_n / fn

    # Due to numerical errors, make sure contact points are in appropriate range
    t1[t1 > 1] = 1
    t1[t1 < 0] = 0
    u1[u1 > 1] = 1
    u1[u1 < 0] = 0
    t2 = 1 - t1
    u2 = 1 - u1
    t1 = t1.reshape((num_inputs, 1))
    t2 = t2.reshape((num_inputs, 1))
    u1 = u1.reshape((num_inputs, 1))
    u2 = u2.reshape((num_inputs, 1))

    v1s = (x1s - x1s0)
    v1e = (x1e - x1e0)
    v2s = (x2s - x2s0)
    v2e = (x2e - x2e0)

    v1 = t1 * v1s + t2 * v1e
    v2 = u1 * v2s + u2 * v2e
    v_rel = v1 - v2

    # Only consider tangential relative velocity
    norm = (f1s + f1e) / fn.reshape((num_inputs, 1))
    rem = np.zeros_like(v_rel)
    for i in range(num_inputs):
        rem[i] = v_rel[i].dot(norm[i]) * norm[i]
    tv_rel = v_rel - rem
    tv_rel_n = (np.sqrt((tv_rel ** 2).sum(axis=1))).reshape((num_inputs, 1))

    tv_rel_u = tv_rel / tv_rel_n

    tv_rel_n *= 1 / dt * vel_tol
    heaviside = 2 / (1 + np.exp(-tv_rel_n)) - 1

    # print("{:.3f} {:.3f} {:.3f} | {:.3f} {:.3f} {:.3f}".format(np.mean(heaviside), np.max(heaviside), np.min(heaviside),
    #                                                            np.mean(tv_rel_n), np.max(tv_rel_n), np.min(tv_rel_n)))

    ffr_e = heaviside * tv_rel_u * ffr_val.reshape((num_inputs, 1))

    ffr = np.zeros((num_inputs, 12), dtype=np.float64)

    ffr[:, :3] = t1 * ffr_e
    ffr[:, 3:6] = t2 * ffr_e
    ffr[:, 6:9] = u1 * -ffr_e
    ffr[:, 9:] = u2 * -ffr_e

    return ffr


@njit(cache=True)
def construct_possible_edge_combos(edges, edge_combos, node_data, ia, scale):
    num_edges = edges.shape[0]

    # Construct list of all edge coordinates
    for i in range(num_edges):
        edges[i] = node_data[3*i:(3*i)+6]

    # Construct list of all possible edge combinations without duplicates (excluding adjacent edges)
    ri = 0  # real index
    for i in range(num_edges):
        base_edge = edges[i]
        add = num_edges - i - (ia + 1)
        edge_combos[ri:ri+add, :6] = base_edge
        edge_combos[ri:ri+add, 6:] = edges[i+ia+1:]
        ri += add

    edge_combos *= scale


@njit(cache=True)
def detect_collisions(edge_combos, edge_ids, collision_limit, contact_len):
    # Compute the min-distances of all possible edge combinations
    assert edge_combos.shape[1] == 12
    num_inputs = edge_combos.shape[0]
    x1s, x1e, x2s, x2e = edge_combos[:, :3], edge_combos[:, 3:6], edge_combos[:, 6:9], edge_combos[:, 9:]
    e1 = x1e - x1s
    e2 = x2e - x2s
    e12 = x2s - x1s
    D1 = np.sum(e1 ** 2, axis=1)  # D1
    D2 = np.sum(e2 ** 2, axis=1)  # D2
    R = np.sum(np.multiply(e1, e2), axis=1)  # R
    S1 = np.sum(np.multiply(e1, e12), axis=1)  # S1
    S2 = np.sum(np.multiply(e2, e12), axis=1)  # S2
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
    minDs = np.sqrt(np.sum((e1*t - e2*u - e12)**2, axis=1))

    # Compute the indices of all edge combinations within the collision limit
    col_indices = np.where(minDs - contact_len < collision_limit)

    # Extract data for "in contact" edges
    f_edge_ids = edge_ids[col_indices]

    closest_distance = np.min(minDs)

    return f_edge_ids, closest_distance


@njit(cache=True)
def prepare_edges(edge_ids, edge_combos, first_iter, node_data, prev_node_data):
    if first_iter:
        for i, (e1, e2) in enumerate(edge_ids):
            edge_combos[i, :6] = node_data[3*e1:(3*e1)+6]
            edge_combos[i, 6:12] = node_data[3*e2:(3*e2)+6]
            edge_combos[i, 12:18] = prev_node_data[3*e1:(3*e1)+6]
            edge_combos[i, 18:] = prev_node_data[3*e2:(3*e2)+6]
    else:
        for i, (e1, e2) in enumerate(edge_ids):
            edge_combos[i, :6] = node_data[3*e1:(3*e1)+6]
            edge_combos[i, 6:12] = node_data[3*e2:(3*e2)+6]


@njit(cache=True)
def prepare_velocities(edge_ids, velocity_data):
    num_edges = edge_ids.shape[0]
    velocities = np.zeros((num_edges, 12), dtype=np.float64)

    for i, (e1, e2) in enumerate(edge_ids):
        velocities[i, :6] = velocity_data[3*e1:(3*e1)+6]
        velocities[i, 6:] = velocity_data[3*e2:(3*e2)+6]

    return velocities


@njit(cache=True)
def chain_rule_contact_nohess(de_grads, f_grad_vals, s_derv_vals):
    num_inputs = s_derv_vals.shape[0]
    de_dx = np.zeros((num_inputs, 12), dtype=np.float64)

    de_dx += s_derv_vals[:, :9] @ de_grads
    s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1))
    for i in range(6):
        de_dx += s_derv_vals[:, i+9] * f_grad_vals[i]

    return de_dx


def chain_rule_contact_hess(de_grads, f_grad_vals, s_derv_vals, f_hess_vals, f_hess_const, s_sopa_vals):
    """  This function is more efficient without numba njit
         since we can do 3D matrix operation without for loop.

         Chain rule operations that can be sped up with njit are done so in optimize_chain_rule(...)
    """
    num_inputs = s_derv_vals.shape[0]
    de_dx = np.zeros((num_inputs, 12), dtype=np.float64)
    d2e_dx2 = np.zeros((num_inputs, 12, 12), dtype=np.float64)

    # Optimize chain rule for code that involves only 2D arrays using numba jit
    s_derv2_vals = np.zeros((15, num_inputs, 12), dtype=np.float64)
    optimize_chain_rule(de_grads, f_grad_vals, s_derv_vals, s_sopa_vals, s_derv2_vals, de_dx)

    # Perform product rule and chain rule for contact hessian
    s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1, 1))
    s_derv2_vals = s_derv2_vals.reshape((15, num_inputs, 12, 1))
    f_grad_vals = f_grad_vals.reshape((6, num_inputs, 1, 12))
    de = de_grads.reshape((9, 1, 12))
    for i in range(9):
        d2e_dx2 += s_derv2_vals[i] @ de[i]
    for i in range(5):
        d2e_dx2 += s_derv_vals[:, i+9] * f_hess_const[i]
        d2e_dx2 += s_derv2_vals[i + 9] @ f_grad_vals[i]
    d2e_dx2 += s_derv_vals[:, -1] * f_hess_vals
    d2e_dx2 += s_derv2_vals[-1] @ f_grad_vals[-1]

    return de_dx, d2e_dx2


@njit(cache=True)
def optimize_chain_rule(de_grads, f_grad_vals, s_derv_vals, s_sopa_vals, s_derv2_vals, de_dx):
    num_inputs = s_derv_vals.shape[0]
    # Perform chain rule for contact gradient (forces)
    # dE/dx = dE/de1 * de1/dx + dE/de2 * de2/dx + and so on
    de_dx += s_derv_vals[:, :9] @ de_grads
    s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1))
    for i in range(6):
        de_dx += s_derv_vals[:, i+9] * f_grad_vals[i]

    # Perform chain rule to obtain d^2E/dD1^2, d^2E/dD1D2, and so on
    # These are necessary to compute the chain rule for computing the overall hessian.
    s_sopa_vals = s_sopa_vals.reshape((15, num_inputs, 15, 1))
    for i in range(15):
        curr_sopa = s_sopa_vals[i]
        curr_derv2 = s_derv2_vals[i]
        for j in range(9):
            curr_derv2 += curr_sopa[:, j] * de_grads[j]
        for j in range(6):
            curr_derv2 += curr_sopa[:, j+9] * f_grad_vals[j]


def chain_rule_friction_jacobian(dfr_dfc, dfc_dx):
    """  This function is more efficient without numba njit
         since we can do 3D matrix operation without for loop.

         Note that d2e_dx2 := dfc_dx
    """
    return dfr_dfc[:, :, :12] + dfr_dfc[:, :, 12:] @ dfc_dx


@njit(cache=True)
def get_friction_jacobian_inputs(data, contact_force, mu_k, dt, vel_tol):
    ffr_jac_input = np.zeros((data.shape[0], 39), dtype=np.float64)

    ffr_jac_input[:, :24] = data
    ffr_jac_input[:, 24:36] = contact_force
    ffr_jac_input[:, 36] = mu_k
    ffr_jac_input[:, 37] = dt
    ffr_jac_input[:, 38] = vel_tol

    return ffr_jac_input


@njit(cache=True)
def py_to_cpp_nohess(py_forces, cpp_forces, edge_ids):
    for i in range(py_forces.shape[0]):
        forces = py_forces[i]

        # Enter into global force and hessian container.
        e1, e2 = edge_ids[i][0]*3, edge_ids[i][1]*3

        cpp_forces[e1:e1+6] += forces[:6]
        cpp_forces[e2:e2+6] += forces[6:]


@njit(cache=True)
def py_to_cpp_hess(py_forces, py_jacobian, cpp_forces, cpp_jacobian, edge_ids):
    for i in range(py_forces.shape[0]):
        forces = py_forces[i]
        jacobian = py_jacobian[i]

        # Enter into global force and hessian container.
        e1, e2 = edge_ids[i][0]*3, edge_ids[i][1]*3

        cpp_forces[e1:e1+6] += forces[:6]
        cpp_forces[e2:e2+6] += forces[6:]

        cpp_jacobian[e1:e1+6, e1:e1+6] += jacobian[:6, :6]
        cpp_jacobian[e1:e1+6, e2:e2+6] += jacobian[:6, 6:]
        cpp_jacobian[e2:e2+6, e1:e1+6] += jacobian[6:, :6]
        cpp_jacobian[e2:e2+6, e2:e2+6] += jacobian[6:, 6:]
