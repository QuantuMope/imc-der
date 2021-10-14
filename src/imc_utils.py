import numpy as np
from numba import njit

"""
    Contains a wide assortment of utility code for IMC.
"""


@njit(cache=True)
def min_dist_vectorized(edges):
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


@njit(cache=True)
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


@njit(cache=True)
def compute_ffr(velocities, forces, mu_k):
    v1s, v1e, v2s, v2e = velocities[:, :3], velocities[:, 3:6], velocities[:, 6:9], velocities[:, 9:]
    f1s, f1e = forces[:, :3], forces[:, 3:6]

    num_inputs = velocities.shape[0]

    fn = np.sqrt(((f1s + f1e) ** 2).sum(axis=1))
    ffr_val = mu_k * fn

    v1 = 0.5 * (v1s + v1e)
    v2 = 0.5 * (v2s + v2e)
    v_rel = v1 - v2

    # Only consider tangential relative velocity
    norm = (f1s + f1e) / fn.reshape((num_inputs, 1))
    rem = np.zeros_like(v_rel)
    for i in range(num_inputs):
        rem[i] = v_rel[i].dot(norm[i]) * norm[i]
    tv_rel = v_rel - rem
    tv_rel_n = (np.sqrt((tv_rel ** 2).sum(axis=1))).reshape((num_inputs, 1))

    heaviside = 1 / (1 + np.exp(-50.0 * (tv_rel_n - 0.15)))

    tv_rel_u = tv_rel / tv_rel_n

    ffr_e = 0.5 * heaviside * tv_rel_u * ffr_val.reshape((num_inputs, 1))

    ffr = np.zeros_like(velocities, dtype=np.float64)

    ffr[:, :3] = ffr_e
    ffr[:, 3:6] = ffr_e
    ffr[:, 6:9] = -ffr_e
    ffr[:, 9:] = -ffr_e

    return ffr


# @njit(cache=True)
@njit(cache=True)
def construct_possible_edge_combos(edges, edge_combos, node_data, ia):
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


@njit(cache=True)
def detect_collisions(edge_combos, edge_ids, collision_limit, contact_len):
    # Compute the min-distances of all possible edge combinations
    assert edge_combos.shape[1] == 12
    num_inputs = edge_combos.shape[0]
    x1s, x1e, x2s, x2e = edge_combos[:, :3], edge_combos[:, 3:6], edge_combos[:, 6:9], edge_combos[:, 9:]
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
    minDs = np.sqrt(np.sum((d1*t - d2*u - d12)**2, axis=1))

    # Compute the indices of all edge combinations within the collision limit
    col_indices = np.where(minDs - contact_len < collision_limit)

    # Extract data for "in contact" edges
    f_edge_ids = edge_ids[col_indices]

    closest_distance = np.min(minDs)

    return f_edge_ids, closest_distance


@njit(cache=True)
def prepare_edges(edge_ids, node_data):
    num_edges = edge_ids.shape[0]
    edge_combos = np.zeros((num_edges, 12), dtype=np.float64)

    # for i, ids in enumerate(edge_ids):
    for i in range(num_edges):
        x, y = edge_ids[i]
        edge_combos[i, :6] = node_data[3*x:(3*x)+6]
        edge_combos[i, 6:] = node_data[3*y:(3*y)+6]
    dists, f_out_vals = min_dist_f_out_vectorized(edge_combos)

    closest_distance = np.min(dists)

    return edge_combos, closest_distance, f_out_vals


@njit(cache=True)
def prepare_velocities(edge_ids, velocity_data):
    num_edges = edge_ids.shape[0]
    velocities = np.zeros((num_edges, 12), dtype=np.float64)

    # for i, ids in enumerate(edge_ids):
    for i in range(num_edges):
        x, y = edge_ids[i]
        vel_x = velocity_data[3*x:(3*x)+6]
        vel_y = velocity_data[3*y:(3*y)+6]
        velocities[i, :6] = vel_x
        velocities[i, 6:] = vel_y

    return velocities


@njit(cache=True)
def chain_rule_contact_nohess(de_grads, f_grad_vals, s_derv_vals):
    num_inputs = s_derv_vals.shape[0]
    dedx = np.zeros((num_inputs, 12), dtype=np.float64)
    dedx[:] += s_derv_vals[:, :9] @ de_grads
    s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1))
    for i in range(6):
        dedx[:] += s_derv_vals[:, i+9] * f_grad_vals[i]

    return dedx


@njit(cache=True)
def optimize_chain_rule(de_grads, f_grad_vals, s_derv_vals, s_sopa_vals, s_derv2_vals, dedx):
    num_inputs = s_derv_vals.shape[0]
    # Perform chain rule for contact gradient (forces)
    # dE/dx = dE/de1 * de1/dx + dE/de2 * de2/dx + and so on
    dedx[:] += s_derv_vals[:, :9] @ de_grads
    s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1))
    for i in range(6):
        dedx[:] += s_derv_vals[:, i+9] * f_grad_vals[i]

    # Perform chain rule to obtain d^2E/dD1^2, d^2E/dD1D2, and so on
    # These are necessary to compute the chain rule for computing the overall hessian.
    s_sopa_vals = s_sopa_vals.reshape((15, num_inputs, 15, 1))
    for i in range(15):
        curr_sopa = s_sopa_vals[i]
        curr_derv2 = s_derv2_vals[i, :]
        for j in range(9):
            curr_derv2 += curr_sopa[:, j] * de_grads[j]
        for j in range(6):
            curr_derv2 += curr_sopa[:, j+9] * f_grad_vals[j]


@njit(cache=True)
def get_ffr_jacobian_inputs(velocities, dedx, mu_k):
    ffr_jac_input = np.zeros((velocities.shape[0], 25), dtype=np.float64)

    ffr_jac_input[:, :12] = velocities
    ffr_jac_input[:, 12:24] = dedx
    ffr_jac_input[:, 24] = mu_k

    return ffr_jac_input


@njit(cache=True)
def py_to_cpp_nohess(py_forces, cpp_forces, edge_ids):
    for i in range(py_forces.shape[0]):
        forces = py_forces[i]

        # Enter into global force and hessian container.
        e1, e2 = edge_ids[i]

        cpp_forces[(3 * e1):(3 * e1) + 6] += forces[:6]
        cpp_forces[(3 * e2):(3 * e2) + 6] += forces[6:]


@njit(cache=True)
def py_to_cpp_hess(py_forces, py_jacobian, cpp_forces, cpp_jacobian, edge_ids):
    for i in range(py_forces.shape[0]):
        forces = py_forces[i]
        jacobian = py_jacobian[i]

        # Enter into global force and hessian container.
        e1, e2 = edge_ids[i]

        cpp_forces[(3 * e1):(3 * e1) + 6] += forces[:6]
        cpp_forces[(3 * e2):(3 * e2) + 6] += forces[6:]

        cpp_jacobian[3*e1:3*e1+6, 3*e1:3*e1+6] += jacobian[:6, :6]
        cpp_jacobian[3*e1:3*e1+6, 3*e2:3*e2+6] += jacobian[:6, 6:]
        cpp_jacobian[3*e2:3*e2+6, 3*e1:3*e1+6] += jacobian[6:, :6]
        cpp_jacobian[3*e2:3*e2+6, 3*e2:3*e2+6] += jacobian[6:, 6:]
