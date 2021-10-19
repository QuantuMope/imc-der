import sys
import mmap
import zmq
import posix_ipc
import numpy as np
import dill as pickle
import imc_utils as iu
from numba import njit


class IMC:
    def __init__(self, params):
        # Load parameters
        radius = params['radius']
        self.scale = params['S']
        self.radius = radius * self.scale  # all values normalized by self.scale
        num_nodes = params['num_nodes']
        self.num_edges = num_nodes - 1
        self.collision_limit = params['collision_limit']
        self.contact_stiffness = params['contact_stiffness']
        self.mu_k = params['mu_k']

        self.contact_len = self.radius * 2
        ce_k = params['ce_k']
        cekr = '_cek_' + str(ce_k) + '_h2_' + str(self.contact_len)
        func_names = ['de', 'first_grad', 'constant_hess', 'first_hess', 'second_derivative' + cekr,
                      'second_order_partials' + cekr, 'friction_jacobian']

        # Load pre-generated functions
        dir = './grads_hessian_functions/'
        functions = []
        for name in func_names:
            with open(dir + name, 'rb') as f:
                functions.append(pickle.load(f))

        self.de_grads          = functions[0]
        self.f_grad_funcs      = functions[1]
        self.f_hess_const      = functions[2].reshape((5, 1, 12, 12))
        self.f_hess_func       = functions[3]  # for t2
        self.s_derv_funcs      = functions[4]
        self.s_sopa_funcs      = functions[5]
        self.ffr_jacobian_func = functions[6]

        self.friction = 0  # is updated by C++ side

        self.ia = 2  # number of adjacent edges to ignore contact

        # Calculate the number of possible edge combinations ignoring adjacent 5 edges
        # use iterative method to prevent overflow
        num_edge_combos = 0
        for i in range(self.num_edges):
            for j in range(i, self.num_edges):
                if i in range(j-self.ia, j+self.ia+1): continue
                num_edge_combos += 1

        self.indices = np.arange(0, self.num_edges)
        self.edges = np.zeros((self.num_edges, 6), dtype=np.float64)
        self.edge_combos = np.zeros((num_edge_combos, 12), dtype=np.float64)
        self.edge_ids = np.zeros((num_edge_combos, 2), dtype=np.int32)

        ri = 0  # real index
        for i in range(self.num_edges):
            add = self.num_edges - i - (self.ia+1)
            self.edge_ids[ri:ri+add, 0] = i
            self.edge_ids[ri:ri+add, 1] = self.indices[i+self.ia+1:]
            ri += add

        # Sizes for data structures
        nv = num_nodes * 3
        h_size = (nv, nv)
        meta_data_size = 7

        # Initialize shared memory
        self.port_no = sys.argv[1]
        self.forces = np.zeros(nv, dtype=np.float64)
        self.hessian = np.zeros(h_size, dtype=np.float64)
        np.ascontiguousarray(self.forces, dtype=np.float64)
        np.ascontiguousarray(self.hessian, dtype=np.float64)
        assert self.forces.flags['C_CONTIGUOUS'] is True
        assert self.hessian.flags['C_CONTIGUOUS'] is True
        node_bytes = self.forces.nbytes
        hess_bytes = self.hessian.nbytes
        meta_bytes = np.zeros(meta_data_size, dtype=np.float64).nbytes
        n = posix_ipc.SharedMemory('node_coordinates' + self.port_no, size=node_bytes, read_only=False)
        u = posix_ipc.SharedMemory('velocities' + self.port_no, size=node_bytes, read_only=False)
        f = posix_ipc.SharedMemory('contact_forces' + self.port_no, size=node_bytes, read_only=False)
        h = posix_ipc.SharedMemory('contact_hessian' + self.port_no, size=hess_bytes, read_only=False)
        m = posix_ipc.SharedMemory('meta_data' + self.port_no, size=meta_bytes, read_only=False)
        self.node_coordinates = np.ndarray(nv, np.float64, mmap.mmap(n.fd, 0))
        self.velocities = np.ndarray(nv, np.float64, mmap.mmap(u.fd, 0))
        self.forces = np.ndarray(nv, np.float64, mmap.mmap(f.fd, 0))
        self.hessian = np.ndarray(h_size, np.float64, mmap.mmap(h.fd, 0))
        self.meta_data = np.ndarray(meta_data_size, np.float64, mmap.mmap(m.fd, 0))
        assert self.node_coordinates.flags['C_CONTIGUOUS'] is True
        assert self.velocities.flags['C_CONTIGUOUS'] is True
        assert self.forces.flags['C_CONTIGUOUS'] is True
        assert self.hessian.flags['C_CONTIGUOUS'] is True
        assert self.meta_data.flags['C_CONTIGUOUS'] is True

    def _construct_possible_edge_combos(self):
        return iu.construct_possible_edge_combos(self.edges, self.edge_combos, self.node_coordinates, self.ia)

    def _detect_collisions(self):
        return iu.detect_collisions(self.edge_combos, self.edge_ids, self.collision_limit, self.contact_len)

    # @staticmethod
    # @njit
    # def _jit_prepare_dvdx(edge_ids, velocities_pre, dt, velocities_curr):
    #     # Compute dv/dx for friction Jacobian. Using chain rule, dv/dx = a/v
    #     dvdx = np.zeros((edge_ids.shape[0], 12, 12), dtype=np.float64)
    #     for i, ids in enumerate(edge_ids):
    #         x, y = ids
    #         vel_x = velocities_curr[3*x:(3*x)+6]
    #         vel_y = velocities_curr[3*y:(3*y)+6]
    #         a_x = (vel_x - velocities_pre[3*x:(3*x)+6]) / dt
    #         a_y = (vel_y - velocities_pre[3*y:(3*y)+6]) / dt
    #         dvdx[i, :6, :6] = np.diag(a_x / vel_x)
    #         dvdx[i, 6:, 6:] = np.diag(a_y / vel_y)
    #
    #     return dvdx
    #
    # def _prepare_dvdx(self, edge_ids, velocities_pre, dt):
    #     return self._jit_prepare_dvdx(edge_ids, velocities_pre, dt, self.velocities)

    def _prepare_edges(self, edge_ids):
        return iu.prepare_edges(edge_ids, self.node_coordinates)

    def _prepare_velocities(self, edge_ids):
        return iu.prepare_velocities(edge_ids, self.velocities)

    def _chain_rule_contact_nohess(self, f_grad_vals, s_derv_vals):
        return iu.chain_rule_contact_nohess(self.de_grads, f_grad_vals, s_derv_vals)

    def _chain_rule_contact_hess(self, f_grad_vals, s_derv_vals, f_hess_vals, s_sopa_vals):
        num_inputs = s_derv_vals.shape[0]
        dedx = np.zeros((num_inputs, 12), dtype=np.float64)
        d2edx2 = np.zeros((num_inputs, 12, 12), dtype=np.float64)

        # Optimize chain rule for code that involves only 2D arrays using numba jit
        s_derv2_vals = np.zeros((15, num_inputs, 12), dtype=np.float64)
        iu.optimize_chain_rule(self.de_grads, f_grad_vals, s_derv_vals, s_sopa_vals, s_derv2_vals, dedx)

        # Perform product rule and chain rule for contact hessian
        s_derv_vals = s_derv_vals.reshape((num_inputs, 15, 1, 1))
        s_derv2_vals = s_derv2_vals.reshape((15, num_inputs, 12, 1))
        f_grad_vals = f_grad_vals.reshape((6, num_inputs, 1, 12))
        de = self.de_grads.reshape((9, 1, 12))
        for i in range(9):
            d2edx2[:] += s_derv2_vals[i] @ de[i]
        for i in range(5):
            d2edx2[:] += s_derv_vals[:, i+9] * self.f_hess_const[i]
            d2edx2[:] += s_derv2_vals[i + 9] @ f_grad_vals[i]
        d2edx2[:] += s_derv_vals[:, -1] * f_hess_vals
        d2edx2[:] += s_derv2_vals[-1] @ f_grad_vals[-1]

        return dedx, d2edx2

    def _compute_ffr(self, velocities, dedx):
        return iu.compute_ffr(velocities, dedx, self.mu_k)

    def _get_ffr_jacobian_inputs(self, velocities, dedx):
        return iu.get_ffr_jacobian_inputs(velocities, dedx, self.mu_k)

    @staticmethod
    def _chain_rule_friction_jacobian(ffr_grad_s, d2edx2):
        """  This function is more efficient without numba njit
             since we can do 3D matrix operation without for loop. """

        ffr_jacobian = np.zeros((ffr_grad_s.shape[0], 12, 12), dtype=np.float64)

        ffr_1 = 0.5 * (ffr_grad_s[:, :3] @ d2edx2)
        ffr_2 = 0.5 * (ffr_grad_s[:, 3:] @ d2edx2)
        ffr_jacobian[:, :3]  = ffr_1
        ffr_jacobian[:, 3:6] = ffr_1
        ffr_jacobian[:, 6:9] = ffr_2
        ffr_jacobian[:, 9:]  = ffr_2

        return ffr_jacobian

    def _py_to_cpp_nohess(self, py_forces, edge_ids):
        iu.py_to_cpp_nohess(py_forces, self.forces, edge_ids)

    def _py_to_cpp_hess(self, py_forces, py_jacobian, edge_ids):
        iu.py_to_cpp_hess(py_forces, py_jacobian, self.forces, self.hessian, edge_ids)

    def _get_forces(self, edges, edge_ids, velocities, s_input_vals):
        num_inputs = edges.shape[0]

        # Obtain first contact energy gradients
        f_grad_vals = np.array([f_grad(*edges) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()

        # Get second contact energy gradients
        s_derv_vals = np.array([s_derv(*s_input_vals) for s_derv in self.s_derv_funcs], dtype=np.float64).T

        # Reshape data structures for proper indexing in case of only one collision
        if num_inputs == 1:
            s_derv_vals = s_derv_vals.reshape((1, 15))
            f_grad_vals = f_grad_vals.reshape((6, 1, 12))

        # Perform chain ruling to get contact gradient and hessian
        dedx = self._chain_rule_contact_nohess(f_grad_vals, s_derv_vals)

        # Calculate friction forces on all four nodes
        ffr = self._compute_ffr(velocities, dedx)

        if self.friction:
            total_forces = dedx + ffr
        else:
            total_forces = dedx

        # Write gradient and hessian to shared memory location for DER
        self._py_to_cpp_nohess(total_forces, edge_ids)

        # Apply contact stiffness to gradient and hessian
        self.forces[:] *= self.contact_stiffness

    def _get_forces_and_hessian(self, edges, edge_ids, velocities, s_input_vals):
        num_inputs = edges.shape[0]

        # Obtain first contact energy gradients
        f_grad_vals = np.array([f_grad(*edges) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()

        # Obtain first contact energy hessians
        f_hess_vals = np.array(self.f_hess_func(*edges), dtype=np.float64)

        # Get second contact energy gradients
        s_derv_vals = np.array([s_derv(*s_input_vals) for s_derv in self.s_derv_funcs], dtype=np.float64).T

        # Get second contact energy hessians
        s_sopa_vals = np.array([s_sopa(*s_input_vals) for s_sopa in self.s_sopa_funcs], dtype=np.float64).squeeze()

        # Reshape data structures for proper indexing in case of only one collision
        if num_inputs == 1:
            s_derv_vals = s_derv_vals.reshape((1, 15))
            s_sopa_vals = s_sopa_vals.reshape((15, 1, 15))
            f_grad_vals = f_grad_vals.reshape((6, 1, 12))

        # Perform chain ruling to get contact gradient and hessian
        dedx, d2edx2 = self._chain_rule_contact_hess(f_grad_vals, s_derv_vals, f_hess_vals, s_sopa_vals)

        # Prepare inputs for friction force functions
        ffr_jacobian_input = self._get_ffr_jacobian_inputs(velocities, dedx)

        # Calculate friction forces on all four nodes
        ffr = self._compute_ffr(velocities, dedx)

        # Calculate the incomplete friction force gradients
        ffr_grad_s = self.ffr_jacobian_func(*ffr_jacobian_input).reshape((num_inputs, 6, 12))

        ffr_jacobian = self._chain_rule_friction_jacobian(ffr_grad_s, d2edx2)

        if self.friction:
            total_forces = dedx + ffr
            total_jacobian = d2edx2 + ffr_jacobian
        else:
            total_forces = dedx
            total_jacobian = d2edx2

        # Write gradient and hessian to shared memory location for DER
        self._py_to_cpp_hess(total_forces, total_jacobian, edge_ids)

        # Apply contact stiffness to gradient and hessian
        self.forces[:]  *= self.contact_stiffness
        self.hessian[:] *= (self.contact_stiffness * self.scale)  # Hessian must be multiplied by scale factor

    def _update_contact_stiffness(self, curr_cd, last_cd):
        diff = curr_cd - last_cd
        if curr_cd > self.contact_len + 0.005 and diff > 0:
            self.contact_stiffness *= 0.999
        elif diff < 0:
            if curr_cd < self.contact_len - 0.004:
                self.contact_stiffness *= 1.01
            elif curr_cd < self.contact_len - 0.002:
                self.contact_stiffness *= 1.005
            elif curr_cd < self.contact_len - 0.001:
                self.contact_stiffness *= 1.003
            elif curr_cd < self.contact_len:
                self.contact_stiffness *= 1.001

    def start_server(self):
        # Initialize ZMQ socket.
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://127.0.0.1:{}".format(self.port_no))
        print("Connected to python server")

        edge_ids = None
        velocities = None
        closest_distance = 0
        last_cd = 0

        while not socket.closed:
            # block until DER gives msg
            socket.recv()

            hessian = int(self.meta_data[5])
            first_iter = int(self.meta_data[0])
            self.friction = int(self.meta_data[1])
            # dt = self.meta_data[6]

            # Scale the nodal coordinates by scaling factor
            self.node_coordinates *= self.scale

            # Run collision detection algorithm and get edge ids at the start of every time step
            if first_iter:
                self.velocities *= self.scale
                self._construct_possible_edge_combos()
                edge_ids, closest_distance = self._detect_collisions()

            # Reset all gradient and hessian values to 0
            self.forces[:] = 0.
            if hessian: self.hessian[:] = 0.

            # Generate forces is contact is detected
            num_con = edge_ids.shape[0]
            if num_con != 0:
                # Obtain edge combinations and velocities
                if first_iter: velocities = self._prepare_velocities(edge_ids)
                edge_combos, closest_distance, f_out_vals = self._prepare_edges(edge_ids)

                # Increase/decrease contact stiffness depending on penetration severity
                if first_iter: self._update_contact_stiffness(closest_distance, last_cd)

                # Compute forces (and Hessian if condition is met)
                if not hessian:
                    self._get_forces(edge_combos, edge_ids, velocities, f_out_vals)
                else:
                    self._get_forces_and_hessian(edge_combos, edge_ids, velocities, f_out_vals)

            self.meta_data[4] = closest_distance / self.scale

            # Unblock DER
            socket.send(b'')

            # After each time step, print out summary information and update previous velocities
            if first_iter:
                last_cd = closest_distance
                print("time: {:.4f} | iters: {} | con: {:03d} | min_dist: {:.6f} | "
                      "k: {:.3e} | fric: {}".format(self.meta_data[2],
                                                    int(self.meta_data[3]),
                                                    num_con,
                                                    self.meta_data[4],
                                                    self.contact_stiffness,
                                                    self.friction))


def main():
    np.seterr(divide='ignore', invalid='ignore')

    # Simulation
    col_limit  = float(sys.argv[2])
    cont_stiff = float(sys.argv[3])
    ce_k       = float(sys.argv[4])
    mu_k       = float(sys.argv[5])
    radius     = float(sys.argv[6])
    num_nodes  = int(sys.argv[7])
    S          = float(sys.argv[8])

    params = {'num_nodes': num_nodes,
              'radius': radius,
              'collision_limit': col_limit,
              'contact_stiffness': cont_stiff,
              'ce_k': ce_k,
              'S': S,
              'mu_k': mu_k}

    contact_model = IMC(params)
    contact_model.start_server()


if __name__ == '__main__':
    main()
