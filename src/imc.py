import sys
import mmap
import zmq
import posix_ipc
import numpy as np
import dill as pickle
import imc_utils as iu


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
        self.dt = None

        self.contact_len = self.radius * 2
        ce_k = params['ce_k']
        ce_params = '_cek_' + str(ce_k) + '_h2_' + str(self.contact_len)
        func_names = ['ce_grad' + ce_params,
                      'ce_hess' + ce_params,
                      'fr_jaco']

        # Load pre-generated functions
        dir = './grads_hessian_functions/'
        functions = []
        for name in func_names:
            with open(dir + name, 'rb') as f:
                functions.append(pickle.load(f))

        self.ce_grad_func = functions[0]
        self.ce_hess_func = functions[1]
        self.fr_jaco_func = functions[2]

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
        p = posix_ipc.SharedMemory('prev_node_coordinates' + self.port_no, size=node_bytes, read_only=False)
        f = posix_ipc.SharedMemory('contact_forces' + self.port_no, size=node_bytes, read_only=False)
        h = posix_ipc.SharedMemory('contact_hessian' + self.port_no, size=hess_bytes, read_only=False)
        m = posix_ipc.SharedMemory('meta_data' + self.port_no, size=meta_bytes, read_only=False)
        self.node_coordinates = np.ndarray(nv, np.float64, mmap.mmap(n.fd, 0))
        self.prev_node_coordinates = np.ndarray(nv, np.float64, mmap.mmap(p.fd, 0))
        self.forces = np.ndarray(nv, np.float64, mmap.mmap(f.fd, 0))
        self.hessian = np.ndarray(h_size, np.float64, mmap.mmap(h.fd, 0))
        self.meta_data = np.ndarray(meta_data_size, np.float64, mmap.mmap(m.fd, 0))
        assert self.node_coordinates.flags['C_CONTIGUOUS'] is True
        assert self.prev_node_coordinates.flags['C_CONTIGUOUS'] is True
        assert self.forces.flags['C_CONTIGUOUS'] is True
        assert self.hessian.flags['C_CONTIGUOUS'] is True
        assert self.meta_data.flags['C_CONTIGUOUS'] is True

    def _construct_possible_edge_combos(self):
        return iu.construct_possible_edge_combos(self.edges, self.edge_combos, self.node_coordinates, self.ia, self.scale)

    def _detect_collisions(self):
        return iu.detect_collisions(self.edge_combos, self.edge_ids, self.collision_limit, self.contact_len)

    def _prepare_edges(self, edge_ids, edge_combos, first_iter):
        return iu.prepare_edges(edge_ids, edge_combos, first_iter, self.node_coordinates, self.prev_node_coordinates)

    def _compute_friction(self, data, contact_forces):
        return iu.compute_friction(data, contact_forces, self.mu_k, self.dt)

    def _get_friction_jacobian_inputs(self, data, contact_forces):
        return iu.get_friction_jacobian_inputs(data, contact_forces, self.mu_k, self.dt)

    @staticmethod
    def _chain_rule_friction_jacobian(dfr_dfc, dfc_dx):
        return iu.chain_rule_friction_jacobian(dfr_dfc, dfc_dx)

    def _py_to_cpp_nohess(self, py_forces, edge_ids):
        iu.py_to_cpp_nohess(py_forces, self.forces, edge_ids)

    def _py_to_cpp_hess(self, py_forces, py_jacobian, edge_ids):
        iu.py_to_cpp_hess(py_forces, py_jacobian, self.forces, self.hessian, edge_ids)

    def _get_forces(self, edges, edge_ids):
        num_inputs = edges.shape[0]

        contact_force_input = edges[:, :12] * self.scale

        # Compute contact energy gradient (contact forces)
        de_dx = np.array(self.ce_grad_func(*contact_force_input), dtype=np.float64).reshape((num_inputs, 12))

        if self.friction:
            # Compute friction forces
            ffr = self._compute_friction(edges, de_dx)
            total_forces = de_dx + ffr
        else:
            total_forces = de_dx

        # Write gradient to shared memory location for DER
        self._py_to_cpp_nohess(total_forces, edge_ids)

        # Apply contact stiffness to gradient
        self.forces *= self.contact_stiffness

    def _get_forces_and_hessian(self, edges, edge_ids):
        num_inputs = edges.shape[0]

        contact_force_input = edges[:, :12] * self.scale

        # Compute contact energy gradient (contact forces)
        de_dx = np.array(self.ce_grad_func(*contact_force_input), dtype=np.float64).reshape((num_inputs, 12))

        # Compute contact energy Hessian (contact force Jacobian)
        d2e_dx2 = np.array(self.ce_hess_func(*contact_force_input), dtype=np.float64).reshape((num_inputs, 12, 12))

        # Apply contact stiffness to gradient and Hessian
        de_dx *= self.contact_stiffness
        # Note that for the Hessian we apply the scale factor here.
        # Technically, the forces need to be multiplied by S while the
        # Hessian is multiplied by S^2 (due to chain rule), but we
        # implicitly factor in a scale factor into the contact stiffness
        # to prevent the stiffness from being too low of a value.
        d2e_dx2 *= self.contact_stiffness * self.scale
        if self.friction:
            # Compute friction forces
            ffr = self._compute_friction(edges, de_dx)

            # Prepare inputs for friction force Jacobian function
            ffr_jacobian_input = self._get_friction_jacobian_inputs(edges, de_dx)

            # Compute partial friction force Jacobian
            dfr_dfc = self.fr_jaco_func(*ffr_jacobian_input)
            np.nan_to_num(dfr_dfc, copy=False)  # check to see if this is still necessary

            # Obtain friction force Jacobian using chain rule
            dfr_dx = self._chain_rule_friction_jacobian(dfr_dfc, d2e_dx2)

            np.nan_to_num(ffr, copy=False)  # check to see if this is still necessary
            total_forces = de_dx + ffr
            total_jacobian = d2e_dx2 + dfr_dx
        else:
            total_forces = de_dx
            total_jacobian = d2e_dx2

        # Write gradient and Hessian to shared memory location for DER
        self._py_to_cpp_hess(total_forces, total_jacobian, edge_ids)

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
        closest_distance = 0
        last_cd = 0

        while not socket.closed:
            # block until DER gives msg
            socket.recv()

            hessian = int(self.meta_data[5])
            first_iter = int(self.meta_data[0])
            self.friction = int(self.meta_data[1])
            self.dt = self.meta_data[6]

            # Run collision detection algorithm and get edge ids at the start of every time step
            if first_iter:
                self._construct_possible_edge_combos()
                edge_ids, scaled_closest_distance = self._detect_collisions()
                closest_distance = scaled_closest_distance / self.scale

                # Reuse edge combos by writing the previous time step node coordinates just once
                num_con = edge_ids.shape[0]
                edge_combos = np.zeros((num_con, 24), dtype=np.float64)

            # Reset all gradient and Hessian values to 0
            self.forces[:] = 0.
            if hessian: self.hessian[:] = 0.

            # Generate forces is contact is detected
            if num_con != 0:
                self._prepare_edges(edge_ids, edge_combos, first_iter)

                # Increase/decrease contact stiffness depending on penetration severity
                if first_iter:
                    self._update_contact_stiffness(scaled_closest_distance, last_cd)

                # Compute forces (and Hessian if condition is met)
                if not hessian:
                    self._get_forces(edge_combos, edge_ids)
                else:
                    self._get_forces_and_hessian(edge_combos, edge_ids)

            # Unblock DER
            socket.send(b'')

            # After each time step, print out summary information and update previous velocities
            if first_iter:
                last_cd = scaled_closest_distance
                print("time: {:.4f} | iters: {} | con: {:03d} | min_dist: {:.6f} | "
                      "k: {:.3e} | fric: {}".format(self.meta_data[2],
                                                    int(self.meta_data[3]),
                                                    num_con,
                                                    closest_distance,
                                                    self.contact_stiffness,
                                                    self.friction))


def main():
    np.seterr(divide='ignore', invalid='ignore')

    # Simulation params
    params = {'collision_limit': float(sys.argv[2]),
              'contact_stiffness': float(sys.argv[3]),
              'ce_k': float(sys.argv[4]),
              'mu_k': float(sys.argv[5]),
              'radius': float(sys.argv[6]),
              'num_nodes': int(sys.argv[7]),
              'S': float(sys.argv[8])}

    contact_model = IMC(params)
    contact_model.start_server()


if __name__ == '__main__':
    main()
