import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dill as pickle
from imc_utils import min_dist_f_out_vectorized


class IMCEnergyPlotter:
    def __init__(self):
        ce_k = 50.0
        cekr = '_cek_' + str(ce_k) + '_h2_' + str(2.0)
        func_names = ['dd', 'first_grad', 'second_grad' + cekr]

        # Load pre-generated functions
        dir = './DER/grads_hessian_functions/'
        functions = []
        for name in func_names:
            with open(dir + name, 'rb') as f:
                functions.append(pickle.load(f))

        self.dd_grads          = functions[0]
        self.f_grad_funcs      = functions[1]
        self.s_grad_funcs      = functions[2]

    def fixbound(self, x, k):
        return (1 / k) * (np.log(1 + np.exp(k * x)) - np.log(1 + np.exp(k * (x - 1.0))))

    def boxcar(self, x, k):
        step1 = 1 / (1 + np.exp(-k*x))
        step2 = 1 / (1 + np.exp(-k*(x-1)))
        return step1 - step2

    def get_energy(self, edges, h2=2.0, ce_k=50.0, k=50.0):
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
        t = self.fixbound(t, k=k)
        u = (t * R - S2) / D2
        uf = self.fixbound(u, k=k)
        t = (1 - self.boxcar(u, k=k)) * self.fixbound(((uf * R + S1) / D1), k=k) + \
            self.boxcar(u, k=k) * t
        t = np.expand_dims(t, -1)
        uf = np.expand_dims(uf, -1)
        dist = np.sqrt(((d1 * t - d2 * uf - d12)**2).sum(axis=1))
        E = np.log(1 + np.exp(ce_k * (h2 - dist)))

        return E, dist

    def perfect_energy(self, edges, h2=2.0, ce_k=50.0):
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
        dist = np.sqrt(np.sum((d1 * t - d2 * u - d12) ** 2, axis=1))
        E = np.log(1 + np.exp(ce_k * (h2 - dist)))
        return E

    def plot_edge(self, edge, h, ax):
        edge = edge.reshape(6)
        p0, p1 = edge[:3], edge[3:]
        d = p1 - p0
        mag = np.linalg.norm(d)
        d = d / mag

        not_d = np.array([1.0, 0.0, 0.0])
        if (not_d == d).all():
            not_d = np.array([0.0, 1.0, 0.0])

        d1 = np.cross(d, not_d)
        d1 = d1 / np.linalg.norm(d1)
        d2 = np.cross(d, d1)

        t = np.linspace(0, mag, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        t, theta = np.meshgrid(t, theta)

        X, Y, Z = [p0[i] + d[i] * t + h * np.sin(theta) * d1[i] + h * np.cos(theta) * d2[i] for i in [0, 1, 2]]

        ax.plot_surface(X, Y, Z)
        ax.plot(*zip(p0, p1), color='red')
        ax.set_xlim(0., 5.)
        ax.set_ylim(0., 5.)
        ax.set_zlim(0., 5.)
        
    def chain_rule_dedx(self, f_grad_vals, s_grad_vals):
        num_inputs = s_grad_vals.shape[0]
        dedx = np.zeros((num_inputs, 12), dtype=np.float64)
        dedx[:] += s_grad_vals[:, :9] @ self.dd_grads
        s_grad_vals = s_grad_vals.reshape((num_inputs, 15, 1))
        for i in range(6):
            dedx[:] += s_grad_vals[:, i+9] * f_grad_vals[i]

        return dedx

    def plot_energy_barrier(self, h):

        base_edge1 = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        base_edge2 = np.array([4.0, 0.0, 0.0, 8.0, 0.0, 0.0])
        perp_edge = np.array([0.0, -2.0, 2.0, 0.0, 2.0, 2.0])
        para_edge = np.array([0.0, 0.0, 2.0, 4.0, 0.0, 2.0])

        delta = 0.15
        all_coords1 = []
        all_coords2 = []
        all_energies = []
        # Get all set of coordinates to try.
        for i, move_edge in enumerate([para_edge, perp_edge]):
            config = 'perpendicular' if i == 0 else 'horizontal'
            coords1, coords2 = [], []
            d = move_edge[3:] - move_edge[:3]
            dm = np.linalg.norm(d)
            if config == 'perpendicular':
                x_space = np.linspace(-1, 9, 150)
                y_space = np.linspace(-5, 1, 150)
            else:
                x_space = np.linspace(-5, 9, 150)
                y_space = np.linspace(-3, 3, 150)
            for x in x_space:
                curr_edge = move_edge.copy()
                curr_edge[0] = x
                curr_edge[3] = x + dm if config == 'horizontal' else x
                for y in y_space:
                    curr_edge[1] = y
                    curr_edge[4] = y + dm if config == 'perpendicular' else y
                    coords1.append([*base_edge1, *curr_edge])
                    coords2.append([*base_edge2, *curr_edge])

            coords1 = np.array(coords1)
            coords2 = np.array(coords2)

            energies1, dist1 = self.get_energy(coords1, h2=2*h)
            energies2, dist2 = self.get_energy(coords2, h2=2*h)

            energies = energies1 + energies2
            # energies = np.zeros_like(energies1)
            # for j in range(energies.shape[0]):
            #     limit = 2*h + delta
            #     if dist1[j] < limit and dist2[j] < limit:
            #         energies[j] += 0.5 * (energies1[j] + energies2[j])
            #     else:
            #         energies[j] += energies1[j] + energies2[j]

            all_coords1.append(coords1)
            all_coords2.append(coords2)
            all_energies.append(energies)

        fig = plt.figure(figsize=(10, 3), dpi=300)

        # Plot based off the midpoint of the edge
        midpoints = all_coords1[0][:, 6:9] + np.array([0., 2., 0.])
        X = midpoints[:, 0]
        Y = midpoints[:, 1]

        # ax1 = fig.add_subplot(221)
        ax1 = fig.add_subplot(121)
        ax1.plot([0, 8], [0, 0], '--', color='blue')
        ax1.plot([0, 8], [1, 1], color='blue')
        ax1.plot([0, 8], [-1, -1], color='blue')
        ax1.plot([0, 0], [1, -1], color='blue')
        ax1.plot([4, 4], [1, -1], color='blue')
        ax1.plot([8, 8], [1, -1], color='blue')
        ax1.scatter(X, Y, c=all_energies[0], s=0.75)
        ax1.scatter([0, 4, 8], [0, 0, 0], color='blue', s=10)
        ax1.set_title('Energy for perpendicular rod at $z=2h$', size=8)

        midpoints = all_coords1[1][:, 6:9] + np.array([2., 0., 0.])
        X = midpoints[:, 0]
        Y = midpoints[:, 1]

        # ax2 = fig.add_subplot(222)
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 8], [0, 0], '--', color='blue')
        ax2.plot([0, 8], [1, 1], color='blue')
        ax2.plot([0, 8], [-1, -1], color='blue')
        ax2.plot([0, 0], [1, -1], color='blue')
        ax2.plot([4, 4], [1, -1], color='blue')
        ax2.plot([8, 8], [1, -1], color='blue')
        ax2.scatter(X, Y, c=all_energies[1], s=0.75)
        ax2.scatter([0, 4, 8], [0, 0, 0], color='blue', s=10)
        ax2.set_title('Energy for parallel rod at $z=2h$', size=8)

        # # Force magnitudes perpendicular
        # _, f_out_vals1 = min_dist_f_out_vectorized(all_coords1[0])
        # _, f_out_vals2 = min_dist_f_out_vectorized(all_coords2[0])
        #
        # f_grad_vals1 = np.array([f_grad(*all_coords1[0]) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()
        # s_grad_vals1 = np.array([s_grad(*f_out_vals1) for s_grad in self.s_grad_funcs], dtype=np.float64).T
        # f_grad_vals2 = np.array([f_grad(*all_coords2[0]) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()
        # s_grad_vals2 = np.array([s_grad(*f_out_vals2) for s_grad in self.s_grad_funcs], dtype=np.float64).T
        #
        # if len(all_coords1[0]) == 0:
        #     s_grad_vals1 = s_grad_vals1.reshape((1, 15))
        #     f_grad_vals1 = f_grad_vals1.reshape((6, 1, 12))
        #     s_grad_vals2 = s_grad_vals2.reshape((1, 15))
        #     f_grad_vals2 = f_grad_vals2.reshape((6, 1, 12))
        #
        # dedx1 = self.chain_rule_dedx(f_grad_vals1, s_grad_vals1)
        # dedx2 = self.chain_rule_dedx(f_grad_vals2, s_grad_vals2)
        #
        # ax3 = fig.add_subplot(223)
        # midpoints = all_coords1[0][:, 6:9] + np.array([0., 2., 0.])
        # X = midpoints[:, 0]
        # Y = midpoints[:, 1]
        # perp_forces = np.zeros((dedx1.shape[0], 3))
        # perp_forces[:] += dedx1[:, :3] + dedx1[:, 3:6] + dedx2[:, :3] + dedx2[:, 3:6]
        # perp_forces = np.sqrt((perp_forces**2).sum(axis=1))
        #
        # ax3.scatter(X, Y, c=perp_forces, s=0.5)
        # ax3.set_title('Perpendicular configuration force magnitudes', size=8)
        #
        # # Force magnitudes parallel
        # _, f_out_vals1 = min_dist_f_out_vectorized(all_coords1[1])
        # _, f_out_vals2 = min_dist_f_out_vectorized(all_coords2[1])
        #
        # f_grad_vals1 = np.array([f_grad(*all_coords1[1]) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()
        # s_grad_vals1 = np.array([s_grad(*f_out_vals1) for s_grad in self.s_grad_funcs], dtype=np.float64).T
        # f_grad_vals2 = np.array([f_grad(*all_coords2[1]) for f_grad in self.f_grad_funcs], dtype=np.float64).squeeze()
        # s_grad_vals2 = np.array([s_grad(*f_out_vals2) for s_grad in self.s_grad_funcs], dtype=np.float64).T
        #
        # if len(all_coords1[1]) == 0:
        #     s_grad_vals1 = s_grad_vals1.reshape((1, 15))
        #     f_grad_vals1 = f_grad_vals1.reshape((6, 1, 12))
        #     s_grad_vals2 = s_grad_vals2.reshape((1, 15))
        #     f_grad_vals2 = f_grad_vals2.reshape((6, 1, 12))
        #
        # dedx1 = self.chain_rule_dedx(f_grad_vals1, s_grad_vals1)
        # dedx2 = self.chain_rule_dedx(f_grad_vals2, s_grad_vals2)
        #
        # ax4 = fig.add_subplot(224)
        # midpoints = all_coords1[1][:, 6:9] + np.array([2., 0., 0.])
        # X = midpoints[:, 0]
        # Y = midpoints[:, 1]
        # para_forces = np.zeros((dedx1.shape[0], 3))
        # para_forces[:] += dedx1[:, :3] + dedx1[:, 3:6] + dedx2[:, :3] + dedx2[:, 3:6]
        # para_forces = np.sqrt((para_forces**2).sum(axis=1))
        #
        # ax4.scatter(X, Y, c=para_forces, s=0.5)
        # ax4.set_title('Parallel configuration force magnitudes', size=8)
        #
        plt.savefig("energy_boundaries.png", bbox_inches='tight', pad_inches=0.03)


def main():
    plotter = IMCEnergyPlotter()

    edge1 = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    edge2 = np.array([0.0, -2.0, 2.0, 0.0, 2.0, 2.0])

    # plotter.plot_edge(edge1, h=1.0)

    plotter.plot_energy_barrier(h=1.0)


if __name__ == '__main__':
    main()
