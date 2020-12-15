import sys
import numpy as np
import matplotlib.pyplot as plt


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plot_pull_force(expR, expT, labels, h=0.0016, moving_avg=False):

    fig = plt.figure(figsize=(4, 3), dpi=600)

    # EI = 1.41372e-06
    # EI = 1.41372e-07
    EI = 9.26493e-07

    # for i in range(len(expR)):
    for i in [0]:
        curr_label = labels[i]
        exp_x = np.sqrt(h / expR[i])
        exp_t = (expT[i] * h ** 2) / EI
        indices = np.where(exp_x > 0.120)
        exp_x = exp_x[indices]
        exp_t = exp_t[indices]
        indices = np.where(exp_x < 0.23)
        exp_x = exp_x[indices]
        exp_t = exp_t[indices]

        exp_t = movingaverage(exp_t[indices], 50) if moving_avg else exp_t

        plt.plot(exp_x,
                 exp_t,
                 label=curr_label, zorder=0, linewidth=0.75)

    # Theoretical Curves
    e = np.linspace(0.120, 0.23, 1000)
    # e = np.linspace(0.06, 0.18, 1000)

    # theoretical w/o friction
    y = e**4 / 2
    # plt.plot(e, y, label='theoretical w/o friction', zorder=10)

    # theoretical w/ friction
    friction = 0.1 * 0.492 * e**3
    y1 = y + friction
    y2 = y - friction
    e = np.append(e, np.flip(e))
    y_t = np.append(y1, np.flip(y2))
    plt.plot(e, y_t, label='$n=1$, theoretical', zorder=10, linewidth=0.5)

    for i in [1,2,3]:
        curr_label = labels[i]
        exp_x = np.sqrt(h / expR[i])
        exp_t = (expT[i] * h ** 2) / EI
        indices = np.where(np.logical_and(exp_x > 0.125, exp_x < 0.23))
        exp_x = exp_x[indices]
        exp_t = exp_t[indices]

        # exp_t = movingaverage(exp_t[indices], 50) if moving_avg else exp_t
        exp_t = movingaverage(exp_t, 50) if moving_avg else exp_t

        plt.plot(exp_x,
                 exp_t,
                 label=curr_label, zorder=0, linewidth=0.5)

    plt.xlabel('$\sqrt{h/R}$')
    plt.ylabel('$Fh^2 / EI$')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.yscale('log')

    # plt.title('Tightening and loosening pull force, $N=1$', size=10, **arial)
    plt.savefig("pull_release.png", bbox_inches='tight', pad_inches=0.03, dpi=600)


def plot_end_to_end(expT, eTe, labels, moving_avg=False):

    fig = plt.figure(figsize=(4, 3), dpi=600)
    # fig = plt.figure(dpi=300)

    # limits = [710, 650, 625, 570, 710, 650, 650, 620]
    limits = [590] * 8
    # limits = [710, 640, 615, 570] * 2
    # limits = [620, 595, 595]

    # colors = ['C0', 'C3']
    for i in range(len(expT)):
        # color = colors[0] if i < 4 else colors[1]
        # curr_label = labels[i]
        end_to_end = (1.0 - eTe[i]) * 1000
        indices = np.where(end_to_end < limits[i])
        F = movingaverage(expT[i], 50) if moving_avg else expT[i]
        end_to_end = end_to_end[indices]
        F = F[indices]
        plt.plot(end_to_end,
                 F,
                 label=labels[i], zorder=0, linewidth=0.75)
        # if i == 0:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              label='SPT', zorder=0, linewidth=0.75)
        # elif i == 4:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              label='IMC', zorder=0, linewidth=0.75)
        # else:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              zorder=0, linewidth=0.75)

    plt.xlabel('End-to-end shortening [mm]')
    plt.ylabel('Force, $F$ [N]')
    plt.legend(loc='lower left', prop={'size': 6})
    plt.yscale('log')
    # plt.ylim([3.1 * 10**-3, 1.1 * 10**-1])

    # plt.title('Implicit contact pull forces, $\mu_k=0.10$, $dt=10$ ms', size=14)
    plt.savefig("friction.png", bbox_inches='tight', pad_inches=0.03)
    
def plot_end_to_end_with_normalized_force(expT, eTe, labels, moving_avg=False):

    fig = plt.figure(figsize=(4, 3), dpi=600)
    EI = 9.26493e-07
    h = 0.0016
    limits = [590] * 8
    # limits = [710, 650, 625, 570, 710, 650, 650, 620]
    # limits = [710, 640, 615, 570] * 2
    # limits = [620, 595, 595]

    colors = ['C0', 'C3']
    for i in range(len(expT)):
        color = colors[0] if i < 4 else colors[1]
        # curr_label = labels[i]
        end_to_end = (1.0 - eTe[i]) * 1000
        indices = np.where(end_to_end < limits[i])
        F = movingaverage(expT[i], 50) if moving_avg else expT[i]
        F = F * h ** 2 / EI
        end_to_end = end_to_end[indices]
        F = F[indices]
        plt.plot(end_to_end,
                 F,
                 label=labels[i], zorder=0, linewidth=0.75)
        # if i == 0:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              label='SPT', zorder=0, linewidth=0.75)
        # elif i == 4:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              label='IMC', zorder=0, linewidth=0.75)
        # else:
        #     plt.plot(end_to_end,
        #              F,
        #              color,
        #              zorder=0, linewidth=0.75)


    plt.xlabel('End-to-end shortening [mm]')
    plt.ylabel('$Fh^2 / EI$')
    plt.legend(loc='lower left', prop={'size': 6})
    plt.yscale('log')
    # plt.ylim([3.1 * 10**-3, 1.1 * 10**-1])

    # plt.title('Implicit contact pull forces, $\mu_k=0.10$, $dt=10$ ms', size=14)
    plt.savefig("friction.png", bbox_inches='tight', pad_inches=0.03)



def plot_normalized_force(expR, expT, eTe, labels):

    fig = plt.figure(figsize=(4, 3))

    E = 0.18e6
    # B = E pi h^4 / 4
    B = E * np.pi * (0.001 ** 4) / 4

    for i in range(len(expT)):
        curr_label = labels[i]
        eTeS = (1.0 - eTe[i]) * 1000
        i_at_e150mm = np.argmin(np.abs(eTeS - 150))
        i_at_e500mm = np.argmin(np.abs(eTeS - 500))
        R150mm = expR[i][i_at_e150mm]
        R500mm = expR[i][i_at_e500mm]
        nF150mm = expT[i][i_at_e150mm] * (R150mm ** 2) / B
        nF500mm = expT[i][i_at_e500mm] * (R500mm ** 2) / B
        plt.plot(i+1,  # unknotting number
                 nF150mm,
                 label='e at 150mm ' + curr_label, zorder=0, marker="s")
        plt.plot(i+1,  # unknotting number
                 nF500mm,
                 label='e at 500mm ' + curr_label, zorder=0, marker="^")

    plt.xticks(np.arange(0, 4))
    plt.xlabel('unknotting number N', size=12)
    plt.ylabel('FR^2/B')
    plt.title('Pull Force, u=0.01, mu=0.1')
    plt.legend(loc='upper left')


def plot_iterations(eTe, iters, comp_time, labels):
    fig, ax1 = plt.subplots(figsize=(4, 3), dpi=600)
    for i in range(len(eTe)):
        curr_iters = movingaverage(iters[i], 300)
        ax1.plot((1.0 - eTe[i]) * 1000, curr_iters, label=labels[i], linewidth=1)
        # curr_comp_time = movingaverage(comp_time[i], 1500)
        # ax1.plot(time[i], curr_comp_time, label=labels[i], linewidth=0.75)

    # ax2 = ax1.twinx()
    # for i in range(len(time)):
    #     curr_comp_time = movingaverage(comp_time[i], 1500)
    #     ax2.scatter(time[i], curr_comp_time, '^', label=labels[i])

    ax1.set_xlabel('R [mm]')
    ax1.set_ylabel('iterations to convergence')
    # ax2.set_ylabel('total computation time per time step [s]')
    # ax2.set_ylim(0.01, 0.115)  # SPT
    ax1.legend(loc='upper left', prop={'size': 7})

    # plt.title('SPT computation and convergence', size=14)
    plt.savefig("iterations.png", bbox_inches='tight', pad_inches=0.03)


def plot_comp_time(time, iters, comp_time, labels):
    fig = plt.figure(figsize=(4, 3))
    for i in range(len(time)):
        curr_comp_time = movingaverage(comp_time[i], 1000)
        plt.plot(time[i], curr_comp_time, label=labels[i])

    plt.xlabel('pull time [s]', size=12)
    plt.ylabel('total computation time [s]', size=12)
    plt.legend(loc='upper left', prop={'size': 12})

    plt.title('Computation time (SPT)', size=14)
    plt.savefig("comp_time.png", bbox_inches='tight')


def plot_penetration(time, minD):
    fig = plt.figure(figsize=(4, 3), dpi=300)
    for i in range(len(time)):
        plt.plot(time[i], minD[i])


def main():
    # sys.argv.append('imc_n1_vrel.txt')
    # sys.argv.append('imc_n1_vrel_scaledenergy.txt')
    # sys.argv.append('imc_n1_final.txt')
    # sys.argv.append('imc_n2_vrel_scaledenergy.txt')
    # sys.argv.append('imc_n2_final.txt')

    sys.argv.append('imc_n1_ps1cm.txt')
    sys.argv.append('imc_n2_ps1cm.txt')
    sys.argv.append('imc_n3_ps1cm.txt')
    sys.argv.append('imc_n4_ps1cm.txt')
    sys.argv.append('imc_n1_final.txt')
    sys.argv.append('imc_n2_final.txt')
    sys.argv.append('imc_n3_final.txt')
    sys.argv.append('imc_n4_final.txt')
    #
    assert len(sys.argv) >= 2, 'File name was not supplied'
    num_plots = len(sys.argv) - 1
    time, expR, expT, eTe, iters, total_iters, minD, comp_time = [], [], [], [], [], [], [], []
    for i in range(1, num_plots+1):
        filename = sys.argv[i]
        data = np.loadtxt(fname=filename)
        time.append(data[:, 0])
        expT.append((data[:, 1] + data[:, 2]) / 2)
        expR.append(data[:, 3])
        eTe.append(data[:, 4])
        iters.append(data[:, 5])
        total_iters.append(data[:, 6])
        minD.append(data[:, 7])
        comp_time.append(data[:, 8])

    # labels = ['$n=1$, IMC', '$n=2, u=9$ mm/s', '$n=2, u=3$ mm/s', '$n=2, u=1$ mm/s'] * 5
    # labels = ['$\mu_k = 0.1$', '$\mu_k = 0.3$', '$\mu_k = 0.5$']
    labels = ['vrel', 'vrel & energy', 'original']
    labels = ['n=1', 'n=2', 'n=3', 'n=4'] * 2

    plot_iterations(eTe, iters, comp_time, labels)
    # plot_comp_time(time, iters, comp_time, labels)
    # plot_pull_force(expR, expT, labels, h=0.0016, moving_avg=False)
    # plot_end_to_end_with_normalized_force(expT, eTe, labels, moving_avg=False)
    # plot_normalized_force(expR, expT, eTe, labels)
    # plot_penetration(time, minD)

    plt.show()


if __name__ == '__main__':
    main()
