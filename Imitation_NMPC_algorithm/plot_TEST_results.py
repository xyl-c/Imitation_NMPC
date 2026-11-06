import numpy as np
from matplotlib import pyplot as plt
from sympy import plot_backends
import matplotlib as mpl
import torch
from quaModel_casadi import QuadrotorModel
from mpcSolver_casadi import *
from multiprocessing import Pool, cpu_count
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import qmc, truncnorm



SET2 = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
]

markers = ['o', 's', '^', 'D']


def eul2qua(scaled_sample):
    qw = np.cos(scaled_sample[:, 3] / 2) * np.cos(scaled_sample[:, 4] / 2) * np.cos(scaled_sample[:, 5] / 2) + np.sin(
        scaled_sample[:, 3] / 2) * np.sin(scaled_sample[:, 4] / 2) * np.sin(scaled_sample[:, 5] / 2)
    qx = np.sin(scaled_sample[:, 3] / 2) * np.cos(scaled_sample[:, 4] / 2) * np.cos(scaled_sample[:, 5] / 2) - np.cos(
        scaled_sample[:, 3] / 2) * np.sin(scaled_sample[:, 4] / 2) * np.sin(scaled_sample[:, 5] / 2)
    qy = np.cos(scaled_sample[:, 3] / 2) * np.sin(scaled_sample[:, 4] / 2) * np.cos(scaled_sample[:, 5] / 2) + np.sin(
        scaled_sample[:, 3] / 2) * np.cos(scaled_sample[:, 4] / 2) * np.sin(scaled_sample[:, 5] / 2)
    qz = np.cos(scaled_sample[:, 3] / 2) * np.cos(scaled_sample[:, 4] / 2) * np.sin(scaled_sample[:, 5] / 2) - np.sin(
        scaled_sample[:, 3] / 2) * np.sin(scaled_sample[:, 4] / 2) * np.cos(scaled_sample[:, 5] / 2)
    q = np.concatenate([qw.reshape(-1, 1), qx.reshape(-1, 1), qy.reshape(-1, 1), qz.reshape(-1, 1)], axis=1)

    return q

def plot_ver_mse():

    mean_save = np.zeros((10,4))
    std_save = np.zeros((10,4))
    for j in range(10):
        ver_save = '../TEST_results/ver_errors{}.csv' .format(j)
        ver_save = np.loadtxt(ver_save, delimiter=',', dtype=np.float32)

        mean = np.mean((ver_save),axis=0)
        std = np.std((ver_save),axis=0)

        mean_save[j,:] = mean
        std_save[j,:] = std

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stix",
        "font.size": 30,
    })

    x = np.linspace(1,10,10)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x, mean_save[:,0],
            label=r"$\tilde{T}$",
            color=SET2[0],
            marker=markers[0],
            linewidth=2,
            markersize=6)

    ax.fill_between(x,
                    mean_save[:,0] - std_save[:,0],
                    mean_save[:,0] + std_save[:,0],
                    color=SET2[0],
                    alpha=0.2)

    ax.set_xlabel('Index of test trials')
    ax.set_ylabel('Thrust test error [N]')
    ax.set_ylim(-3, 5)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.legend(loc='upper right', framealpha=0.9,fontsize=30)
    ax.grid(True, alpha=0.6, linestyle="-", linewidth=2.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.2)
    plt.tight_layout()

    plt.savefig('./figs_vis/ver_res_T.png', dpi=850, bbox_inches="tight")
    plt.show()

def plot_err_std():
    mpc_save = '../multi_plot_results/mpc_save_3.csv'
    mpc_save = np.loadtxt(mpc_save, delimiter=',', dtype=np.float32)
    ref_save = '../multi_plot_results/ref_save_3.csv'
    ref_save = np.loadtxt(ref_save, delimiter=',', dtype=np.float32)
    nn_save = '../multi_plot_results/nn_save_3.csv'
    nn_save = np.loadtxt(nn_save, delimiter=',', dtype=np.float32)

    mean_err = np.zeros(100)
    std_err = np.zeros(100)
    for j in range(100):
        nn = nn_save[0 + 12 * j:3 + 12 * j, :]
        mpc = mpc_save[0 + 12 * j:3 + 12 * j, :]
        psi_nn = nn_save[5 + 12 * j:6 + 12 * j, :]
        psi_mpc = mpc_save[5 + 12 * j:6 + 12 * j, :]
        y_mpc = np.concatenate((mpc, psi_mpc), axis=0)
        y_nn = np.concatenate((nn, psi_nn), axis=0)

        err = np.linalg.norm(y_nn - y_mpc, axis=0)
        mean = np.mean(err)
        std = np.std(err)

        mean_err[j] = mean
        std_err[j] = std
    mean_err_1 = mean_err[0:25]
    std_err_1 = std_err[0:25]
    mean_err_2 = mean_err[25:50]
    std_err_2 = std_err[25:50]
    mean_err_3 = mean_err[50:75]
    std_err_3 = std_err[50:75]
    mean_err_4 = mean_err[75:100]
    std_err_4 = std_err[75:100]
    x = np.linspace(1,25,25)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stix",
        "font.size": 16,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, mean_err_1,
            label='Circular trajectories',
            color=SET2[0],
            marker=markers[0],
            linewidth=2,
            markersize=6)

    ax.fill_between(x,
                    mean_err_1 - std_err_1,
                    mean_err_1 + std_err_1,
                    color=SET2[0],
                    alpha=0.6)

    ax.plot(x, mean_err_2,
            label='Helical trajectories',
            color=SET2[1],
            marker=markers[1],
            linewidth=2,
            markersize=6)

    ax.fill_between(x,
                    mean_err_2 - std_err_2,
                    mean_err_2 + std_err_2,
                    color=SET2[1],
                    alpha=0.2)

    ax.plot(x, mean_err_3,
            label='Straight-line trajectories',
            color=SET2[2],
            marker=markers[2],
            linewidth=2,
            markersize=6)

    ax.fill_between(x,
                    mean_err_3 - std_err_3,
                    mean_err_3 + std_err_3,
                    color=SET2[2],
                    alpha=0.2)

    ax.plot(x, mean_err_4,
            label='Figure-eight trajectories',
            color=SET2[3],
            marker=markers[3],
            linewidth=2,
            markersize=6)

    ax.fill_between(x,
                    mean_err_4 - std_err_4,
                    mean_err_4 + std_err_4,
                    color=SET2[3],
                    alpha=0.3)


    ax.set_xlabel('Index of trajectory')
    ax.set_ylabel('Imitation error')

    ax.set_xticks([1, 5, 10, 15, 20, 25])

    ax.legend(loc='upper right', framealpha=0.8)

    ax.grid(True, alpha=0.6, linestyle="-", linewidth=2.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.2)

    plt.tight_layout()

    plt.savefig('./figs_vis/err_std.png', dpi=800, bbox_inches="tight")
    plt.show()

def plot_box():

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stix",
        "font.size": 16,
    })

    time_save_cpu = '../multi_plot_results/time_save_cpu_1.csv'
    time_save_cpu = np.loadtxt(time_save_cpu, delimiter=',', dtype=np.float32)
    time_save_gpu = '../multi_plot_results/time_save_gpu_1.csv'
    time_save_gpu = np.loadtxt(time_save_gpu, delimiter=',', dtype=np.float32)
    time_save_numpy = '../multi_plot_results/time_save_numpy_1.csv'
    time_save_numpy = np.loadtxt(time_save_numpy, delimiter=',', dtype=np.float32)
    time_nmpc = np.zeros((100,time_save_gpu.shape[1]))
    time_gpu = np.zeros((100, time_save_gpu.shape[1]))
    time_cpu = np.zeros((100, time_save_gpu.shape[1]))
    time_numpy = np.zeros((100, time_save_gpu.shape[1]))
    for i in range(100):
        time_nmpc[i, :] = time_save_gpu[0 + 2*i, :]
        time_gpu[i, :] = time_save_gpu[1 + 2 * i, :]
        time_cpu[i, :] = time_save_cpu[1 + 2 * i, :]
        time_numpy[i, :] = time_save_numpy[1 + 2 * i, :]
    time_nmpc = np.mean(time_nmpc, axis=0)*1000.0
    time_gpu = np.mean(time_gpu, axis=0)*1000.0
    time_cpu = np.mean(time_cpu, axis=0)*1000.0
    time_numpy = np.mean(time_numpy, axis=0)*1000.0
    print(time_cpu.shape)


    data = [time_nmpc, time_gpu, time_cpu, time_numpy]
    x_labels = ['NMPC', 'IMPC using\n Pytorch on GPU', 'IMPC using\n Pytorch on CPU', 'IMPC using\n NumPy']
    colors = [SET2[4], SET2[5], SET2[6], SET2[7]]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_plot = ax.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(
            linewidth=3,
            color='black',
            facecolor='lightblue'
        ),
        medianprops=dict(linewidth=2, color='red'),
        whiskerprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=2, color='black'),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='gray')
    )



    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Computation time [ms]')
    ax.set_xticklabels(x_labels)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.2)


    ax.grid(True, alpha=0.6, linestyle="-", linewidth=2.5)

    zoom_axes = []
    zoom_positions = [
        [0.18, 0.30, 0.55, 0.55],  # [left, bottom, width, height]
        [0.281, 0.30, 0.55, 0.55],
        [0.531, 0.30, 0.55, 0.55],
        [0.781, 0.30, 0.55, 0.55],
    ]

    q1 = np.percentile(data[1], 0.001)
    q3 = np.percentile(data[1], 90)
    iqr = q3 - q1
    lower_bound = q1 - 0.8 * iqr
    upper_bound = q3 + 0.8 * iqr

    ax_inset = inset_axes(ax, width='30%', height='90%',
                          loc='upper left',
                          bbox_to_anchor=zoom_positions[1],
                          bbox_transform=ax.transAxes)

    bp_inset = ax_inset.boxplot(
        data[1], widths=0.6,
        patch_artist=True,
        boxprops=dict(
            linewidth=2,
            color='black',
            facecolor='lightblue'
        ),
        medianprops=dict(linewidth=2, color='red'),
        whiskerprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=2, color='black'),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='gray')
    )
    bp_inset['boxes'][0].set_facecolor(colors[1])

    y_min = min(data[1].min(), lower_bound - 0.0001)
    y_max = min(data[1].max(), upper_bound + 0.03)
    ax_inset.set_ylim(y_min, y_max)
    ax_inset.set_xticklabels([''])


    q1 = np.percentile(data[2], 0.001)
    q3 = np.percentile(data[2], 90)
    iqr = q3 - q1
    lower_bound = q1 - 1.2 * iqr
    upper_bound = q3 + 0.8 * iqr

    ax_inset = inset_axes(ax, width='30%', height='90%',
                          loc='upper left',
                          bbox_to_anchor=zoom_positions[2],
                          bbox_transform=ax.transAxes)

    bp_inset = ax_inset.boxplot(
        data[2], widths=0.6,
        patch_artist=True,
        boxprops=dict(
            linewidth=2,
            color='black',
            facecolor='lightblue'
        ),
        medianprops=dict(linewidth=2, color='red'),
        whiskerprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=2, color='black'),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='gray')
    )
    bp_inset['boxes'][0].set_facecolor(colors[2])

    y_min = min(data[2].min(), lower_bound - 0.0001)
    y_max = min(data[2].max(), upper_bound + 0.03)
    ax_inset.set_ylim(y_min, y_max)
    ax_inset.set_xticklabels([''])


    q1 = np.percentile(data[3], 0.1)
    q3 = np.percentile(data[3], 90)
    iqr = q3 - q1
    lower_bound = q1 - 0.8 * iqr
    upper_bound = q3 + 0.8 * iqr
    ax_inset = inset_axes(ax, width='30%', height='90%',
                          loc='upper left',
                          bbox_to_anchor=zoom_positions[3],
                          bbox_transform=ax.transAxes)

    bp_inset = ax_inset.boxplot(
        data[3], widths=0.6,
        patch_artist=True,
        boxprops=dict(
            linewidth=2,
            color='black',
            facecolor='lightblue'
        ),
        medianprops=dict(linewidth=2, color='red'),
        whiskerprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=2, color='black'),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='gray')
    )
    bp_inset['boxes'][0].set_facecolor(colors[3])

    y_min = min(data[3].min(), lower_bound - 0.001)
    y_max = min(data[3].max(), upper_bound + 0.005)
    ax_inset.set_ylim(y_min, y_max)

    ax_inset.set_xticklabels([''])


    plt.tight_layout()

    plt.savefig('./figs_vis/time_box.png', dpi=850, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # plot_err_std()  # plot imitation error

    plot_box()  # plot box plot for calculating time

    # plot_ver_mse()  # plot validation error
