import numpy as np
from torch import nn
import torch
from mpc_BC_network_3 import *
from quaModel_casadi import QuadrotorModel
from mpcSolver_casadi import *
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from multiprocessing import Pool, cpu_count
import time


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

def euler2quan(phi, theta, psi):
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qx = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    q = np.array([qw, qx, qy, qz])
    return q




def control():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    horizon = 3.0
    N = 30
    model = QuadrotorModel(horizon, N)
    mpc_solver = MPCSolver(model)

    sim_time = 30.0
    Ts = 0.02
    steps = int(sim_time / Ts)
    t_vec = Ts * np.linspace(0, steps, steps + 1)
    ##############################################################################################
    #   Any desired trajectory can be defined here.
    ##############################################################################################
    # ref_traj = np.concatenate((2 * np.cos(0.2 * t_vec.reshape(1, -1)),
    #                            2 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                            10.0 + 2 * np.sin(0.1 * t_vec.reshape(1, -1)),
    #                            np.arctan2(1.0, 1.0) * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((0.5 * t_vec.reshape(1, -1)+2.0,
    #                            0.5 * t_vec.reshape(1, -1)+2.0,
    #                            10.0 + 0.5 * t_vec.reshape(1, -1),
    #                            np.arctan2(1.0, 1.0) * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((5 * np.cos(0.2 * t_vec.reshape(1, -1)),
    #                            5 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                            10.0 + 1 * t_vec.reshape(1, -1),
    #                            0.0 * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((10 * np.cos(0.2 * t_vec.reshape(1, -1)),
    #                            10 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                            10.0 * np.ones([1, t_vec.shape[0]]),
    #                            (np.pi/2) * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((10 * np.cos(0.2 * t_vec.reshape(1, -1)),
    #                            10 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                            -10.0 - 1 * t_vec.reshape(1, -1),
    #                            0.0 * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((10 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                                0.5 * 10 * np.sin(2.0 * 0.2 * t_vec.reshape(1, -1)),
    #                                -10.0 * np.ones([1, t_vec.shape[0]]),
    #                                np.arctan2(2.0 * np.cos(0.4 * t_vec.reshape(1, -1)), 2.0 * np.cos(0.2 * t_vec.reshape(1, -1)))), axis=0)
    # ref_traj = np.concatenate(((10 * np.sin(0.2 * t_vec.reshape(1, -1))-10.0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
    #                                (10 * np.sin(0.2 * t_vec.reshape(1, -1))+10.0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
    #                                0.5 * 10 * np.sin(2.0 * 0.2 * t_vec.reshape(1, -1)),
    #                                np.arctan2(1.0, 1.0)* np.ones([1, t_vec.shape[0]])), axis=0)
    ref_traj = np.concatenate(
        ((10 * np.cos(0.2 * t_vec.reshape(1, -1)) - 10.0 * np.ones([1, t_vec.shape[0]])) * np.sqrt(2) / 2.0,
         (10 * np.cos(0.2 * t_vec.reshape(1, -1)) + 10.0 * np.ones([1, t_vec.shape[0]])) * np.sqrt(2) / 2.0,
         -10 * np.sin(0.2 * t_vec.reshape(1, -1)),
         np.arctan2(1.0, 1.0) * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate(
    #     ((0.0 + 5.0 * np.cos(0.5 * t_vec.reshape(1, -1)) - 0.0 * np.ones([1, t_vec.shape[0]])) * np.sqrt(2) / 2.0,
    #      (0.0 + 5.0 * np.cos(0.5 * t_vec.reshape(1, -1)) + 0.0 * np.ones([1, t_vec.shape[0]])) * np.sqrt(2) / 2.0,
    #      0.0 - 5.0 * np.sin(0.5 * t_vec.reshape(1, -1)),
    #      0.0*np.pi * np.ones([1, t_vec.shape[0]])), axis=0)

    # ref_traj = np.concatenate(
    #     (0.0  * np.ones([1, t_vec.shape[0]]),
    #      0.0  * np.ones([1, t_vec.shape[0]]),
    #      -0.5 * np.ones([1, t_vec.shape[0]]),
    #      0.0 * np.pi * np.ones([1, t_vec.shape[0]])), axis=0)
##############################################################################################
##############################################################################################

    # file_path_init = './dataset/QQUA_4e6.csv'
    # data_np = np.loadtxt(file_path_init, delimiter=',', dtype=np.float32)
    # data_np = np.concatenate((data_np[:, 0:6], data_np[:, 15:16], data_np[:, 6:12], data_np[:, -4:]), axis=1)
    # d_mean, d_std = data_np[:, 0:13].mean(0), data_np[:, 0:13].std(0)
    # d_mean, d_std = torch.tensor(d_mean, device=device), torch.tensor(d_std, device=device)
    # print(d_mean)
    # print(d_std)
    # d_max = torch.tensor(data_np[:, -4:].max(0), device=device)
    # d_min = torch.tensor(data_np[:, -4:].min(0), device=device)
    # print(d_max)
    # print(d_min)

    d_mean = torch.tensor([1.0589e-05, 1.1117e-05, 1.3680e-05, 1.6457e-05, -4.5605e-05,
                       -2.8571e-04, -2.4076e-04, -9.0192e-05, 3.2181e-05, -1.3122e-04,
                       3.5321e-06, -3.5471e-05, 2.1887e-05], device=device)
    d_std = torch.tensor([0.3367, 0.3367, 0.3367, 0.7798, 0.7798, 0.8725, 0.8727, 0.9322, 0.9321,
                      0.9320, 0.2500, 0.2500, 0.2500], device=device)
    d_max = torch.tensor([40.0000, 0.0837, 0.0982, 0.1114], device=device)
    d_min = torch.tensor([-0.0000, -0.0837, -0.0981, -0.1113], device=device)

    nx = model.nx
    nu = model.nu

    nn_controller = torch.load('./modelSave/mpcBc_net_298.pth', weights_only=False).to(device)
    nn_controller.eval()

    x_sim_mpc = np.zeros([nx-1, steps + 1])
    x_sim_nn = x_sim_mpc.copy()
    u_sim_mpc = np.zeros([nu, steps])
    # The initial position is within [2, 2, 2]^3 of the initial reference trajectory point.
    x0_mpc = np.array([ref_traj[0,0] + 1.0, ref_traj[1,0] + 1.0,ref_traj[2,0] + 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_sim_mpc[:, 0] = x0_mpc.copy()
    x_sim_nn[:, 0] = x0_mpc.copy()
    phi, theta, psi = x0_mpc[3], x0_mpc[4], x0_mpc[5]
    phi_nn, theta_nn, psi_nn = x0_mpc[3], x0_mpc[4], x0_mpc[5]
    q = euler2quan(phi, theta, psi)
    x0_mpc = np.concatenate((x0_mpc[0:3], q, x0_mpc[6:12]), axis=0)
    X_init = np.tile(x0_mpc.reshape(-1, 1), (1, N + 1))
    U_init = np.zeros([nu, N])
    U_init[0, :] = model.m * model.g
    init_guess = np.concatenate([X_init.flatten(order='F'), U_init.flatten(order='F')])

    x0_nn = x0_mpc.copy()
    u_sim_nn = u_sim_mpc.copy()

    for i in range(steps):
        ref_now = ref_traj[:, i]
        #  NMPC control
        p_val_mpc = np.concatenate([x0_mpc, ref_now.flatten(order='F')])
        start_mpc = time.perf_counter()
        sol = mpc_solver.solver(x0=init_guess,
                                p=p_val_mpc.reshape(-1),
                                lbg=model.lbg, ubg=model.ubg,
                                lbx=model.lbx, ubx=model.ubx)
        end_mpc = time.perf_counter()
        w_opt = sol['x'].full().flatten()
        X_opt = w_opt[:nx * (N + 1)].reshape((nx, N + 1), order='F')
        U_opt_mpc = w_opt[nx * (N + 1):].reshape((nu, N), order='F')
        x0_mpc = model.step_update(x0_mpc, U_opt_mpc, Ts)
        x0_mpc = np.array(x0_mpc).reshape(-1)
        q0, q1, q2, q3 = x0_mpc[3], x0_mpc[4], x0_mpc[5], x0_mpc[6]
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 * q2 + q3 * q3)))
        theta = np.arcsin(2 * (q0 * q2 - q1 * q3))
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), (1 - 2 * (q1 * q1 + q2 * q2)))
        x_sim_mpc[:, i + 1] = np.concatenate((x0_mpc[0:3], np.array([phi, theta, psi]), x0_mpc[7:13]), axis=0)
        u_sim_mpc[:, i] = U_opt_mpc[:, 0]

        X_init_shift = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_init_shift = np.hstack([U_opt_mpc[:, 1:], U_opt_mpc[:, -1:]])
        init_guess = np.concatenate([X_init_shift.flatten(order='F'), U_init_shift.flatten(order='F')])

        if i % 50 == 0:
            print(f"Sim time {i * Ts:.2f}s / {sim_time:.1f}s")

        #  IMPC control
        p0_nn = np.concatenate([x0_nn[0:3] - ref_now[0:3], np.array([phi_nn, theta_nn, psi_nn]), np.array([ref_now[3]]), x0_nn[-6:]], axis=0) # calculate position error
        e0_nn = (torch.tensor(p0_nn, device=device, dtype=torch.float32).reshape(1, -1) - d_mean) / d_std
        start_nn = time.perf_counter()
        U_opt_nn = nn_controller(e0_nn)  # NN forward
        end_nn = time.perf_counter()
        U_opt_nn = U_opt_nn * (d_max - d_min) + d_min
        U_opt_nn = U_opt_nn.cpu().detach().numpy().reshape(-1, 1)

        x0_nn = model.step_update(x0_nn, U_opt_nn, Ts)
        x0_nn = ca.reshape(x0_nn, -1, 1)
        x0_nn = np.array(x0_nn).reshape(-1)
        q0_nn, q1_nn, q2_nn, q3_nn = x0_nn[3], x0_nn[4], x0_nn[5], x0_nn[6]
        psi_nn = np.arctan2(2 * (q0_nn * q3_nn + q1_nn * q2_nn), (1 - 2 * (q2_nn * q2_nn + q3_nn * q3_nn)))
        theta_nn = np.arcsin(2 * (q0_nn * q2_nn - q1_nn * q3_nn))
        phi_nn = np.arctan2(2 * (q0_nn * q1_nn + q2_nn * q3_nn), (1 - 2 * (q1_nn * q1_nn + q2_nn * q2_nn)))
        x_sim_nn[:, i + 1] = np.concatenate((x0_nn[0:3], np.array([phi_nn, theta_nn, psi_nn]), x0_nn[7:13]), axis=0)
        u_sim_nn[:, i] = U_opt_nn.reshape(-1)
        print(f"mpc: {end_mpc - start_mpc:.6f} sec,,NN: {end_nn - start_nn:.6f} sec")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_sim_mpc[0, :], x_sim_mpc[1, :], x_sim_mpc[2, :], label='NMPC')
    ax.plot(x_sim_nn[0, :], x_sim_nn[1, :], x_sim_nn[2, :], label='IMPC')
    ax.plot(ref_traj[0, :], ref_traj[1, :], ref_traj[2, :], 'r--', label='REF')
    ax.set_xlabel("$p_x$[m]")
    ax.set_ylabel("$p_y$[m]")
    ax.set_zlabel("$p_z$[m]")
    ax.legend()

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[0, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[0, :].reshape(-1), label='IMPC')
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[0, :].reshape(-1), label='REF')
    plt.xlabel("t(s)")
    plt.ylabel("x")
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[1, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[1, :].reshape(-1), label='IMPC')
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[1, :].reshape(-1), label='REF')
    plt.xlabel("t(s)")
    plt.ylabel("y")
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[2, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[2, :].reshape(-1), label='IMPC')
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[2, :].reshape(-1), label='REF')
    plt.xlabel("t(s)")
    plt.ylabel("z")
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[5, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[5, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[3, :].reshape(-1) * 180.0 / np.pi, label='REF')
    plt.xlabel("t(s)")
    plt.ylabel("psi")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[3, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[3, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("phi")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[4, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[4, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("theta")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[5, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[5, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("psi")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[6, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[6, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("vx")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[7, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[7, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("vy")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[8, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[8, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("vz")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[9, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[9, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("wx")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[10, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[10, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("wy")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim_mpc[11, :].reshape(-1) * 180.0 / np.pi, label='NMPC')
    plt.plot(np.arange(steps + 1) * Ts, x_sim_nn[11, :].reshape(-1) * 180.0 / np.pi, label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("wz")
    plt.legend()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(steps) * Ts, u_sim_mpc[0, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps) * Ts, u_sim_nn[0, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("T")
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(steps) * Ts, u_sim_mpc[1, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps) * Ts, u_sim_nn[1, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("Mx")
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(steps) * Ts, u_sim_mpc[2, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps) * Ts, u_sim_nn[2, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("My")
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(steps) * Ts, u_sim_mpc[3, :].reshape(-1), label='NMPC')
    plt.plot(np.arange(steps) * Ts, u_sim_nn[3, :].reshape(-1), label='IMPC')
    plt.xlabel("t(s)")
    plt.ylabel("Mz")
    plt.legend()

    plt.show()



if __name__ == '__main__':

    control()




