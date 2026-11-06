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

##############################################################################################
    #   Reformulated forward propagation.
##############################################################################################
nn_controller = torch.load('./modelSave/mpcBc_net_298.pth', weights_only=False).to(torch.device('cpu'))
nn_controller.eval()
W1 = nn_controller.model[0].weight.detach().cpu().numpy().astype(np.float32)
b1 = nn_controller.model[0].bias.detach().cpu().numpy().astype(np.float32)
W2 = nn_controller.model[2].weight.detach().cpu().numpy().astype(np.float32)
b2 = nn_controller.model[2].bias.detach().cpu().numpy().astype(np.float32)
W3 = nn_controller.model[4].weight.detach().cpu().numpy().astype(np.float32)
b3 = nn_controller.model[4].bias.detach().cpu().numpy().astype(np.float32)
W4 = nn_controller.model[6].weight.detach().cpu().numpy().astype(np.float32)
b4 = nn_controller.model[6].bias.detach().cpu().numpy().astype(np.float32)

d_mean = np.array([ 1.0589e-05,  1.1117e-05,  1.3680e-05,  1.6457e-05, -4.5605e-05,
        -2.8571e-04, -2.4076e-04, -9.0192e-05,  3.2181e-05, -1.3122e-04,
         3.5321e-06, -3.5471e-05,  2.1887e-05]).astype(np.float32)
d_std = np.array([0.3367, 0.3367, 0.3367, 0.7798, 0.7798, 0.8725, 0.8727, 0.9322, 0.9321,
        0.9320, 0.2500, 0.2500, 0.2500]).astype(np.float32)
d_max = np.array([40.0000,  0.0837,  0.0982,  0.1114]).astype(np.float32)
d_min = np.array([-0.0000, -0.0837, -0.0981, -0.1113]).astype(np.float32)

def relu(x):
    return np.maximum(x, 0, out=x)

def forward_numpy(x):

    z1 = W1 @ x + b1
    a1 = relu(z1)

    z2 = W2 @ a1 + b2
    a2 = relu(z2)

    z3 = W3 @ a2 + b3
    a3 = relu(z3)

    z4 = W4 @ a3 + b4
    a4 = 1 / (1 + np.exp(-z4))

    return a4
##############################################################################################
##############################################################################################

def control(ref_traj, sim_time, d_mean, d_std, d_max, d_min, device, nn_controller, steps):

    horizon = 3.0
    N = 30
    model = QuadrotorModel(horizon, N)
    mpc_solver = MPCSolver(model)

    nx = model.nx
    nu = model.nu
    x_sim_mpc = np.zeros([nx-1, steps + 1])
    x_sim_nn = x_sim_mpc.copy()
    u_sim_mpc = np.zeros([nu, steps])
    x0_mpc = np.array([ref_traj[0,0] + 2.0, ref_traj[1,0] + 2.0, ref_traj[2,0] + 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
    time_save = np.zeros([2, steps])

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
        p0_nn = np.concatenate([x0_nn[0:3] - ref_now[0:3], np.array([phi_nn, theta_nn, psi_nn]), np.array([ref_now[3]]), x0_nn[-6:]], axis=0)
        e0_nn = (torch.tensor(p0_nn, device=device, dtype=torch.float32).reshape(1, -1) - d_mean) / d_std
        start_nn = time.perf_counter()
        U_opt_nn = nn_controller(e0_nn)
        end_nn = time.perf_counter()
        U_opt_nn = U_opt_nn * (d_max - d_min) + d_min
        U_opt_nn = U_opt_nn.cpu().detach().numpy().reshape(-1, 1)

        # e0_nn = (p0_nn.astype(np.float32) - d_mean) / d_std
        # start_nn = time.perf_counter()
        # U_opt_nn = forward_numpy(e0_nn)  # NN forward
        # end_nn = time.perf_counter()
        # U_opt_nn = U_opt_nn * (d_max - d_min) + d_min
        # U_opt_nn = U_opt_nn.reshape(-1, 1)

        x0_nn = model.step_update(x0_nn, U_opt_nn, Ts)
        x0_nn = ca.reshape(x0_nn, -1, 1)
        x0_nn = np.array(x0_nn).reshape(-1)
        q0_nn, q1_nn, q2_nn, q3_nn = x0_nn[3], x0_nn[4], x0_nn[5], x0_nn[6]
        psi_nn = np.arctan2(2 * (q0_nn * q3_nn + q1_nn * q2_nn), (1 - 2 * (q2_nn * q2_nn + q3_nn * q3_nn)))
        theta_nn = np.arcsin(2 * (q0_nn * q2_nn - q1_nn * q3_nn))
        phi_nn = np.arctan2(2 * (q0_nn * q1_nn + q2_nn * q3_nn), (1 - 2 * (q1_nn * q1_nn + q2_nn * q2_nn)))
        x_sim_nn[:, i + 1] = np.concatenate((x0_nn[0:3], np.array([phi_nn, theta_nn, psi_nn]), x0_nn[7:13]), axis=0)
        u_sim_nn[:, i] = U_opt_nn.reshape(-1)
        print(f"mpc: {end_mpc - start_mpc:.6f} 秒,,NN: {end_nn - start_nn:.6f} 秒")
        time_save[0, i] = end_mpc - start_mpc
        time_save[1, i] = end_nn - start_nn

    return x_sim_mpc, x_sim_nn, time_save


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    nn_controller = torch.load('./modelSave/mpcBc_net_298.pth', weights_only=False).to(device)
    nn_controller.eval()

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

    sim_time = 30.0
    Ts = 0.02

    NN = 100
    steps = int(sim_time / Ts)
    t_vec = Ts * np.linspace(0, steps, steps + 1)
    mpc_save = np.zeros([NN*(13-1), steps + 1])
    nn_save = np.zeros([NN * (13 - 1), steps + 1])
    time_save = np.zeros([NN * (2), steps])
    ref_save = np.zeros([NN * (4), steps + 1])
    for i in range(NN):
        print(f"进度： {i/NN :.2f}%")
        if i<25 and i>=0:
            px_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            py_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            pz_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            f_d = np.random.uniform(low=0.1, high=np.pi/2, size=(1, 1)).item()
            A_d = np.random.uniform(low=-3/f_d, high=3/f_d, size=(1, 1)).item()
            psi_0 = np.random.uniform(low=-2*np.pi/3, high=2*np.pi/3, size=(1, 1)).item()
            ref_traj = np.concatenate(((px_0 + A_d * np.cos(f_d * t_vec.reshape(1, -1))-pz_0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
                                       (px_0 + A_d * np.cos(f_d * t_vec.reshape(1, -1))+pz_0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
                                       py_0 + A_d * np.sin(f_d * t_vec.reshape(1, -1)),
                                       psi_0 * np.ones([1, t_vec.shape[0]])), axis=0)
            mpc_save[12*i:12*i+12, :], nn_save[12*i:12*i+12, :], time_save[2*i:2*i+2, :] = control(ref_traj, sim_time,d_mean,d_std,d_max,d_min,device,nn_controller, steps)
            ref_save[4*i:4*i+4, :] = ref_traj
        if i < 50 and i >= 25:
            px_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            py_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            pz_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            f_d = np.random.uniform(low=0.1, high=np.pi / 2, size=(1, 1)).item()
            A_d = np.random.uniform(low=-3 / f_d, high = 3 / f_d, size=(1, 1)).item()
            v_z = np.random.uniform(low=-3.0, high=3.0, size=(1, 1)).item()
            psi_0 = np.random.uniform(low=-2*np.pi/3, high=2*np.pi/3, size=(1, 1)).item()
            ref_traj = np.concatenate((px_0 + A_d * np.cos(f_d * t_vec.reshape(1, -1)),
                                       py_0 + A_d * np.sin(f_d * t_vec.reshape(1, -1)),
                                       pz_0 + v_z * t_vec.reshape(1, -1),
                                       psi_0 * np.ones([1, t_vec.shape[0]])), axis=0)
            mpc_save[12 * i:12 * i + 12, :], nn_save[12 * i:12 * i + 12, :], time_save[2 * i:2 * i + 2, :] = control(ref_traj, sim_time,d_mean,d_std,d_max,d_min,device,nn_controller, steps)
            ref_save[4 * i:4 * i + 4, :] = ref_traj
        if i < 75 and i >= 50:
            px_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            py_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            pz_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            v_x = np.random.uniform(low=-3.0, high=3.0, size=(1, 1)).item()
            v_y = np.random.uniform(low=-3.0, high=3.0, size=(1, 1)).item()
            v_z = np.random.uniform(low=-3.0, high=3.0, size=(1, 1)).item()
            psi_0 = np.random.uniform(low=-2*np.pi/3, high=2*np.pi/3, size=(1, 1)).item()
            ref_traj = np.concatenate((px_0 + v_x * t_vec.reshape(1, -1),
                                       py_0 + v_y * t_vec.reshape(1, -1),
                                       pz_0 + v_z * t_vec.reshape(1, -1),
                                       psi_0 * np.ones([1, t_vec.shape[0]])), axis=0)
            mpc_save[12 * i:12 * i + 12, :], nn_save[12 * i:12 * i + 12, :], time_save[2 * i:2 * i + 2, :] = control(ref_traj, sim_time,d_mean,d_std,d_max,d_min,device,nn_controller, steps)
            ref_save[4 * i:4 * i + 4, :] = ref_traj
        if i < 100 and i >= 75:
            px_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            py_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            pz_0 = np.random.uniform(low=-1000.0, high=1000.0, size=(1, 1)).item()
            f_d = np.random.uniform(low=0.1, high=np.pi / 2, size=(1, 1)).item()
            A_d = np.random.uniform(low=-3 / f_d, high=3 / f_d, size=(1, 1)).item()
            psi_0 = np.random.uniform(low=-2*np.pi/3, high=2*np.pi/3, size=(1, 1)).item()
            ref_traj = np.concatenate(((px_0 + A_d * np.sin(f_d * t_vec.reshape(1, -1))-pz_0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
                                       (px_0 + A_d * np.sin(f_d * t_vec.reshape(1, -1))+pz_0 * np.ones([1, t_vec.shape[0]]))*np.sqrt(2)/2.0,
                                       py_0 + 0.5 * A_d * np.sin(2.0 * f_d * t_vec.reshape(1, -1)),
                                       psi_0 * np.ones([1, t_vec.shape[0]])), axis=0)
            mpc_save[12 * i:12 * i + 12, :], nn_save[12 * i:12 * i + 12, :], time_save[2 * i:2 * i + 2, :] = control(ref_traj, sim_time,d_mean,d_std,d_max,d_min,device,nn_controller, steps)
            ref_save[4 * i:4 * i + 4, :] = ref_traj

    np.savetxt('./TEST_results/mpc_save_1.csv', mpc_save, delimiter=",", fmt='%f')
    np.savetxt('./TEST_results/nn_save_1.csv', nn_save, delimiter=",", fmt='%f')
    np.savetxt('./TEST_results/ref_save_1.csv', ref_save, delimiter=",", fmt='%f')
    np.savetxt('./TEST_results/time_save_gpu_1.csv', time_save, delimiter=",", fmt='%f')
    # np.savetxt('./TEST_results/time_save_cpu_1.csv', time_save, delimiter=",", fmt='%f')
    # np.savetxt('./TEST_results/time_save_numpy_1.csv', time_save, delimiter=",", fmt='%f')


