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

def lhs_truncated_gaussian(n, d, means, variances, lower_bounds, upper_bounds):

    assert len(means) == d and len(variances) == d
    assert len(lower_bounds) == d and len(upper_bounds) == d
    for i in range(d):
        if lower_bounds[i] >= upper_bounds[i]:
            raise ValueError(f"dim {i}: low bound {lower_bounds[i]} must smaller than {upper_bounds[i]}")
    sampler = qmc.LatinHypercube(d=d)
    uniform_samples = sampler.random(n=n)
    samples = np.zeros((n, d))
    for i in range(d):
        a, b = lower_bounds[i], upper_bounds[i]

        a_std = (a - means[i]) / np.sqrt(variances[i])
        b_std = (b - means[i]) / np.sqrt(variances[i])

        samples[:, i] = truncnorm.ppf(
            uniform_samples[:, i],
            a=a_std,
            b=b_std,
            loc=means[i],
            scale=np.sqrt(variances[i])
        )

    return np.array(samples)

def parallel_solve_nmpc(x_param):
    try:
        horizon = 3.0
        N = 30
        model = QuadrotorModel(horizon, N)
        solver_obj = MPCSolver(model)
        solver = solver_obj.solver_method()
        X_init = np.tile(x_param[:model.nx].reshape(-1, 1), (1, N + 1))
        U_init = np.zeros([model.nu, N])
        U_init[0, :] = model.m * model.g
        init_guess = np.concatenate([X_init.flatten(order='F'), U_init.flatten(order='F')])
        sol = solver(x0=init_guess,
                     p=x_param.reshape(-1),
                     lbg=model.lbg, ubg=model.ubg,
                     lbx=model.lbx, ubx=model.ubx)
        w_opt = sol['x'].full().flatten()
        U_opt = w_opt[model.nx * (N + 1):].reshape((model.nu, N), order='F')
        u0 = np.array(U_opt[:, 0]).flatten()
        return np.concatenate([x_param.flatten(), u0])
    except Exception:
        return None

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

def ver_test():
    horizon = 3.0
    N = 30
    model = QuadrotorModel(horizon, N)
    mpc_solver = MPCSolver(model)
    dims = model.nx
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    nn_controller = torch.load('../modelSave/mpcBc_net_298.pth', weights_only=False).to(device)
    nn_controller.eval()

    # file_path_init = '../dataset/QQUA_4e6.csv'
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

    err_scale = 2.0
    bounds = [
        [-err_scale, err_scale],
        [-err_scale, err_scale],
        [-err_scale, err_scale],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [model.vx_min, model.vx_max],
        [model.vy_min, model.vy_max],
        [model.vz_min, model.vz_max],
        [model.wx_min, model.wx_max],
        [model.wy_min, model.wy_max],
        [model.wz_min, model.wz_max],
        [-np.pi, np.pi]
    ]
    min_vals = np.array([b[0] for b in bounds])
    max_vals = np.array([b[1] for b in bounds])


    for num in range(10):
        n_test = 1000

        mn_test = np.zeros(dims)
        sd_l_test = np.array(np.sqrt([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))  # 方差
        sample_gaussian = lhs_truncated_gaussian(int(n_test - n_test / 10.0), dims, mn_test, sd_l_test, min_vals,
                                                 max_vals)
        sampler = qmc.LatinHypercube(d=dims)
        sample_uniform = sampler.random(n=int(n_test / 10.0))
        sample_uniform = np.array(sample_uniform)
        sample_uniform_test = sample_uniform * (max_vals - min_vals) + min_vals
        test_in = np.concatenate([sample_gaussian, sample_uniform_test], axis=0)
        q = eul2qua(test_in)
        test_in_q = np.concatenate([test_in[:, :3], q, test_in[:, -7:-1], np.zeros([n_test, 3]), test_in[:, -1:]],
                                   axis=1)

        print(f"测试集：使用 16 个 CPU 核心进行并行 MPC 计算")
        with Pool(processes=16) as pool:
            results = pool.map(parallel_solve_nmpc, test_in_q)
        results = [r for r in results if r is not None]
        data_np_test = np.array(results, dtype=np.float32)
        data_test = np.concatenate((test_in[:, 0:6], test_in[:, -1:], test_in[:, 6:12], data_np_test[:, -4:]), axis=1)
        data_test = torch.tensor(data_test, dtype=torch.float32, device=device)
        test_in = (data_test[:, 0:13] - d_mean) / d_std
        u_bc_out = nn_controller(test_in)
        u_bc_out = u_bc_out * (d_max - d_min) + d_min
        errors = (data_test[:, -4:] - u_bc_out).detach().cpu().numpy()
        np.savetxt('../TEST_results/ver_errors{}.csv'.format(num), errors, delimiter=",", fmt='%f')
        mse = np.mean(np.sum(np.square(errors), axis=1), axis=0)
        err_m4 = np.abs(np.sum(errors.T, axis=0))/4.0
        mean = np.mean(err_m4)
        std = np.std(err_m4)

if __name__ == '__main__':

    ver_test()



