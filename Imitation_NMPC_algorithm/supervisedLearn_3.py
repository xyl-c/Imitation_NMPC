import torch
from mpc_BC_network_3 import *
from quaModel_casadi import QuadrotorModel
from mpcSolver_casadi import *

from torch import nn
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from DataSet_insert import Data
from torch.optim.lr_scheduler import StepLR
from scipy.stats import qmc, truncnorm
import os
import time
from multiprocessing import Pool, cpu_count
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

def lhs_truncated_gaussian(n, d, means, variances, lower_bounds, upper_bounds):

    assert len(means) == d and len(variances) == d
    assert len(lower_bounds) == d and len(upper_bounds) == d
    for i in range(d):
        if lower_bounds[i] >= upper_bounds[i]:
            raise ValueError(f"dimension {i}: low bound {lower_bounds[i]} must less than upper bound {upper_bounds[i]}")
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

def generate_truncated_gaussians(mean, std, low, high, n_samples):

    d = len(mean)
    a_list = [(low[i] - mean[i]) / (std[i]+1e-10) for i in range(d)]
    b_list = [(high[i] - mean[i]) / (std[i]+1e-10) for i in range(d)]
    samples = np.empty((n_samples, d))
    for i in range(d):
        samples[:, i] = truncnorm.rvs(a_list[i], b_list[i],
                                      loc=mean[i], scale=std[i],
                                      size=n_samples)
    return np.array(samples)


def parallel_solve_nmpc_ipopt(x_param):
    try:
        horizon = 3.0
        N = 30
        model = QuadrotorModel(horizon, N)
        solver_obj = MPCSolver(model)
        solver = solver_obj.solver_method()
        X_init = np.tile(x_param[ :model.nx].reshape(-1, 1), (1, N + 1))
        U_init = np.zeros([model.nu, N])
        U_init[0, :] = model.m * model.g
        init_guess = np.concatenate([X_init.flatten(order='F'), U_init.flatten(order='F')])
        sol = solver(x0=init_guess,
                     p=x_param.reshape(-1),
                     lbg=model.lbg, ubg=model.ubg,
                     lbx=model.lbx, ubx=model.ubx)
        w_opt = sol['x'].full().flatten()
        U_opt = w_opt[model.nx * (N + 1):].reshape((model.nu, N), order='F')
        u0 = np.array(U_opt[:,0]).flatten()
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

def train_bc_usv(Ts,N):
    horizon = 3.0
    num_steps = 30
    model = QuadrotorModel(horizon, num_steps)
    n_states = model.nx
    n_controls = model.nu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_train = Data()
    dataset_val = Data()
    mpcBc_net = MpcBcNetwork(n_states, n_controls).to(device)
    mpcBc_net_Optim = torch.optim.AdamW(params=mpcBc_net.parameters(), lr=0.001)
    loss_hu = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mpcBc_net_Optim,
        mode='min',
        factor=0.6,
        patience=6,
        min_lr=1e-6
    )
    err_scale = 2.0
    batch_size = 512
    epochs = 350
    train_step_bc = 0
    train_step = 0
    writer = SummaryWriter('logs_train')
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
    file_path_init = './dataset_4_18/QQUA_4e6.csv'
    dims = n_states
    if os.path.exists(file_path_init):
        print("dataset exist")
        data_np = np.loadtxt(file_path_init, delimiter=',', dtype=np.float32)
        data_np = np.concatenate((data_np[:, 0:6], data_np[:, 15:16], data_np[:, 6:12], data_np[:, -4:]), axis=1)
        np.random.shuffle(data_np)
        len_traindata = round(0.95 * data_np.shape[0])
        train_data = data_np[0:len_traindata, :]
        val_data = data_np[len_traindata:, :]
        dataset_train.del_data()
        dataset_train.add_data(torch.tensor(train_data))
        d_mean, d_std, d_max, d_min = dataset_train.normalize()
        d_mean, d_std, d_max, d_min = d_mean.to(device), d_std.to(device), d_max.to(device), d_min.to(device)
        print(d_mean)
        print(d_std)
        print(d_max)
        print(d_min)
        dataset_val.del_data()
        dataset_val.add_data(torch.tensor(val_data))

    else:
        print("dataset not exist")
        n_train = 10000
        mn_test = np.zeros(dims-2)
        sd_l_test = np.array(np.sqrt([0.2, 0.2, 0.2,  0.2, 0.2,   0.2, 0.2, 0.2,   0.2, 0.2, 0.2]))
        sample_gaussian_state = generate_truncated_gaussians(mn_test, sd_l_test, min_vals, max_vals,
                                                            n_samples=int(n_train))
        sample_uniform_yaw = np.random.uniform(low=0, high=1, size=(int(n_train), 1))
        scaled_uniform_yaw = sample_uniform_yaw * (np.pi + np.pi) - np.pi
        scaled_uniform_yaw_ref = scaled_uniform_yaw + np.random.normal(loc=0, scale=0.2, size=(int(n_train), 1))
        scaled_sample_eul = np.concatenate((sample_gaussian_state[:,:5], scaled_uniform_yaw, sample_gaussian_state[:,5:], scaled_uniform_yaw_ref), axis=1)
        q = eul2qua(scaled_sample_eul)
        scaled_sample = np.concatenate([scaled_sample_eul[:,:3], q, scaled_sample_eul[:,-7:-1], np.zeros([n_train,3]),scaled_sample_eul[:,-1:]], axis=1)
        print(f"Perform parallel MPC computations using 16 CPU cores")
        with Pool(processes = 16) as pool:
            results = pool.map(parallel_solve_nmpc_ipopt, scaled_sample)
        valid_idx = [i for i, r in enumerate(results) if r is not None]
        results_filtered = [results[i] for i in valid_idx]
        scaled_sample_eul_filtered = scaled_sample_eul[valid_idx]
        data_np = np.array(results_filtered, dtype = np.float32)
        data_np = np.concatenate((scaled_sample_eul_filtered[:, :-1], data_np[:, -8:]), axis=1)
        data_save = data_np
        np.savetxt(file_path_init, data_save, delimiter=",", fmt='%f')
        data_np = np.concatenate((data_np[:, 0:6], data_np[:, 15:16], data_np[:, 6:12], data_np[:, -4:]), axis=1)
        np.random.shuffle(data_np)
        len_traindata = round(0.9 * data_np.shape[0])
        train_data = data_np[0:len_traindata, :]
        val_data = data_np[len_traindata:, :]
        dataset_train.del_data()
        dataset_train.add_data(torch.tensor(train_data))
        d_mean, d_std, d_max, d_min = dataset_train.normalize()
        d_mean, d_std, d_max, d_min = d_mean.to(device), d_std.to(device), d_max.to(device), d_min.to(device)
        print(d_mean)
        print(d_std)
        print(d_max)
        print(d_min)
        dataset_val.del_data()
        dataset_val.add_data(torch.tensor(val_data))

    n_test = 1000
    mn_test = np.zeros(dims)
    sd_l_test = np.array(np.sqrt([0.001, 0.001, 0.001,  0.1, 0.1, 2.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 2.0]))  # 方差
    sample_gaussian = lhs_truncated_gaussian(int(n_test - n_test / 10.0), dims, mn_test, sd_l_test, min_vals, max_vals)
    sampler = qmc.LatinHypercube(d=dims)
    sample_uniform = sampler.random(n=int(n_test / 10.0))
    sample_uniform = np.array(sample_uniform)
    sample_uniform_test = sample_uniform * (max_vals - min_vals) + min_vals
    scaled_sample_test = np.concatenate([sample_gaussian, sample_uniform_test], axis=0)
    q = eul2qua(scaled_sample_test)
    scaled_sample_test_q = np.concatenate([scaled_sample_test[:, :3], q, scaled_sample_test[:, -7:-1], np.zeros([n_test, 3]), scaled_sample_test[:, -1:]], axis=1)
    print(f"Validation set: Perform parallel MPC computations using 16 CPU cores")
    with Pool(processes = 16) as pool:
        results = pool.map(parallel_solve_nmpc_ipopt, scaled_sample_test_q)
    valid_idx = [i for i, r in enumerate(results) if r is not None]
    results_filtered = [results[i] for i in valid_idx]
    scaled_sample_test_filtered = scaled_sample_test[valid_idx]
    data_test = np.array(results_filtered, dtype=np.float32)
    data_test = np.concatenate((scaled_sample_test_filtered[:,0:6], scaled_sample_test_filtered[:,-1:], scaled_sample_test_filtered[:,6:12], data_test[:, -4:]), axis=1)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    buffer_loader = DataLoader(dataset=dataset_train, batch_size=512, shuffle=True, num_workers=12, drop_last=False)
    buffer_loader_val = DataLoader(dataset=dataset_val, batch_size=512, shuffle=True, num_workers=12, drop_last=False)
    buffer_loader_tets = DataLoader(TensorDataset(data_test[:,0:13], data_test[:,-4:]), batch_size=512, shuffle=True, drop_last=False)
    loss_func = nn.MSELoss()

    for nt_bc in range(epochs):
        mpcBc_net.train()
        for data in buffer_loader:
            tra_batch, u_sol_batch = data
            tra_batch, u_sol_batch = tra_batch.to(device), u_sol_batch.to(device)
            u_bc_batch = mpcBc_net(tra_batch.detach())
            loss_bc = loss_func(u_bc_batch, u_sol_batch.detach())
            mpcBc_net_Optim.zero_grad()
            loss_bc.backward()
            mpcBc_net_Optim.step()
            train_step = train_step + 1
            label_dis = u_sol_batch * (d_max - d_min) + d_min
            u_bc_out_dis = u_bc_batch * (d_max - d_min) + d_min
            mae_err = np.abs(label_dis.detach().cpu().numpy() - u_bc_out_dis.detach().cpu().numpy()).mean(0)
            if train_step % 100 == 0:
                print('training times：{}, Loss_bc：{}'.format(train_step, loss_bc.item()))
                print('training times：{}, mae_err：{}'.format(train_step, mae_err.tolist()))
                writer.add_scalar('Loss_bc', loss_bc.item(), train_step)
                writer.add_scalar('mae_T_train', mae_err[0].item(), train_step)
                writer.add_scalar('mae_mx_train', mae_err[1].item(), train_step)
                writer.add_scalar('mae_my_train', mae_err[2].item(), train_step)
                writer.add_scalar('mae_mz_train', mae_err[3].item(), train_step)
                print('Learning rate：{}'.format(mpcBc_net_Optim.param_groups[0]['lr']))
        mpcBc_net.eval()
        with torch.no_grad():
            torch.save(mpcBc_net, './modelSave/mpcBc_net_{}.pth'.format(nt_bc))
            print('Model saved')
            loss_val_total = 0
            mae_val_total = np.zeros(4)
            val_step = 0
            for data in buffer_loader_val:
                val_in, val_label = data
                val_in, val_label = val_in.to(device), val_label.to(device)
                val_in = (val_in - d_mean) / d_std
                val_label_norm = (val_label - d_min) / (d_max - d_min)
                val_out_norm = mpcBc_net(val_in)
                loss_val = loss_hu(val_label_norm, val_out_norm)
                loss_val_total = loss_val.item() + loss_val_total
                val_out = val_out_norm * (d_max - d_min) + d_min
                mae_err_val = np.abs(val_out.detach().cpu().numpy() - val_label.detach().cpu().numpy()).mean(0)
                mae_val_total = mae_err_val + mae_val_total
                val_step = val_step + 1
            scheduler.step(loss_val_total / val_step)
            print(val_label[0:10, 0])
            print(val_out[0:10, 0])
            print(val_label[0:10, 1])
            print(val_out[0:10, 1])
            print('Validation error (MSE)：{}'.format(loss_val_total / val_step))
            writer.add_scalar('val_loss', loss_val_total / val_step, nt_bc)
            mean_mae_val = mae_val_total / val_step
            print('Validation error (MSE)：{}'.format(mean_mae_val.tolist()))
            writer.add_scalar('mae_T_val', mean_mae_val[0].item(), nt_bc)
            writer.add_scalar('mae_mx_val', mean_mae_val[1].item(), nt_bc)
            writer.add_scalar('mae_my_val', mean_mae_val[2].item(), nt_bc)
            writer.add_scalar('mae_mz_val', mean_mae_val[3].item(), nt_bc)
            loss_test_total = 0
            mae_test_total = np.zeros(4)
            test_step = 0
            for data in buffer_loader_tets:
                test_in, test_label = data
                test_in, test_label = test_in.to(device), test_label.to(device)
                test_in = (test_in - d_mean) / d_std
                test_label_norm = (test_label - d_min) / (d_max - d_min)
                test_out_norm = mpcBc_net(test_in)
                loss_test = loss_hu(test_label_norm, test_out_norm)
                loss_test_total = loss_test.item() + loss_test_total
                test_out = test_out_norm*(d_max - d_min) + d_min
                mae_err_test = np.abs(test_out.detach().cpu().numpy() - test_label.detach().cpu().numpy()).mean(0)
                mae_test_total = mae_err_test + mae_test_total
                test_step = test_step + 1
            print('---------------------------')
            print(test_label[0:10,0])
            print(test_out[0:10,0])
            print(test_label[0:10, 1])
            print(test_out[0:10, 1])
            print('Test error MSE：{}'.format(loss_test_total / test_step))
            writer.add_scalar('test_loss', loss_test_total/test_step, nt_bc)
            mean_mae_test = mae_test_total / test_step
            print('Test error MSE：{}'.format(mean_mae_test.tolist()))

    writer.close()
    mpcBc_net.eval()
    n_test = 1000
    mn_test = np.zeros(dims)
    sd_l_test = np.array(np.sqrt([0.00001, 0.00001, 0.00001,  0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 0.1])) # 方差
    sample_gaussian = generate_truncated_gaussians(mn_test, sd_l_test, min_vals, max_vals,
                                                   n_samples=int(n_test - n_test / 10.0))
    sample_uniform = np.random.uniform(low=0, high=1, size=(int(n_test / 10.0), dims))
    scaled_uniform = sample_uniform * (max_vals - min_vals) + min_vals
    test_in = np.concatenate([sample_gaussian, scaled_uniform], axis=0)
    q = eul2qua(test_in)
    test_in_q = np.concatenate([test_in[:, :3], q, test_in[:, -7:-1], np.zeros([n_test, 3]), test_in[:, -1:]], axis=1)
    print(f"Test set: Parallel MPC computation using 16 CPU cores")
    with Pool(processes = 16) as pool:
        results = pool.map(parallel_solve_nmpc_ipopt, test_in_q)
    valid_idx = [i for i, r in enumerate(results) if r is not None]
    results_filtered = [results[i] for i in valid_idx]
    test_in_filtered = test_in[valid_idx]
    data_np_test = np.array(results_filtered, dtype=np.float32)
    data_test = np.concatenate((test_in_filtered[:,0:6], test_in_filtered[:,-1:], test_in_filtered[:,6:12], data_np_test[:, -4:]), axis=1)
    data_test = torch.tensor(data_test, dtype=torch.float32, device=device)
    test_in = (data_test[:,0:13] - d_mean) / d_std
    u_bc_out = mpcBc_net(test_in)
    u_bc_out = u_bc_out * (d_max - d_min) + d_min
    errors = (data_test[:,-4:] - u_bc_out).detach().cpu().numpy()
    mean_error = np.abs(errors).mean(0)
    max_error = np.abs(errors).max(0)
    print(mean_error)
    print(max_error)

    plt.figure()
    plt.hist(errors[:, 0], bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error T1 bin')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.figure()
    plt.hist(errors[:, 1], bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error T2 bin')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.figure()
    plt.hist(errors[:, 2], bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error T3 bin')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.figure()
    plt.hist(errors[:, 3], bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error T4 bin')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    Ts = 0.02
    N = 35
    train_bc_usv(Ts, N)





