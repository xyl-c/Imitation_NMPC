import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import time
from quaModel_casadi import *

class MPCSolver:
    def __init__(self, Model):

        N = Model.N
        self.nu = Model.nu
        self.nx = Model.nx
        self.f = Model.f
        self.Q = Model.Q
        self.R = Model.R
        # self.R_du = Model.R_du
        # hov_u = ca.vertcat(Model.m * Model.g, 0., 0., 0.)
        hov_u = ca.vertcat(0., 0., 0., 0.)

        self.U = ca.SX.sym('U', self.nu, N)
        self.X = ca.SX.sym('X', self.nx, N + 1)
        self.X0_param = ca.SX.sym('X0_param', self.nx)
        self.X_ref = ca.SX.sym('X_ref', 4)

        self.obj = 0
        self.g = []

        for i in range(N):
            x_next_pred = self.f(self.X[:, i], self.U[:, i])
            self.g.append(self.X[:, i + 1] - x_next_pred)
            q0, q1, q2, q3 = self.X[3, i], self.X[4, i], self.X[5, i], self.X[6, i]
            psi_k = ca.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 * q2 + q3 * q3)))
            psi_e = ca.arctan2(ca.sin(psi_k - self.X_ref[3]), ca.cos(psi_k - self.X_ref[3]))
            Xa = ca.vertcat(self.X[0:3, i] - self.X_ref[0:3], psi_e)

            self.obj = self.obj + Xa.T @ self.Q @ Xa + (self.U[:, i] - hov_u).T @ self.R @ (self.U[:, i] - hov_u)
            # if i > 0:
            #     du = self.U[:, i] - self.U[:, i-1]
            #     self.obj = self.obj + du.T @ self.R_du @ du
        q0, q1, q2, q3 = self.X[3, N], self.X[4, N], self.X[5, N], self.X[6, N]
        psi_k = ca.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 * q2 + q3 * q3)))
        psi_e = ca.arctan2(ca.sin(psi_k - self.X_ref[3]), ca.cos(psi_k - self.X_ref[3]))
        Xa = ca.vertcat(self.X[0:3, N] - self.X_ref[0:3], psi_e)
        self.obj = self.obj + Xa.T @ self.Q @ Xa

        opt_vars = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        self.g.append(self.X[:, 0] - self.X0_param)
        self.g = ca.vertcat(*self.g)

        self.nlp = {'f': self.obj, 'x': opt_vars, 'g': self.g, 'p': ca.vertcat(self.X0_param, self.X_ref)}
        self.opts = {'ipopt.print_level': 0, 'print_time': 0}


        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)

    def solver_method(self):
        solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)
        return solver

# def thrust_allocation(u):
#     T, Mx, My, Mz = u[0], u[1], u[2], u[3]
#     T_max = 28.0
#     l = 0.15
#     kf = 6.11e-2
#     km = 1.5e-1
#     B = np.array([[kf, kf, kf, kf],
#                   [0, l*kf, 0, -l*kf],
#                   [-l*kf, 0, l*kf, 0],
#                   [km, -km, km, -km]])
#     FM = np.array([T, Mx, My, Mz])
#     f = np.linalg.solve(B, FM)
#     f = np.clip(f, 0, T_max/4)
#     return f

def euler2quan(phi, theta, psi):
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qx = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    q = np.array([qw, qx, qy, qz])
    return q

if __name__ == '__main__':
    N = 30
    horizon = 3.0
    model = QuadrotorModel(horizon, N)
    mpc_solver = MPCSolver(model)
    nx = model.nx
    nu = model.nu

    sim_time = 20.0
    Ts = 0.01
    steps = int(sim_time / Ts)
    x_sim = np.zeros([nx-1, steps + 1])
    u_sim = np.zeros([nu, steps])
    x0 = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_sim[:, 0] = x0.copy()
    phi, theta, psi = x0[3], x0[4], x0[5]
    q = euler2quan(phi, theta, psi)
    x0 = np.concatenate((x0[0:3], q, x0[6:12]), axis=0)
    X_init = np.tile(x0.reshape(-1,1), (1, N + 1))
    U_init = np.zeros([nu, N])
    U_init[0, :] = model.m * model.g
    init_guess = np.concatenate([X_init.flatten(order='F'), U_init.flatten(order='F')])

    t_vec = Ts * np.linspace(0, steps, steps+1)
    ref_traj = np.concatenate((5.0 * np.cos(0.2 * t_vec.reshape(1, -1)),
                               5.0 * np.sin(0.2 * t_vec.reshape(1, -1)),
                               -10.0 - 2.0 * np.sin(0.1 * t_vec.reshape(1, -1)),
                               np.arctan2(1.0 * np.cos(0.2 * t_vec.reshape(1, -1)), -1.0 * np.sin(0.2 * t_vec.reshape(1, -1)))), axis=0)
    # ref_traj = np.concatenate((1.0 * np.ones([1, t_vec.shape[0]]),
    #                            1.2 * t_vec.reshape(1, -1),
    #                            10.0 * np.ones([1, t_vec.shape[0]]),
    #                            0.0 * np.ones([1, t_vec.shape[0]])), axis=0)
    # ref_traj = np.concatenate((2 * np.cos(0.2 * t_vec.reshape(1, -1)),
    #                            2 * np.sin(0.2 * t_vec.reshape(1, -1)),
    #                            10.0 + 0.5 * np.sin(0.1 * t_vec.reshape(1, -1)),
    #                            0.0 * np.ones([1, t_vec.shape[0]])), axis=0)

    start_time = time.time()
    for i in range(steps):
        ref_now = ref_traj[:,i]
        p_val = np.concatenate([x0, ref_now.flatten(order='F')])
        start_mpc = time.perf_counter()
        sol = mpc_solver.solver(x0=init_guess,
                                p=p_val.reshape(-1),
                                lbg=model.lbg, ubg=model.ubg,
                                lbx=model.lbx, ubx=model.ubx)
        end_mpc = time.perf_counter()
        print(f"mpc: {end_mpc - start_mpc:.6f} 秒")
        w_opt = sol['x'].full().flatten()
        X_opt = w_opt[:nx * (N + 1)].reshape((nx, N + 1), order='F')
        U_opt = w_opt[nx * (N + 1):].reshape((nu, N), order='F')
        x0 = model.step_update(x0, U_opt, Ts)
        x0 = np.array(x0).reshape(-1)

        q0, q1, q2, q3 = x0[3], x0[4], x0[5], x0[6]
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 * q2 + q3 * q3)))
        theta = np.arcsin(2 * (q0 * q2 - q1 * q3))
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), (1 - 2 * (q1 * q1 + q2 * q2)))
        x_sim[:, i+1] = np.concatenate((x0[0:3], np.array([phi, theta, psi]), x0[7:13]), axis=0)

        u_sim[:, i] = U_opt[:,0]
        # u_sim[:, i] = thrust_allocation(U_opt[:, 0])
        # x_sim[:, i+1] = x0

        X_init_shift = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_init_shift = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
        init_guess = np.concatenate([X_init_shift.flatten(order='F'), U_init_shift.flatten(order='F')])

        if i % 50 == 0:
            print(f"Sim time {i * Ts:.2f}s / {sim_time:.1f}s")

    end_time = time.time()
    print(end_time - start_time)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_sim[0, :], x_sim[1, :], x_sim[2, :])
    ax.plot(ref_traj[0, :], ref_traj[1, :], ref_traj[2, :], 'r--')

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[0, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("x")
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[1, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("y")
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[2, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("z")
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[5, :].reshape(-1) * 180.0 / np.pi)
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[3, :].reshape(-1) * 180.0 / np.pi)
    plt.xlabel("t(s)")
    plt.ylabel("psi")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[3, :].reshape(-1)*180.0/np.pi)
    plt.xlabel("t(s)")
    plt.ylabel("phi")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[4, :].reshape(-1)*180.0/np.pi)
    plt.xlabel("t(s)")
    plt.ylabel("theta")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[5, :].reshape(-1)*180.0/np.pi)
    plt.xlabel("t(s)")
    plt.ylabel("psi")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[6, :].reshape(-1) )
    plt.xlabel("t(s)")
    plt.ylabel("vx")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[7, :].reshape(-1) )
    plt.xlabel("t(s)")
    plt.ylabel("vy")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[8, :].reshape(-1) )
    plt.xlabel("t(s)")
    plt.ylabel("vz")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(steps) * Ts, u_sim[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T1")
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(steps) * Ts, u_sim[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T2")
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(steps) * Ts, u_sim[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T3")
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(steps) * Ts, u_sim[3, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T4")

    plt.show()


