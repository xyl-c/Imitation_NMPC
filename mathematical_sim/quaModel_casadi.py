import casadi as ca
import numpy as np


class QuadrotorModel(object):
    def __init__(self, horizon, N):
        self.Ts = horizon/N
        self.N = N
        self.g = 9.79136
        self.m = 1.287
        self.I = np.diag([
            0.00416,
            0.004855,
            0.005544
        ])


        self.nx = 13
        self.nu = 4

        self.f_max = 40.0
        self.f_min = 0.0
        self.mx_max = 0.5
        self.mx_min = -0.5
        self.my_max = 0.5
        self.my_min = -0.5
        self.mz_max = 0.216
        self.mz_min = -0.216

        self.vx_max = 5.0
        self.vx_min = -5.0
        self.vy_max = 5.0
        self.vy_min = -5.0
        self.vz_max = 5.0
        self.vz_min = -5.0

        self.wx_max = 1.0
        self.wx_min = -1.0
        self.wy_max = 1.0
        self.wy_min = -1.0
        self.wz_max = 1.0
        self.wz_min = -1.0


        self.Q = np.diag([200., 200., 500., 200.])
        self.R = np.diag([1., 300., 500.0, 300.0])

        px_s, py_s, pz_s = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        qw_s, qx_s, qy_s, qz_s = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        vx_s, vy_s, vz_s = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        wx_s, wy_s, wz_s = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        x = ca.vertcat(px_s, py_s, pz_s, qw_s, qx_s, qy_s, qz_s, vx_s, vy_s, vz_s, wx_s, wy_s, wz_s)
        u = ca.SX.sym('u', 4)       # [T, Mx, My, Mz]
        qw, qx, qy, qz = x[3], x[4], x[5], x[6]
        R = ca.vertcat(
            ca.horzcat(1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw),
            ca.horzcat(2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw),
            ca.horzcat(2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2)
        )

        wx, wy, wz = x[10], x[11], x[12]
        Omega = ca.vertcat(
            ca.horzcat(0, -wx, -wy, -wz),
            ca.horzcat(wx, 0, wz, -wy),
            ca.horzcat(wy, -wz, 0, wx),
            ca.horzcat(wz, wy, -wx, 0)
        )
        q_dot = 0.5 * Omega @ x[3:7]

        T, Mx, My, Mz = u[0], u[1], u[2], u[3]
        # f1, f2, f3, f4 = u[0], u[1], u[2], u[3]
        # T = f1 + f2 + f3 + f4
        # Mx = self.l * ( -f1 - f2 + f3 + f4 )/np.sqrt(2)
        # My = self.l * ( f1 - f2 - f3 + f4 )/np.sqrt(2)
        # Mz = self.k_m * ( -f1 + f2 - f3 + f4 )

        Fg = ca.vertcat(0, 0, self.m * self.g)
        F_thrust = R @ ca.vertcat(0, 0, -T)
        acc = (Fg + F_thrust) / self.m

        omega = x[10:13]
        I_ca = ca.DM(self.I)
        I_inv_ca = ca.DM(np.linalg.inv(self.I))
        M = ca.vertcat(Mx, My, Mz)
        omega_dot = I_inv_ca @ (M - ca.cross(omega, I_ca @ omega))

        xdot = ca.vertcat(x[7:10], q_dot, acc, omega_dot)
        self.f_c = ca.Function('f', [x, u], [xdot])

        X0 = ca.SX.sym('X0', self.nx)
        U0 = ca.SX.sym('U0', self.nu)
        k1 = self.f_c(X0, U0)
        k2 = self.f_c(X0 + (self.Ts / 2.0) * k1, U0)
        k3 = self.f_c(X0 + (self.Ts / 2.0) * k2, U0)
        k4 = self.f_c(X0 + self.Ts * k3, U0)
        x_next = X0 + (self.Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q_next = x_next[3:7]
        q_norm = ca.sqrt(ca.sumsqr(q_next) + 1e-12)
        q_next_normalized = q_next / q_norm
        x_next = ca.vertcat(x_next[0:3], q_next_normalized, x_next[7:13])
        self.f = ca.Function('f', [X0, U0], [x_next])

        self.lbg = np.zeros(self.nx * (N + 1))
        self.ubg = np.zeros(self.nx * (N + 1))

        self.lbx = []
        self.ubx = []
        for _ in range(N + 1):
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)
            self.lbx.append(-np.inf)
            self.ubx.append(np.inf)

            self.lbx.append(self.vx_min)
            self.ubx.append(self.vx_max)
            self.lbx.append(self.vy_min)
            self.ubx.append(self.vy_max)
            self.lbx.append(self.vz_min)
            self.ubx.append(self.vz_max)
            self.lbx.append(self.wx_min)
            self.ubx.append(self.wx_max)
            self.lbx.append(self.wy_min)
            self.ubx.append(self.wy_max)
            self.lbx.append(self.wz_min)
            self.ubx.append(self.wz_max)
        for _ in range(N):
            self.lbx.append(self.f_min)
            self.ubx.append(self.f_max)
            self.lbx.append(self.mx_min)
            self.ubx.append(self.mx_max)
            self.lbx.append(self.my_min)
            self.ubx.append(self.my_max)
            self.lbx.append(self.mz_min)
            self.ubx.append(self.mz_max)

    def step_update(self, x_k, u_k, Ts):
        X0 = ca.SX.sym('X0', self.nx)
        U0 = ca.SX.sym('U0', self.nu)
        k1 = self.f_c(X0, U0)
        k2 = self.f_c(X0 + (Ts / 2.0) * k1, U0)
        k3 = self.f_c(X0 + (Ts / 2.0) * k2, U0)
        k4 = self.f_c(X0 + Ts * k3, U0)
        x_next = X0 + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q_next = x_next[3:7]
        q_norm = ca.sqrt(ca.sumsqr(q_next) + 1e-12)
        q_next_normalized = q_next / q_norm
        x_next = ca.vertcat(x_next[0:3], q_next_normalized, x_next[7:13])
        f = ca.Function('f', [X0, U0], [x_next])

        s_kp1 = f(x_k, u_k[:, 0])
        return s_kp1


