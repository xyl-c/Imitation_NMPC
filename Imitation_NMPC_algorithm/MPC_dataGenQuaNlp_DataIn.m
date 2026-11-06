clc; clear;
tic;

import casadi.*
casadi_mex_path = '';   % ??: 'C:\casadi\matlab\casadiMEX.mexw64'
out_csv = 'QUA1e6_parallel_hybrid_2.csv';

samples = readmatrix('in_data_1e6_hybrid_2.csv');
n_train = height(samples);
q = eul2qua(samples);
scaled_sample = [samples(:, 1:3), q, samples(:, 7:12), zeros(n_train, 3), samples(:, end)];

data_save = zeros(n_train, 17 + 4);
fprintf('data2');

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');  
end

files_to_attach = {};
if ~isempty(casadi_mex_path)
    files_to_attach{end+1} = casadi_mex_path;
end
% files_to_attach{end+1} = dynamics_file;

try
    addAttachedFiles(pool, files_to_attach);
catch ME
    warning('addAttachedFiles ???%s\n?? worker ????????', ME.message);
end

solverConst = parallel.pool.Constant(@init_solver_on_worker);
dq = parallel.pool.DataQueue;
afterEach(dq, @(~) progressPrinter());
fprintf('???????????%d\n', n_train);

parfor idx = 1:n_train
    solve_in = scaled_sample(idx,:);
    solver = solverConst.Value;
    try
        u_all = solver([solve_in(1:13)'; solve_in(14:17)']);
        u0 = u_all(:, 1);
    catch
        solve_in = zeros(1,17);
        u0 = [0; 0; 0; 0];
    end
    data_save(idx, :) = [solve_in, u0'];
    send(dq, 0);  
end

non_zero_rows = any(data_save, 2);  
data_save_cleaned = data_save(non_zero_rows, :);
samples_cleaned = samples(non_zero_rows, :);
data_save = [data_save_cleaned(:,1:3), samples_cleaned(:,4:6), data_save_cleaned(:,8:21)];
writematrix(data_save, out_csv);
fprintf('??????????????%s\n', out_csv);
delete(solverConst); 
totletime = toc;
disp(totletime);

function solver_handle = init_solver_on_worker()
    import casadi.*
    N = 30;
    horizon = 3.0;
    Ts = horizon/N;
    
    Q = diag([200.0, 200.0, 500.0, 200.0]);  
    R = diag([1.0, 300.0, 500.0, 300.0]);


    x_sym = SX.sym('x', 13, 1);
    u_sym = SX.sym('u', 4, 1);

    px = x_sym(1); py = x_sym(2); pz = x_sym(3);
    qw = x_sym(4); qx = x_sym(5); qy = x_sym(6); qz = x_sym(7);
    vx = x_sym(8); vy = x_sym(9); vz = x_sym(10);
    wx = x_sym(11); wy = x_sym(12); wz = x_sym(13);
    T = u_sym(1); Mx = u_sym(2); My = u_sym(3); Mz = u_sym(4);

    g = 9.79136;
    m = 1.287;
    I_mat = diag([0.00416, 0.004855, 0.005544]);
    l = 0.225;
    k_m = 0.2;

    Rmat = [1 - 2*qy^2 - 2*qz^2, 2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw;
            2*qx*qy + 2*qz*qw, 1 - 2*qx^2 - 2*qz^2,   2*qy*qz - 2*qx*qw;
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw,   1 - 2*qx^2 - 2*qy^2];

    Omega = SX.zeros(4,4);
    Omega(1,2:4) = [-wx, -wy, -wz];
    Omega(2,1) = wx; Omega(2,3) = wz; Omega(2,4) = -wy;
    Omega(3,1) = wy; Omega(3,2) = -wz; Omega(3,4) = wx;
    Omega(4,1) = wz; Omega(4,2) = wy; Omega(4,3) = -wx;
    qvec = [qw; qx; qy; qz];
    q_dot = 0.5 * Omega * qvec;

    % T = f1 + f2 + f3 + f4;
    % Mx = l * ( -f1 - f2 + f3 + f4 )/sqrt(2);
    % My = l * ( f1 - f2 - f3 + f4 )/sqrt(2);
    % Mz = k_m * ( -f1 + f2 - f3 + f4 );

    Fg = SX([0;0; m*g]);
    F_thrust = Rmat * SX([0; 0; -T]);
    acc = (Fg + F_thrust) / m;

    omega = SX([wx; wy; wz]);
    omega_dot = inv(SX(I_mat)) * (SX([Mx;My;Mz]) - cross(omega, SX(I_mat)*omega));

    xdot_sym = SX.zeros(13,1);
    xdot_sym(1:3) = SX([vx; vy; vz]);
    xdot_sym(4:7) = q_dot;
    xdot_sym(8:10) = acc;
    xdot_sym(11:13) = omega_dot;

    f_cas = Function('f_cas',{x_sym, u_sym},{xdot_sym});

    k1 = f_cas(x_sym, u_sym);
    k2 = f_cas(x_sym + (Ts/2)*k1, u_sym);
    k3 = f_cas(x_sym + (Ts/2)*k2, u_sym);
    k4 = f_cas(x_sym + Ts*k3, u_sym);
    x_next = x_sym + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);

    q_next = x_next(4:7);
    qnorm = sqrt(sum(q_next.*q_next) + 1e-12);
    q_next_normalized = q_next / qnorm;
    x_next(4:7) = q_next_normalized;

    RK4_step = Function('RK4_step',{x_sym,u_sym},{x_next});

    
    X = SX.sym('X', 13, N+1);
    U = SX.sym('U', 4, N);
    X_ref = SX.sym('X_ref', 4);
    X0_param = SX.sym('X0_param', 13);

    J = 0;
    g_c = [];

    for k = 1:N
        x_next_pred = RK4_step(X(:,k), U(:,k));
        g_c = [g_c; (X(:, k + 1) - x_next_pred)];

        e_pos = X(1:3,k) - X_ref(1:3);

        q0 = X(4,k); q1 = X(5,k); q2 = X(6,k); q3 = X(7,k);
        psi_k = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
        psi_e = atan2(sin(psi_k - X_ref(4)), cos(psi_k - X_ref(4)));

        Xe = [e_pos; psi_e];

        J = J + Xe' * Q * Xe + U(:,k)' * R * U(:,k);

        % if k > 1
        %     du = U(:,k) - U(:,k-1);
        %     J = J + du' * R_du * du;
        % end
    end
    e_pos_end = X(1:3,N+1) - X_ref(1:3);
    q0 = X(4,N+1); q1 = X(5,N+1); q2 = X(6,N+1); q3 = X(7,N+1);
    psi_k = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
    psi_e = atan2(sin(psi_k - X_ref(4)), cos(psi_k - X_ref(4)));
    Xe_terminal = [e_pos_end; psi_e];
    J = J + Xe_terminal' * Q * Xe_terminal;

    OPT_variables = [reshape(X, 13*(N+1), 1); reshape(U, 4*N, 1)];
    g_c = [g_c; (X(:, 1) - X0_param)];

    nlp = struct('x', OPT_variables, 'f', J, 'g', g_c, 'p', [X0_param; X_ref]);
    opts = struct;
    % opts.ipopt.max_iter = 200;
    opts.ipopt.print_level = 0;
    opts.print_time = false;
    solver = nlpsol('solver','ipopt', nlp, opts);
    
    solver_handle = @(x_param)solve_with_guess(x_param);

    % solver_handle = @(x0, ref)solve_with_guess(opti, X0, ref, U, X, x0, ref);

    function Usol = solve_with_guess(x_param)
        f_max = 40; f_min = 0;
        mx_max = 0.5; mx_min = -0.5;
        my_max = 0.5; my_min = -0.5;
        mz_max = 0.216; mz_min = -0.216;
        vx_max = 5.0; vx_min = -5.0;
        vy_max = 5.0; vy_min = -5.0;
        vz_max = 5.0; vz_min = -5.0;
        wx_max = 1.0; wx_min = -1.0;
        wy_max = 1.0; wy_min = -1.0;
        wz_max = 1.0; wz_min = -1.0; 
        x_min = [-inf;-inf;-inf;-inf;-inf;-inf;-inf;vx_min;vy_min;vz_min;wx_min;wy_min;wz_min];
        x_max = [inf;inf;inf;inf;inf;inf;inf;vx_max;vy_max;vz_max;wx_max;wy_max;wz_max];
        u_min = [f_min;mx_min;my_min;mz_min];
        u_max = [f_max;mx_max;my_max;mz_max];
    
        x0 = x_param(1:13);
        m = 1.287;
        g = 9.79136;
        N = 30;   
        init_X = repmat(x0, N+1, 1);
        u_init =  m * g * ones(4, N);
        init_guess = [init_X; reshape(u_init, 4*N, 1)];
    
        lbx = [];
        ubx = [];
        for k = 1:N+1
            lbx = [lbx; x_min]; 
            ubx = [ubx; x_max];
        end
        for k = 1:N
            lbx = [lbx; u_min]; 
            ubx = [ubx; u_max];
        end
        
        sol = solver('x0', init_guess, ...
                     'lbx', lbx, 'ubx', ubx, ...
                     'lbg', zeros(13*(N+1),1), 'ubg', zeros(13*(N+1),1), ...
                     'p', x_param);
        solx = full(sol.x);
        Usol = reshape(solx(13*(N+1)+1:end), 4, N);
    
    end

end



function progressPrinter()
    persistent count
    if isempty(count), count = 0; end
    count = count + 1;

    if mod(count, 500) == 0  % ? 500 ??????
        fprintf('??: %d ???\n', count);
    end
end

function q = eul2qua(scaled_sample)
    qw = cos(scaled_sample(:, 4) / 2) .* cos(scaled_sample(:, 5) / 2) .* cos(scaled_sample(:, 6) / 2) + sin(scaled_sample(:, 4) / 2) .* sin(scaled_sample(:, 5) / 2) .* sin(scaled_sample(:, 6) / 2);
    qx = sin(scaled_sample(:, 4) / 2) .* cos(scaled_sample(:, 5) / 2) .* cos(scaled_sample(:, 6) / 2) - cos(scaled_sample(:, 4) / 2) .* sin(scaled_sample(:, 5) / 2) .* sin(scaled_sample(:, 6) / 2);
    qy = cos(scaled_sample(:, 4) / 2) .* sin(scaled_sample(:, 5) / 2) .* cos(scaled_sample(:, 6) / 2) + sin(scaled_sample(:, 4) / 2) .* cos(scaled_sample(:, 5) / 2) .* sin(scaled_sample(:, 6) / 2);
    qz = cos(scaled_sample(:, 4) / 2) .* cos(scaled_sample(:, 5) / 2) .* sin(scaled_sample(:, 6) / 2) - sin(scaled_sample(:, 4) / 2) .* sin(scaled_sample(:, 5) / 2) .* cos(scaled_sample(:, 6) / 2);
    q = [qw,qx,qy,qz];
end



function samples = lhs_truncated_gaussian(n, d, means, variances, lower_bounds, upper_bounds)

    if length(means) ~= d || length(variances) ~= d
        error('???????????d??');
    end
    
    if length(lower_bounds) ~= d || length(upper_bounds) ~= d
        error('?????????d??');
    end

    for i = 1:d
        if lower_bounds(i) >= upper_bounds(i)
            error('?? %d: ?? %f ?????? %f', i, lower_bounds(i), upper_bounds(i));
        end
    end

    uniform_samples = lhsdesign(n, d);

    samples = zeros(n, d);
    
    for i = 1:d
        a = lower_bounds(i);
        b = upper_bounds(i);
        mu = means(i);
        sigma = sqrt(variances(i));

        a_std = (a - mu) / sigma;
        b_std = (b - mu) / sigma;

        samples(:, i) = truncated_normal_ppf(uniform_samples(:, i), mu, sigma, a_std, b_std);
    end
end

function x = truncated_normal_ppf(p, mu, sigma, a_std, b_std)

    phi_a = normcdf(a_std);
    phi_b = normcdf(b_std);

    phi_a = max(phi_a, 1e-10);
    phi_b = min(phi_b, 1 - 1e-10);

    p_scaled = phi_a + p .* (phi_b - phi_a);

    z = norminv(p_scaled);

    x = mu + sigma * z;

    x = max(x, mu + sigma * a_std);
    x = min(x, mu + sigma * b_std);
end




