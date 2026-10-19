import numpy as np
from scipy.integrate import solve_ivp

def coe_to_eci(mu, a, e, i, raan, omega, nu):
    # r and v in perifocal (PQW) frame
    p = a * (1.0 - e**2)
    cnu, snu = np.cos(nu), np.sin(nu)
    r_mag = p / (1.0 + e * cnu)

    r_pqw = r_mag * np.array([cnu, snu, 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    # rotation matrix from PQW to ECI
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                  ci],
    ])

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    return r_eci, v_eci

class PropagateSatelliteSecondMan():
    # -- Constants --
    mu_e    = 398600.44    # km**3/s**2
    R_e     = 6378.137     # km
    J2      = 1.08264e-3

    # -- Dimensions -- 
    STATE_DIM       = 6     # dynamical state [r, v]
    PARAM_DIM       = 10    # estimation vector [r, v, dv1, t1]
    # MEASUREMENT_DIM = 2     # azimuth, elevation
    MEASUREMENT_DIM = 3     # 3-component LOS vector

    nu = 1e-3   # Convergence limit
    i_max = 10  # Iteration limit

    sigma_noise = 1e-5 # Noise during simulation point generation
    R_meas = sigma_noise**2 * np.eye(MEASUREMENT_DIM)   # 2 x 2 measurement covariance matrix

    tol_scale = np.array([10., 10., 10., 1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3, 50.])

    def __init__(self, 
                 num_measurement, 
                 time_split, 
                 P_0_init,
                 observer_coe=(500, 0.01, 45.05, 29.93, 132.9, -107.74),
                 target_coe=(1000, 0.02, 45.0, 94.80, 199.00, -54.13),
                 t1_truth = 905.0, 
                 t1_guess = 955.0,
                 dv_truth=(0.010, 0.010, 0.010),   # km/s  (= 10 m/s per axis, Zhou)
                 dv_guess=(0.015, 0.015, 0.015),
                 t0 = 0,
                 t2 = 1800,
                 rdm_seed = 2,
                 record = True,
                 j2_partials = True):

        # if not (t0 < t1_truth < t2):
        #     raise ValueError(f"t1_truth={t1_truth} outside arc [{t0}, {t2}]")
        
        self.record = record
        self.time_split = time_split
        self.special_rec = []
        self.results = []
        random = np.random.default_rng(rdm_seed)

        self.atol_traj = 1e-9
        self.atol_stt = np.concatenate([
            np.full(self.STATE_DIM, 1e-9),
            np.full(self.STATE_DIM**2, 1e-9),
            np.full(self.STATE_DIM**3, 1e-8),
        ])
        self.j2_partials = j2_partials

        self.K = num_measurement + 1
        self.t_eval = np.arange(0, (self.K) * time_split, time_split)
        self.t0 = self.t_eval[0]
        self.t2 = self.t_eval[-1]

        self.t1_truth = t1_truth
        self.dv_truth = np.asarray(dv_truth)

        # Observer truth - no maneouvre
        a_obs = (observer_coe[0] + self.R_e) / (1.0 + observer_coe[1])
        r_obs, v_obs = coe_to_eci(self.mu_e, a_obs, observer_coe[1], *np.deg2rad(observer_coe[2:]))
        self.observer = self.propagate_states(r_obs, v_obs)

        # Target truth - with 
        a_tgt = (target_coe[0] + self.R_e) / (1.0 + target_coe[1])
        r_tgt, v_tgt = coe_to_eci(self.mu_e, a_tgt, target_coe[1], *np.deg2rad(target_coe[2:]))
        self.target = self.propagate_states(r_tgt, v_tgt, dv=self.dv_truth, t1=self.t1_truth)

        self.X_0 = np.array(self.target[0] + [10, 10, 10, 1e-3, 1e-3, 1e-3])

        # Simulated angle measurements with noise
        self.z_tau = np.zeros((self.K, self.MEASUREMENT_DIM))
        for i in range(self.K):
            self.z_tau[i] = self.state_to_losvec(self.target[i], self.observer[i])
            self.z_tau[i] += random.normal(0.0, self.sigma_noise, self.MEASUREMENT_DIM)

        # Truth angle measurements 
        self.z_exp = np.zeros((self.K, self.MEASUREMENT_DIM))

        # Initial parameter estimation X_0
        # t1 in the middle 
        x0_init_guess = self.X_0

        if t1_guess is None:
            t1_guess = 0.5 * (self.t0 + self.t2)
        self.X_i = np.concatenate([x0_init_guess, np.asarray(dv_guess, float), [float(t1_guess)]])

        # feeds the weight matrix
        self.P_w      = P_0_init.copy()   
        # reported uncertainty only, not iterated back
        self.P_report = P_0_init.copy()

        # Covariance matrix 
        self.R_2 = self.sigma_noise**2 * np.eye(self.K * self.MEASUREMENT_DIM)
        
        if self.record:
            self.results.append([0, *self.X_i])

    # Turns r, v state vector into azimuth/elevation
    def state_to_aziel(self, x_target, x_observer):
        r_obs = x_observer[:3]
        v_obs = x_observer[3:6]
        rel = x_target[:3] - r_obs

        R_hat = r_obs / np.linalg.norm(r_obs)
        H = np.cross(r_obs, v_obs)
        C_hat = H / np.linalg.norm(H)
        T_hat = np.cross(C_hat, R_hat)

        R_orbital = np.vstack((T_hat, C_hat, R_hat))
        rel_local = R_orbital @ rel

        x, y, z = rel_local
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / np.linalg.norm(rel_local), -1.0, 1.0))

        return np.array([azimuth, elevation])
    
    # Turn r, v state to LOS vector
    def state_to_losvec(self, x_target, x_observer):
        rel = x_target[:3] - x_observer[:3]        
        l = rel / np.linalg.norm(rel)
        return l
        
    # J2 dynamics-inclusive dynamics for spacecraft
    def dynamics(self, t, state):
        r = state[:3]
        v = state[3:]

        # J2 Perturbation
        a_j2r = [self.mu_e * r[0] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[1] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[2] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-9/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2)
                ]
        
        norm_r = np.linalg.norm(r)
        a = -self.mu_e * r/norm_r**3 + a_j2r
        return np.concatenate([v, a])

    #y_ref first differential (Eq. 10)
    def calc_first_diff(self, x):
        r = x[:3]

        rn2 = r @ r
        rn = np.sqrt(rn2)
        I3 = np.eye(3)
        dadr = -self.mu_e * (I3 / rn**3 - 3.0 * np.outer(r, r) / rn**5)
        if self.j2_partials:
            dadr = dadr + self._dadr_j2(r)
        A = np.zeros((6, 6))
        A[:3, 3:] = I3
        A[3:, :3] = dadr

        return A
    
    # y_ref second differential (Eq. 10)
    def calc_second_diff(self, x):
        H = np.zeros((6, 6, 6))
        H[3:, :3, :3] = self._d2adr2(x[:3])
        return H

    # LINEAR: used to solve for phi when integrating
    def combined_dynamics(self, t, y):
        x = y[:self.STATE_DIM]
        phi_flat = y[self.STATE_DIM:]
        phi = phi_flat.reshape((self.STATE_DIM, self.STATE_DIM))

        dxdt = self.dynamics(t, x)
        A = self.calc_first_diff(x)
        dphidt = A @ phi

        return np.hstack((dxdt, dphidt.flatten()))
    
     # SECOND: used to solve for phi when integrating
    def combined_dynamics_stt(self, t, y):
        x = y[:self.STATE_DIM]
        phi_flat = y[self.STATE_DIM:self.STATE_DIM + self.STATE_DIM**2]
        psi_flat = y[self.STATE_DIM + self.STATE_DIM**2:]

        phi = phi_flat.reshape((self.STATE_DIM, self.STATE_DIM))
        psi = psi_flat.reshape((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM))

        dxdt = self.dynamics(t, x)
        A = self.calc_first_diff(x)
        H = self.calc_second_diff(x)
        dphidt = A @ phi

        dpsidt = np.einsum('ij,jmn->imn', A, psi) + np.einsum('ijk,jm,kn->imn', H, phi, phi)

        return np.hstack((dxdt, dphidt.flatten(), dpsidt.flatten()))
    
    def propagate_states(self, r0, v0, dv=None, t1=None):
        x0 = np.concatenate([r0, v0])
        if dv is None:
            sol = solve_ivp(
                self.dynamics,
                [self.t0, self.t2],
                x0,
                t_eval= self.t_eval,
                rtol=1e-11, atol=1e-12, method='DOP853'
            )
            return np.array(sol.y.T)
        
        t1 = self._clamp_t1(t1)
        pre = self.t_eval <= t1

        # real segment 1 - [t0, t1]
        te1 = np.unique(np.append(self.t_eval[pre], t1))
        sol1 = solve_ivp(self.dynamics, [self.t0, t1], x0, t_eval=te1, rtol=1e-11, atol=1e-12, method='DOP853')
        x1_minus = sol1.y[:, -1]

        # real segment 2 - [t1, t2] - w/ impulsive maneouvre
        x1_plus = x1_minus.copy()
        x1_plus[3:] += np.asarray(dv, float)

        states = np.empty((self.K, self.STATE_DIM))
        states[pre] = sol1.y[:, :pre.sum()].T if te1.size > pre.sum() else sol1.y.T

        if (~pre).any():
            sol2 = solve_ivp(self.dynamics, 
                             [t1, self.t2], 
                             x1_plus,
                             t_eval=self.t_eval[~pre],
                             rtol=1e-11, atol=1e-12, method='DOP853')
            states[~pre] = sol2.y.T
        return states

    def propagate_stt(self, X):
        r0 = X[:3]
        v0 = X[3:6]
        dv = X[6:9]
        t1 = self._clamp_t1(X[9])
        I6 = np.eye(self.STATE_DIM)
        Z666 = np.zeros((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM))

        pre = self.t_eval <= t1

        # reference segment 1 - [t0, t1] (Eq. 11) (x1_minus)
        y0_s1 = np.concatenate([r0, v0, I6.ravel(), Z666.ravel()])
        te1 = np.unique(np.append(self.t_eval[pre], t1))
        sol1 = solve_ivp(self.combined_dynamics_stt, [self.t0, t1], y0_s1, t_eval=te1, rtol=1e-9, atol=self.atol_stt, method='DOP853')
        end1 = sol1.y[:, -1]
        i_stm = self.STATE_DIM + self.STATE_DIM**2

        x1_minus = end1[:self.STATE_DIM]
        Phi_0_1 = end1[self.STATE_DIM:i_stm].reshape(self.STATE_DIM, self.STATE_DIM)
        Psi_0_1 = end1[i_stm:].reshape(self.STATE_DIM, self.STATE_DIM, self.STATE_DIM)

        # maneouvre terms of segment 1
        f_minus = self.dynamics(t1, x1_minus)
        A_minus = self.calc_first_diff(x1_minus)
        H_minus = A_minus @ Phi_0_1
        g_minus = A_minus @ f_minus

        # reference segment 2 - [t1, t2] - w/ impulsive maneouvre
        x1_plus = x1_minus.copy()
        x1_plus[3:] += dv
        y0_s2 = np.concatenate([x1_plus, I6.flatten(), Z666.flatten()])
        te2 = self.t_eval[~pre]                   
        sol2 = solve_ivp(self.combined_dynamics_stt, [t1, self.t2], y0_s2, t_eval=te2, rtol=1e-9, atol=self.atol_stt, method='DOP853')

        # maneouvre terms of segment 2
        f_plus = self.dynamics(t1, x1_plus)
        H_plus = self.calc_first_diff(x1_plus)
        g_plus = H_plus @ f_plus

        # Partial Derivative Terms (Eq. 23) (C should equal 0)
        A1 = Phi_0_1
        A2 = Psi_0_1
        B1 = f_minus - f_plus
        B2 = g_minus - g_plus + 2.0 * (H_plus @ (f_plus - f_minus))
        C = H_minus - H_plus @ A1 
        D = -H_plus

        states = np.zeros((self.K, self.STATE_DIM))
        Phi_mod = np.zeros((self.K, self.STATE_DIM, self.PARAM_DIM))
        Psi_mod = np.zeros((self.K, self.STATE_DIM, self.PARAM_DIM, self.PARAM_DIM))

        # CHECK IF THIS IS RIGHT
        i1 = 0
        i2 = 0                   
        for k, t in enumerate(self.t_eval):
            if t <= t1:
                # CHECK IF THIS IS RIGHT sol1.sol(t)
                s = sol1.y[:, i1]
                i1 += 1
                states[k] = s[:self.STATE_DIM]
                
                Phi_mod[k, :, :6] = s[self.STATE_DIM:i_stm].reshape(self.STATE_DIM, self.STATE_DIM)
                Psi_mod[k, :, :6, :6] = s[i_stm:].reshape(self.STATE_DIM, self.STATE_DIM, self.STATE_DIM)
            else:
                s = sol2.y[:, i2]
                i2 += 1
                states[k] = s[:self.STATE_DIM]
                Phi_1_t = s[self.STATE_DIM:i_stm].reshape(self.STATE_DIM, self.STATE_DIM)
                Psi_1_t = s[i_stm:].reshape(self.STATE_DIM, self.STATE_DIM, self.STATE_DIM)

                # Modified First Order STT (Eq. 29)
                Phi_mod[k, :, :6] = Phi_1_t @ A1
                Phi_mod[k, :, 6:9] = Phi_1_t[:, 3:6]
                Phi_mod[k, :, 9]  = Phi_1_t @ B1

                # Modified Second Order STT (Eq. 30) 
                # (1,1)
                Psi_mod[k, :, :6, :6] = np.einsum('ijk,jp,kq ->ipq', Psi_1_t, A1, A1) + np.einsum('ij,jpq->ipq', Phi_1_t, A2)
                # (2,2)
                Psi_mod[k, :, 6:9, 6:9] = Psi_1_t[:, 3:6, 3:6]
                # (3,3)
                Psi_mod[k, :, 9, 9] = np.einsum('ijk,j,k->i', Psi_1_t, B1, B1) + Phi_1_t @ B2
                # (3,1), (1,3)
                M = np.einsum('ijk,jp,k->ip', Psi_1_t, A1, B1) + Phi_1_t @ C
                Psi_mod[k, :, :6, 9] = M
                Psi_mod[k, :, 9, :6] = M
                # (2,1), (1,2)
                N = np.einsum('ikj,kp->ipj', Psi_1_t[:, :, 3:6], A1)
                Psi_mod[k, :, :6, 6:9] = N
                Psi_mod[k, :, 6:9, :6] = N.transpose(0, 2, 1)
                # (3,2), (2,3)
                O = (np.einsum('ijk,j->ik', Psi_1_t, B1) + Phi_1_t @ D)[:, 3:6]
                Psi_mod[k, :, 6:9, 9] = O
                Psi_mod[k, :, 9, 6:9] = O

        return states, Phi_mod, Psi_mod
    
    # Measurement Jacobian Eq. 32
    def calculate_U(self, x_target, x_observer, epsilon = 1e-4):
        U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))

        for i in range(3):  # Perturb each state component
            dx = np.zeros(6)
            dx[i] = epsilon

            h_plus = self.state_to_losvec(x_target + dx, x_observer)
            h_minus = self.state_to_losvec(x_target - dx, x_observer)

            U[:, i] = (h_plus - h_minus) / (2 * epsilon)

        return U
    
    # Second order measurement Jacobian (Eq. 32)
    def calculate_U_and_Q(self, x_target, x_observer):
        rel = x_target[:3] - x_observer[:3]
        rho = np.linalg.norm(rel)
        l = rel / rho
        I3 = np.eye(3)
 
        U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))
        U[:, :3] = (I3 - np.outer(l, l)) / rho         
 
        Q = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM, self.STATE_DIM))
        Q[:, :3, :3] = (3.0 * np.einsum('i,j,k->ijk', l, l, l)
                        - np.einsum('ij,k->ijk', I3, l)
                        - np.einsum('ik,j->ijk', I3, l)
                        - np.einsum('jk,i->ijk', I3, l)) / rho**2
        return U, Q

    def do_calc(self, print_vals=True):
        self.converged = False          
        self.converged_i = 0

        # Print initial guess
        if print_vals:
            cols = ["x(t0)", "y(t0)", "z(t0)", "x'(t0)", "y'(t0)", "z'(t0)", "dv_x", "dv_y", "dv_z", "t1"]
            print("\n" + " " * 14 + f"{'Position, km':^32}{'Velocity, m/s':^32}{'Impulsive vector, m/s':^32}{'Epoch, s':^10}")
            print(f"{'':<14}" + "".join(f"{c:>10}" for c in cols))
            print(f"{'Initial guess':<14}" + "".join(f"{v:>10.4f}" for v in self._error_row(self.X_i)))
            
        for i in range(self.i_max):
            # print("iterating")
            # Propagate reference orbit and obtain STT
            self.expected_points, phi, psi = self.propagate_stt(self.X_i)
            
            # print("predicting measurements")
            # Predicted measurements
            for j in range(self.K):
                self.z_exp[j] = self.state_to_losvec(self.expected_points[j], self.observer[j])

            # Omega_k = U_k * phi_k (Sensitivity Matrix Omega) ()
            Omega = np.zeros((self.K, self.MEASUREMENT_DIM, self.PARAM_DIM))
            Sigma = np.zeros((self.K, self.MEASUREMENT_DIM, self.STATE_DIM))
            
            # Measurement Model Partials
            U = np.zeros((self.K, self.MEASUREMENT_DIM, self.STATE_DIM))
            Q = np.zeros((self.K, self.MEASUREMENT_DIM, self.STATE_DIM, self.STATE_DIM))
            for k in range(self.K):
                U[k], Q[k] = self.calculate_U_and_Q(self.expected_points[k], self.observer[k])

            # Measurement Model Partials - Simplification (Eq. 34)
            Xi = np.einsum('tik, tkp -> tip', U, phi)
            Theta = np.einsum('tij, tjpq -> tipq', U, psi) + np.einsum('tikl, tkp, tlq -> tipq', Q, phi, phi)

            # Find difference in recorded and propagated azimuth/range measurements
            delta_z = (self.z_tau - self.z_exp).reshape((self.K * self.MEASUREMENT_DIM))
            
            # Calculate Weight Matrix W = (Omega P Omega^T + R)^-1 (Eq. 50)
            W = np.zeros((self.K * self.MEASUREMENT_DIM, self.K * self.MEASUREMENT_DIM))
            md = self.MEASUREMENT_DIM

            M = np.einsum('tiab,bc->tiac', Theta, self.P_w)
            for k in range(self.K):
                first = Xi[k] @ self.P_w @ Xi[k].T
                second = 0.5 * np.einsum('iac,jca->ij',  M[k], M[k])     # tr(Theta_i P Theta_j P)

                P_k = first + second + self.R_meas
                W[k * md:k * md + md, k * md:k * md + md] = P_k

            W = np.linalg.pinv(W)
            W = W / np.linalg.norm(W)

            # print("estimator step")
            # Estimator Step (Eq.44) 
            Omega = Xi.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM)
            Sigma = Theta.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM, self.PARAM_DIM)

            self.W_final = W
            self.Omega_final = Omega
            self.Sigma_final = Sigma

            # First order
            # scaling: km, km/s, s
            S = self.tol_scale                                  
            H_lin = Omega.T @ W @ Omega
            b_lin = Omega.T @ W @ delta_z
            delta_X_linear = S * np.linalg.solve(H_lin * np.outer(S, S), S * b_lin)

            # Second order step (Eq. 56)
            Gamma_bar = Omega + np.einsum('ikp,p->ik', Sigma, delta_X_linear)
            GtW = Gamma_bar.T @ W                               

            # Eqs. 59-60 
            Omega_bar = GtW @ Omega                             
            dZ_bar    = GtW @ delta_z                          

            Ob_s = Omega_bar * np.outer(S, S)                   
            u = S * np.linalg.solve(Ob_s, S * dZ_bar)

            # Eq. 61, Sigma_bar[u, u] = Gamma_bar^T W Sigma[u, u]
            s_uu = GtW @ np.einsum('ikp,k,p->i', Sigma, u, u)

            # Eq. 62
            corr = S * np.linalg.solve(Ob_s, S * s_uu)
            dX_hat = u - 0.5 * corr

            # if the curvature term is large, take the first order step instead
            if np.linalg.norm(0.5 * corr / S) > 0.5 * np.linalg.norm(u / S):
                dX_hat = delta_X_linear

            # # First order
            # H_lin = Omega.T @ W @ Omega
            # b_lin = Omega.T @ W @ delta_z
            # delta_X_linear = np.linalg.lstsq(H_lin, b_lin, rcond=None)[0]

            # # Gamma_lin = Omega + np.einsum('ikp,p->ik', Sigma, delta_X_linear)
            # # GtW       = Gamma_lin.T @ W
            # # dZ_lin    = -2.0 * (GtW @ delta_z)
            # # Omega_lin = -2.0 * (GtW @ Omega)
            # # Sigma_lin = -2.0 * np.einsum('ij,jmn->imn', GtW, Sigma) 

            # # Series Reversion (Eq. 62) (second order approx)
            # corr   = np.einsum('ijk,j,k->i', Sigma, delta_X_linear, delta_X_linear)
            # dX_hat = delta_X_linear - 0.5 * np.linalg.lstsq(H_lin, Omega.T @ W @ corr, rcond=None)[0]
            # Gamma_lin = Omega + np.einsum('ikp,p->ik', Sigma, dX_hat)   # for the covariance update

            # dx1 = np.linalg.lstsq(Omega_lin, dZ_lin, rcond=None)[0]
            # # kron_dx1 = np.kron(dx1, dx1)
            # # Sigma_lin_2d = Sigma_lin.reshape(Sigma_lin.shape[0], -1)
            # # corr = Sigma_lin_2d @ kron_dx1
            # # dX_hat = dx1 - 0.5 * np.linalg.lstsq(Omega_lin, corr, rcond=None)[0]

            # corr = np.einsum('ijk,j,k->i', Sigma_lin, dx1, dx1)
            # dX_hat = dx1 - 0.5 * np.linalg.lstsq(Omega_lin, corr, rcond=None)[0]
            # dX_hat = dx1 - 0.5 * np.linalg.lstsq(H_lin, Omega.T @ W @ np.einsum('ijk,j,k->i', Sigma, dx1, dx1), rcond=1)[0]

            # Clamp
            # Using std from P_0_init
            # otherwise sparse-case has / 0 problem
            # scale_step = np.array([10.0, 10.0, 10.0, 1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3, 50.0])
            # step_magnitude = np.linalg.norm(dX_hat / scale_step)
            # max_step_magnitude = 10.0

            # if step_magnitude > max_step_magnitude:
            #     dX_hat = dX_hat * (max_step_magnitude / step_magnitude)

            # Parameter update 
            self.X_i = self.X_i + dX_hat
            self.X_i[9] = self._clamp_t1(self.X_i[9])

            # Covariance update
            # self.P_i = np.linalg.pinv(Gamma_lin.T @ W @ Omega) @ Gamma_lin.T @ W @ self.R_2 @ W.T @ Gamma_lin @ np.linalg.pinv(Omega.T @ W @ Gamma_lin)
            # self.P_i = 0.5 * (self.P_i + self.P_i.T)

            Hinv = np.linalg.pinv(Omega.T @ W @ Omega)
            self.P_w = Hinv @ Omega.T @ W @ self.R_2 @ W.T @ Omega @ Hinv
            self.P_w = 0.5 * (self.P_w + self.P_w.T)

            Gamma_bar = Omega + np.einsum('ikp,p->ik', Sigma, delta_X_linear)
            self.P_report = (np.linalg.pinv(Gamma_bar.T @ W @ Omega)
                            @ Gamma_bar.T @ W @ self.R_2 @ W.T @ Gamma_bar
                            @ np.linalg.pinv(Omega.T @ W @ Gamma_bar))
            self.P_report = 0.5 * (self.P_report + self.P_report.T)
            
            # Record the iteration and difference in current estimated X_0 and actual initial guess
            if self.record:
                self.results.append([i + 1, *self.X_i])
            
            # Print estimation errors
            if print_vals:
                label = f"Iteration {i + 1}"
                print(f"{label:<14}" + "".join(f"{v:>10.4f}" for v in self._error_row(self.X_i)))

            # Convergence check
            scale = np.sqrt(np.diag(self.P_w))
            if np.linalg.norm(dX_hat / self.tol_scale) <= self.nu:
                print("Successfully converged!")
                self.converged = True
                self.converged_i = i + 1
                return
       
        if not self.converged:
            print("Converge unsuccessful")
    
    # Helper - clamp t1 between t0 and t1
    def _clamp_t1(self, t1):
        floor, ceiling = self.t0 + self.time_split, self.t2 - self.time_split
        return float(np.clip(t1, floor, ceiling))

    # Helper -  Calculate dadr J2
    def _dadr_j2(self, r):
        k = self.mu_e * self.J2 * self.R_e**2
        rn = np.sqrt(r @ r)
        z = r[2]
        e = np.array([0.0, 0.0, 1.0])
        I3 = np.eye(3)
        r5, r7, r9 = rn**-5, rn**-7, rn**-9
        result = k * (
            - 1.5 * (I3 * r5 - 5.0 * np.outer(r, r) * r7)
            - 3.0 * (np.outer(e, e) * r5 - 5.0 * z * np.outer(e, r) * r7)
            + 7.5 * (2.0 * z * np.outer(r, e) * r7 + z * z * I3 * r7
                     - 7.0 * z * z * np.outer(r, r) * r9)
        )
        return result

    # Helper - Calculate dadr2
    def _d2adr2(self, r):
        rn = np.sqrt(r @ r)
        I3 = np.eye(3)
        Ir0 = np.einsum('ij,k->ijk', I3, r)
        Ir1 = np.einsum('ik,j->ijk', I3, r)
        Ir2 = np.einsum('jk,i->ijk', I3, r)
        rrr = np.einsum('i,j,k->ijk', r, r, r)
 
        d2a = (3.0 * self.mu_e / rn**5) * (Ir0 + Ir1 + Ir2) \
            - (15.0 * self.mu_e / rn**7) * rrr

        if not self.j2_partials:
            return d2a
 
        k = self.mu_e * self.J2 * self.R_e**2
        z = r[2]
        e = np.array([0.0, 0.0, 1.0])

        r7 = rn**-7
        r9 = rn**-9
        r11 = rn**-11
 
        TA = -1.5 * (-5.0 * (Ir0 + Ir1 + Ir2) * r7 + 35.0 * rrr * r9)
        TB = -3.0 * (-5.0 * np.einsum('i,j,k->ijk', e, e, r) * r7
                    - 5.0 * (np.einsum('k,i,j->ijk', e, e, r) * r7
                    + z * np.einsum('i,jk->ijk', e, I3) * r7
                    - 7.0 * z * np.einsum('i,j,k->ijk', e, r, r) * r9))
        TC = 7.5 * (2.0 * (np.einsum('k,j,i->ijk', e, e, r) * r7
                    + z * np.einsum('j,ik->ijk', e, I3) * r7
                    - 7.0 * z * np.einsum('j,i,k->ijk', e, r, r) * r9)
                    + (2.0 * z * np.einsum('k,ij->ijk', e, I3) * r7
                        - 7.0 * z * z * Ir0 * r9)
                    - 7.0 * (2.0 * z * np.einsum('k,i,j->ijk', e, r, r) * r9
                             + z * z * (Ir1 + Ir2) * r9
                             - 9.0 * z * z * rrr * r11))
        return d2a + k * (TA + TB + TC)
    
    # Helper - estimate errors
    def _error_row(self, X):
        return np.array([
            X[0] - self.target[0, 0],                 # x(t0)  km
            X[1] - self.target[0, 1],                 # y(t0)  km
            X[2] - self.target[0, 2],                 # z(t0)  km
            (X[3] - self.target[0, 3]) * 1e3,         # xdot   m/s
            (X[4] - self.target[0, 4]) * 1e3,         # ydot   m/s
            (X[5] - self.target[0, 5]) * 1e3,         # zdot   m/s
            (X[6] - self.dv_truth[0]) * 1e3,          # dvx    m/s
            (X[7] - self.dv_truth[1]) * 1e3,          # dvy    m/s
            (X[8] - self.dv_truth[2]) * 1e3,          # dvz    m/s
            X[9] - self.t1_truth,                     # t1     s
        ])


     