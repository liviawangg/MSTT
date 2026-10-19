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

class PropagateSatelliteLinearMan():
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

    # component-wise convergence thresholds (position km, velocity km/s, epoch s)
    tol_pos = 1e-4
    tol_vel = 1e-7
    tol_dv  = 1e-7
    tol_t1  = 1e-2

    tol_scale = np.array([10., 10., 10., 1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3, 50.])

    results = []
    special_rec = []

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
                 record = False,
                 j2_partials = True):

        # if not (t0 < t1_truth < t2):
        #     raise ValueError(f"t1_truth={t1_truth} outside arc [{t0}, {t2}]")

        self.j2_partials = j2_partials

        self.record = record
        self.time_split = time_split
        self.special_rec = []
        self.results = []
        random = np.random.default_rng(rdm_seed)

        self.K = num_measurement + 1
        self.t_eval = np.arange(0, (self.K) * time_split, time_split)
        self.t0 = self.t_eval[0]
        self.t2 = self.t_eval[-1]

        self.t1_truth = t1_truth
        self.dv_truth = np.asarray(dv_truth)

        # Observer truth - no maneouvre
        a_obs = (observer_coe[0] + self.R_e) / (1.0 + observer_coe[1])
        # a_obs = observer_coe[0] + self.R_e
        r_obs, v_obs = coe_to_eci(self.mu_e, a_obs, observer_coe[1], *np.deg2rad(observer_coe[2:]))
        self.observer = self.propagate_states(r_obs, v_obs)

        # Target truth - with 
        a_tgt = (target_coe[0] + self.R_e) / (1.0 + target_coe[1])
        # a_tgt = target_coe[0] + self.R_e
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

        self.P_w      = P_0_init.copy()   # feeds the weight matrix
        self.P_report = P_0_init.copy()   # reported uncertainty only, not fed back

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

    #y_ref first differential
    def calc_first_diff(self, x, t):
        # epsilon = 1e-4
        # r = x[:3]
        # v = x[3:self.STATE_DIM]

        # dadr = np.zeros((3, 3))

        # for i in range(3):
        #     dr = np.zeros(3)
        #     dr[i] = epsilon

        #     state_plus = np.concatenate([r + dr, v])
        #     state_minus = np.concatenate([r - dr, v])

        #     a_plus = self.dynamics(0, state_plus)[3:]
        #     a_minus = self.dynamics(0, state_minus)[3:]

        #     dadr[:, i] = (a_plus - a_minus) / (2 * epsilon)

        # A = np.block([
        #     [np.zeros((3, 3)), np.eye(3)],
        #     [dadr, np.zeros((3, 3))]
        # ])

        # Analytical solution
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

    # used to solve for phi when integrating
    def combined_dynamics(self, t, y):
        x = y[:self.STATE_DIM]
        phi_flat = y[self.STATE_DIM:]
        phi = phi_flat.reshape((self.STATE_DIM, self.STATE_DIM))

        dxdt = self.dynamics(t, x)
        A = self.calc_first_diff(x, t)
        dphidt = A @ phi

        return np.hstack((dxdt, dphidt.flatten()))
    
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

    def propagate_stm(self, X):
        r0 = X[:3]
        v0 = X[3:6]
        dv = X[6:9]
        t1 = self._clamp_t1(X[9])
        I6 = np.eye(self.STATE_DIM)

        pre = self.t_eval <= t1
        self.atol_stm = np.full(self.STATE_DIM + self.STATE_DIM**2, 1e-9)

        # reference segment 1 - [t0, t1]
        y0_s1 = np.concatenate([r0, v0, I6.flatten()])
        te1 = np.unique(np.append(self.t_eval[pre], t1))
        sol1 = solve_ivp(self.combined_dynamics, [self.t0, t1], y0_s1, t_eval=te1, rtol=1e-9, atol=self.atol_stm, method='DOP853')
        # sol1 = solve_ivp(self.combined_dynamics, [self.t0, t1], y0_s1, t_eval=te1, rtol=1e-9, atol=1e-11, method='DOP853')
        end1 = sol1.y[:, -1]

        x1_minus = end1[:self.STATE_DIM]
        Phi_0_1 = end1[self.STATE_DIM:].reshape(self.STATE_DIM, self.STATE_DIM)

        # reference segment 2 - [t1, t2] - w/ impulsive maneouvre
        x1_plus = x1_minus.copy()
        x1_plus[3:] = x1_plus[3:] + dv
        y0_s2 = np.concatenate([x1_plus, I6.flatten()])
        te2 = self.t_eval[~pre]
        sol2 = solve_ivp(self.combined_dynamics, [t1, self.t2], y0_s2, t_eval=te2, rtol=1e-9, atol=self.atol_stm, method='DOP853')
        # sol2 = solve_ivp(self.combined_dynamics, [t1, self.t2], y0_s2, t_eval=te2, rtol=1e-9, atol=1e-11, method='DOP853')
        b_t1 = np.concatenate([-dv, np.zeros(3)]) # B1 = f- - f+
    

        states = np.zeros((self.K, self.STATE_DIM))
        Phi_mod = np.zeros((self.K, self.STATE_DIM, self.PARAM_DIM))

        i1 = 0
        i2 = 0    
        for k, t in enumerate(self.t_eval):
            if t <= t1:
                s = sol1.y[:, i1]
                i1 += 1
                states[k] = s[:self.STATE_DIM]
                Phi_mod[k, :, :6] = s[self.STATE_DIM:].reshape(6, 6)   # cols 7-10 = 0
            else:
                s = sol2.y[:, i2]
                i2 += 1
                states[k] = s[:self.STATE_DIM]
                Phi_1_t = s[self.STATE_DIM:].reshape(6, 6)
                Phi_mod[k, :, :6] = Phi_1_t @ Phi_0_1
                Phi_mod[k, :, 6:9] = Phi_1_t[:, 3:6]
                Phi_mod[k, :, 9]  = Phi_1_t @ b_t1
        return states, Phi_mod

    # Measurement Jacobian Eq. 32
    def calculate_U(self, x_target, x_observer, epsilon = 1e-4):
        # Finite Difference
        # U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))

        # for i in range(3):  # Perturb each state component
        #     dx = np.zeros(6)
        #     dx[i] = epsilon

        #     h_plus = self.state_to_losvec(x_target + dx, x_observer)
        #     h_minus = self.state_to_losvec(x_target - dx, x_observer)

        #     U[:, i] = (h_plus - h_minus) / (2 * epsilon)

        # Analytical Solution
        rel = x_target[:3] - x_observer[:3]
        rho = np.linalg.norm(rel)
        l = rel / rho
        U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))
        U[:, :3] = (np.eye(3) - np.outer(l, l)) / rho
        return U

        return U

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
            # Propagate reference orbit and obtain STM
            self.expected_points, phi = self.propagate_stm(self.X_i)
            
            # Predicted measurements
            for j in range(self.K):
                self.z_exp[j] = self.state_to_losvec(self.expected_points[j], self.observer[j])

            # Omega_k = U_k * phi_k (Sensitivity Matrix Omega)
            Omega = np.zeros((self.K, self.MEASUREMENT_DIM, self.PARAM_DIM))
            for k in range(self.K):
                U_k = self.calculate_U(self.expected_points[k], self.observer[k])
                Omega[k] = U_k @ phi[k]

            # Find difference in recorded and propagated azimuth/range measurements
            delta_z = (self.z_tau - self.z_exp).reshape((self.K * self.MEASUREMENT_DIM))
            
            # Calculate Weight Matrix W = (Omega P Omega^T + R)^-1
            W = np.zeros((self.K * self.MEASUREMENT_DIM, self.K * self.MEASUREMENT_DIM))
            md = self.MEASUREMENT_DIM
            for l in range(self.K):
                # previous change -> P_i to P_w in this loop
                P_w = np.einsum('ia, jq, aq -> ij', Omega[l, :, :], Omega[l, :, :], self.P_w) + self.R_meas
                W[l * md:l * md + md, l * md:l * md + md] = P_w
            W = np.linalg.pinv(W)
            W = W / np.linalg.norm(W)

            # Estimator Step 
            Omega_f = Omega.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM)
            H = Omega_f.T @ W @ Omega_f
            b = Omega_f.T @ W @ delta_z
            delta_X_linear = np.linalg.lstsq(H, b, rcond=None)[0]
            self.W_final = W
            self.Omega_final = Omega_f

            # Parameter update
            self.X_i = self.X_i + delta_X_linear
            self.X_i[9] = self._clamp_t1(self.X_i[9])

            # Covariance update
            # P_dx = np.linalg.pinv(Omega_f.T @ W @ Omega_f) @ Omega_f.T @ W @ self.R_2 @ W.T @ Omega_f @ np.linalg.pinv(Omega_f.T @ W @ Omega_f)
            # self.P_i = P_dx
            Omega = Omega_f
            Hinv = np.linalg.pinv(Omega.T @ W @ Omega)
            self.P_w = Hinv @ Omega.T @ W @ self.R_2 @ W.T @ Omega @ Hinv
            self.P_w = 0.5 * (self.P_w + self.P_w.T)

            self.P_report = self.P_w.copy()
            # Record the iteration and difference in current estimated X_0 and actual initial guess
            if self.record:
                self.results.append([i + 1, *self.X_i])
            
            # Print estimation errors
            if print_vals:
                label = f"Iteration {i + 1}"
                print(f"{label:<14}" + "".join(f"{v:>10.4f}" for v in self._error_row(self.X_i)))

            # Convergence check
            scale = np.sqrt(np.diag(self.P_w))
            if np.linalg.norm(delta_X_linear / self.tol_scale) <= self.nu:
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
    
    # Helper - estimate errors
    def _error_row(self, X):
        return np.array([
            X[0] - self.target[0, 0],                 # x(t0)  km
            X[1] - self.target[0, 1],                 # y(t0)  km
            X[2] - self.target[0, 2],                 # z(t0)  km
            (X[3] - self.target[0, 3]) * 1e3,         # xdot   m/s
            (X[4] - self.target[0, 4]) * 1e3,         # ydot   m/s
            (X[5] - self.target[0, 5]) * 1e3,         # zdot   m/s
            (X[6] - self.dv_truth[0]) * 1e3,           # dvx    m/s
            (X[7] - self.dv_truth[1]) * 1e3,           # dvy    m/s
            (X[8] - self.dv_truth[2]) * 1e3,           # dvz    m/s
            X[9] - self.t1_truth,                      # t1     s
        ])

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
