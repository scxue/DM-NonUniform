import torch
import os
import logging
import math
import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t
        
    def edm_sigma(self, t):
        return self.marginal_std(t)/self.marginal_alpha(t)
    
    def edm_inverse_sigma(self, edmsigma):
        alpha = 1 / (edmsigma ** 2 + 1).sqrt()
        sigma = alpha*edmsigma
        lambda_t = torch.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t



class StepOptim(object):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns
        self.T = 1.0 # t_T of diffusion sampling, for VP models, T=1.0; for EDM models, T=80.0

    def alpha(self, t):
        t = torch.as_tensor(t, dtype = torch.float64)
        return self.ns.marginal_alpha(t).numpy()
    def sigma(self, t):
        return np.sqrt(1 - self.alpha(t) * self.alpha(t))
    def lambda_func(self, t):
        return np.log(self.alpha(t)/self.sigma(t))
    def H0(self, h):
        return np.exp(h) - 1
    def H1(self, h):
        return np.exp(h) * h - self.H0(h)
    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)
    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)
    def inverse_lambda(self, lamb):
        lamb = torch.as_tensor(lamb, dtype = torch.float64)
        return self.ns.inverse_lambda(lamb)
    def edm_sigma(self, t):
        return np.sqrt(1./(self.alpha(t)*self.alpha(t)) - 1)
    def edm_inverse_sigma(self, edm_sigma):
        alpha = 1 / (edm_sigma*edm_sigma+1).sqrt()
        sigma = alpha*edm_sigma
        lambda_t = np.log(alpha/sigma)
        t = self.inverse_lambda(lambda_t)
        return t



    def sel_lambdas_lof_obj(self, lambda_vec, eps):

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        hv = np.zeros(N)
        for i in range(N):
            hv[i] = lambda_vec_ext[i+1] - lambda_vec_ext[i]
        elv = np.exp(lambda_vec_ext)
        emlv_sq = np.exp(-2*lambda_vec_ext)
        alpha_vec = 1./np.sqrt(1+emlv_sq)
        sigma_vec = 1./np.sqrt(1+np.exp(2*lambda_vec_ext))
        data_err_vec = (sigma_vec**2)/alpha_vec
        # for pixel-space diffusion models, we empirically find (sigma_vec**1)/alpha_vec will be better

        truncNum = 3 # For NFEs <= 7, set truncNum = 3 to avoid numerical instability; for NFEs > 7, truncNum = 0
        res = 0. 
        c_vec = np.zeros(N)
        for s in range(N):
            if s in [0, N-1]:
                n, kp = s, 1 
                J_n_kp_0 = elv[n+1] - elv[n]
                res += abs(J_n_kp_0 * data_err_vec[n])
            elif s in [1, N-2]:
                n, kp = s-1, 2
                J_n_kp_0 = -elv[n+1] * self.H1(hv[n+1]) / hv[n]
                J_n_kp_1 = elv[n+1] * (self.H1(hv[n+1])+hv[n]*self.H0(hv[n+1])) / hv[n]
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2)
            else:
                n, kp = s-2, 3  
                J_n_kp_0 = elv[n+2] * (self.H2(hv[n+2])+hv[n+1]*self.H1(hv[n+2])) / (hv[n]*(hv[n]+hv[n+1]))
                J_n_kp_1 = -elv[n+2] * (self.H2(hv[n+2])+(hv[n]+hv[n+1])*self.H1(hv[n+2])) / (hv[n]*hv[n+1])
                J_n_kp_2 = elv[n+2] * (self.H2(hv[n+2])+(2*hv[n+1]+hv[n])*self.H1(hv[n+2])+hv[n+1]*(hv[n]+hv[n+1])*self.H0(hv[n+2])) / (hv[n+1]*(hv[n]+hv[n+1]))
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                    c_vec[n+2] += data_err_vec[n+2] * J_n_kp_2
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2 + (data_err_vec[n+2] * J_n_kp_2)**2)
        res += sum(abs(c_vec))
        return res

    def get_ts_lambdas(self, N, eps, initType):
        # eps is t_0 of diffusion sampling, e.g. 1e-3 for VP models
        # initType: initTypes with '_origin' are baseline time step discretizations (without optimization)
        # initTypes without '_origin' are optimized time step discretizations with corresponding baseline
        # time step discretizations as initializations. For latent-space diffusion models, 'unif_t' is recommended.
        # For pixel-space diffusion models, 'unif' is recommended (which is logSNR initialization)

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()

        # constraints
        constr_mat = np.zeros((N, N-1)) 
        for i in range(N-1):
            constr_mat[i][i] = 1.
            constr_mat[i+1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.zeros(N)
        for i in range(N):
            ub_vec[i] = np.inf
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # initial vector
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N+1)
        elif initType in ['unif_t', 'unif_t_origin']:
            t_vec = torch.linspace(self.T, eps, N+1)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['edm', 'edm_origin']:
            rho = 7
            edm_sigma_min, edm_sigma_max = self.edm_sigma(eps).item(), self.edm_sigma(self.T).item()
            edm_sigma_vec = torch.linspace(edm_sigma_max**(1. / rho), edm_sigma_min**(1. / rho), N + 1).pow(rho)
            t_vec = self.edm_inverse_sigma(edm_sigma_vec)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['quad', 'quad_origin']:
            t_order = 2
            t_vec = torch.linspace(self.T**(1./t_order), eps**(1./t_order), N+1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        else:
            print('InitType not found!')
            return 

        if initType in ['unif_origin', 'unif_t_origin', 'edm_origin', 'quad_origin']:
                lambda_res = lambda_vec_ext
                t_res = torch.tensor(self.inverse_lambda(lambda_res))
        else: 
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(self.sel_lambdas_lof_obj, lambda_vec_init, method='trust-constr', args=(eps), constraints=[linear_constraint], options={'verbose': 1})
            lambda_res = torch.tensor(np.concatenate((np.array([lambda_T]), res.x, np.array([lambda_eps]))))
            t_res = torch.tensor(self.inverse_lambda(lambda_res))
        return t_res, lambda_res
    