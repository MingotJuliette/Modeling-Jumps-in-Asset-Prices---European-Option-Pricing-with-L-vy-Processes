import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, norm
from scipy.special import factorial

# ============================================================
# 1. Initialisation of the parameters
# ============================================================
def automatic_init_merton(logr, thr=3.0):
    """
    Return an initial guess for (c, sigma, lam, muJ, sigmaJ)
    using for pour MLE et ECF.
    """
    logr = np.asarray(logr)
    c0 = np.median(logr)
    sigma_rob = 1.4826 * np.median(np.abs(logr - c0))

    # Détection des sauts
    jumps = logr[np.abs(logr - c0) > thr * sigma_rob]
    lam0 = max(len(jumps)/len(logr), 1e-6)

    # Paramètres des sauts
    if len(jumps) >= 2:
        muJ0 = np.mean(jumps)
        sigmaJ0 = np.std(jumps, ddof=1)
        sigmaJ0 = max(sigmaJ0, 1e-3)
    else:
        muJ0 = -0.02 if c0 >= 0 else 0.02
        sigmaJ0 = max(0.05, 0.5*sigma_rob)

    # Sigma diffusion
    sigma0 = np.std(logr, ddof=1)
    if lam0 > 0.05:
        sigma0 = max(0.8*sigma0, 1e-6)

    return np.array([c0, sigma0, lam0, muJ0, sigmaJ0], dtype=float)

# ============================================================
# 2. caracteristique function merton
# ============================================================
def phi_merton(u, c, sigma, lam, muJ, sigmaJ):
    jump_term = np.exp(1j*u*muJ - 0.5*sigmaJ**2 * u**2)
    return np.exp(1j*u*c - 0.5*sigma**2*u**2 + lam*(jump_term - 1))

def ecf_empirical(u, r):
    return np.mean(np.exp(1j * np.outer(u, r)), axis=1)

def ecf_objective(params, u, r, wfun=lambda u: 1.0/(1.0 + u**2)):
    c, sigma, lam, muJ, sigmaJ = params
    if sigma <= 0 or lam < 0 or sigmaJ <= 0:
        return 1e10
    phi_hat = ecf_empirical(u, r)
    phi_theo = phi_merton(u, c, sigma, lam, muJ, sigmaJ)
    diff = np.abs(phi_hat - phi_theo)**2
    return np.sum(diff * wfun(u))

def estimate_merton_ecf(logr, u):
    x0 = automatic_init_merton(logr)
    result = minimize(
        ecf_objective, x0, args=(u, logr),
        method="Nelder-Mead",
        options={"maxiter":5000, "fatol":1e-9, "xatol":1e-9}
    )
    c, sigma, lam, muJ, sigmaJ = result.x
    sigma, sigmaJ, lam = abs(sigma), abs(sigmaJ), max(lam, 0.0)
    return np.array([c, sigma, lam, muJ, sigmaJ]), result

# ============================================================
# 3. density of merton and mle
# ============================================================
def merton_pdf(x, c, sigma, lam, muJ, sigmaJ, n_max=50):
    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)
    for n in range(n_max+1):
        w_n = np.exp(-lam) * lam**n / factorial(n)
        mu_n = c + n*muJ
        var_n = sigma**2 + n*sigmaJ**2
        pdf += w_n * norm.pdf(x, mu_n, np.sqrt(var_n))
    return pdf

def merton_neg_loglik(params, data):
    c, sigma, lam, muJ, sigmaJ = params
    if sigma <= 0 or sigmaJ <= 0 or lam < 0:
        return 1e12
    f = merton_pdf(data, c, sigma, lam, muJ, sigmaJ)
    f = np.maximum(f, 1e-300)
    return -np.sum(np.log(f))

def estimate_merton_mle(logr):
    x0 = automatic_init_merton(logr)
    result = minimize(
        merton_neg_loglik, x0, args=(logr,),
        method="Nelder-Mead",
        options={"maxiter":4000, "fatol":1e-10, "xatol":1e-10}
    )
    return result.x, result

# ============================================================
# 4. Simulation and KS
# ============================================================
def simulate_merton(n, params, random_state=None):
    rng = np.random.default_rng(random_state)
    c, sigma, lam, muJ, sigmaJ = params
    Z = rng.standard_normal(n)
    X = c + sigma*Z
    N = rng.poisson(lam, size=n)
    jumps = np.array([rng.normal(muJ, sigmaJ, size=k).sum() for k in N])
    return X + jumps

def ks_distance_merton(logr, params, n_sim=10000, random_state=123):
    X = simulate_merton(n_sim, params, random_state=random_state)
    return ks_2samp(logr, X).statistic