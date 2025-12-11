import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp,  kurtosis

# ============================================================
# 1. Initialisation of the parameters
# ============================================================
def automatic_init_vg(logr):
    """
    Retourne un initial guess robuste pour VG : (c, sigma, theta, nu)
    """
    logr = np.asarray(logr)
    
    # Drift / location
    c0 = np.mean(logr)
    
    # Volatility robuste (MAD ajusté)
    MAD = np.median(np.abs(logr - c0))
    sigma0 = MAD / 0.6744897501960817  # Phi^-1(0.75)
    
    # Skewness / asymetry
    skew = np.mean((logr - c0)**3) / sigma0**3
    theta0 = np.sign(skew) * min(abs(skew)*sigma0, sigma0)
    
    # Kurtosis / shape nu
    r_kurt = kurtosis(logr, fisher=False)  # normal = 3
    nu0 = np.clip(sigma0**2 / max(r_kurt - 3, 0.1), 0.01, 10.0)
    
    return np.array([c0, sigma0, theta0, nu0], dtype=float)

# ============================================================
# 2. Catacteristic function VG
# ============================================================
def phi_vg(u, c, sigma, theta, nu, t=1):
    """
    Catacteristic function VG (location c)
    """
    base = 1 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2
    return np.exp(1j * c * u) * base**(-t/nu)

def ecf_empirical(u, r):
    return np.mean(np.exp(1j * np.outer(u, r)), axis=1)

def ecf_objective(params, u, r, wfun=lambda u: 1.0/(1.0 + u**2)):
    c, sigma, theta, nu = params
    if sigma <= 0 or nu <= 0:
        return 1e10
    phi_hat = ecf_empirical(u, r)
    phi_theo = phi_vg(u, c, sigma, theta, nu)
    diff = np.abs(phi_hat - phi_theo)**2
    return np.sum(diff * wfun(u))

def estimate_vg_ecf(logr, u, wfun=lambda u: 1.0/(1.0 + u**2)):
    """
    VG estimation via ECF with initial guess robuste
    """
    x0 = automatic_init_vg(logr)
    result = minimize(
        lambda p: ecf_objective(p, u, logr, wfun),
        x0,
        method="Nelder-Mead",
        options={"maxiter":5000, "fatol":1e-9, "xatol":1e-9}
    )
    c, sigma, theta, nu = result.x
    sigma, nu = abs(sigma), abs(nu)
    return np.array([c, sigma, theta, nu]), result

# ============================================================
# 3. Simulation VG
# ============================================================
def simulate_vg(n, params, random_state=None):
    """
    Simule n realisations of VG 
    """
    rng = np.random.default_rng(random_state)
    c, sigma, theta, nu = params
    shape = 1.0 / nu
    scale = nu
    G = rng.gamma(shape=shape, scale=scale, size=n)
    Z = rng.standard_normal(n)
    X = c + theta * G + sigma * np.sqrt(G) * Z
    return X

def ks_distance_vg(logr, params, n_sim=None, random_state=123):
    if n_sim is None:
        n_sim = len(logr)
    X = simulate_vg(n_sim, params, random_state=random_state)
    return ks_2samp(logr, X).statistic