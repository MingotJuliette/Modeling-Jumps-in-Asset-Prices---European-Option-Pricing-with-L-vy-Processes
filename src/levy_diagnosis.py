import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tailestim import HillEstimator
import src.Merton as M_methode
import src.Variance_Gamma as VG_methode
import seaborn as sns
from scipy.fft import fft

################################################################################
#######                Anylsis et diagnosis                         ########## 
################################################################################

def make_uniform_u_grid(tail_index, N, u_min_cap=1, u_max_cap=20):
    
    u_max0 = 10 / tail_index
    u_max = np.clip(u_max0, u_min_cap, u_max_cap)

    u = np.linspace(0, u_max, N)
    return u, u_max

def compute_log_returns(df, price_col=('close','gc=f')):
    if isinstance(price_col, tuple):
        s = df[price_col]
    else:
        s = df[price_col]
    logr = np.log(s).diff().dropna()
    return logr

def summary_stats(logr):
    print("count", len(logr))
    print("mean", logr.mean())
    print("std", logr.std())
    print("skew", stats.skew(logr))
    print("kurtosis (excess)", stats.kurtosis(logr))
    return

def estimate_tail_index(log_returns, epsilon=1e-8):

    data = np.abs(np.asarray(log_returns))

    data = np.clip(data, epsilon, None)

    est = HillEstimator()
    est.fit(data)
    res = est.get_result()

    return res.xi_star_



def plot_diagnostics(logr, out_prefix="diagnostics"):
    x = np.asarray(logr)
    n = len(x)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # -------------------------------------------------
    # 1) Histogram + KDE
    # -------------------------------------------------
    axes[0].hist(x, bins=200, density=True, alpha=0.6, color='skyblue', label="Empirical")
    kde = stats.gaussian_kde(x)
    grid = np.linspace(np.min(x), np.max(x), 500)
    axes[0].plot(grid, kde(grid), color='darkblue', lw=2, label="KDE")
    axes[0].set_title("Histogram + KDE")
    axes[0].legend()

    # -------------------------------------------------
    # 2) QQ-plot vs Normal
    # -------------------------------------------------
    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot vs Normal")

    # -------------------------------------------------
    # 3) Hill Tail Index
    # -------------------------------------------------
    absx = np.sort(np.abs(x))[::-1]
    ks_range = np.arange(5, int(n*0.10))
    hill_vals = [(1.0 / ((1.0/k) * np.sum(np.log(absx[:k]/absx[k-1])))) for k in ks_range]
    axes[2].plot(ks_range, hill_vals, color='purple', lw=2)
    axes[2].set_xlabel("k (top order statistics)")
    axes[2].set_ylabel("Tail index α")
    axes[2].set_title("Hill Tail Index Plot")
    axes[2].grid(True)

    # -------------------------------------------------
    # 4) Log-Log Survival Plot
    # -------------------------------------------------
    sorted_absx = np.sort(np.abs(x))
    survival = 1 - np.arange(1, n+1)/n
    axes[3].loglog(sorted_absx, survival, marker='.', linestyle='none', color='orange')
    axes[3].set_title("Log-Log Survival Plot")
    axes[3].set_xlabel("|x|")
    axes[3].set_ylabel("Survival Prob (1-F(x))")
    axes[3].grid(True)

    # -------------------------------------------------
    # 5) Empirical PDF vs Gaussian
    # -------------------------------------------------
    mu, sigma = np.mean(x), np.std(x)
    norm_pdf = stats.norm.pdf(grid, mu, sigma)
    axes[4].hist(x, bins=150, density=True, alpha=0.4, color='lightgreen', label="Empirical")
    axes[4].plot(grid, norm_pdf, lw=2, color='darkgreen', label=f"N({mu:.4f},{sigma:.4f})")
    axes[4].set_title("PDF vs Gaussian")
    axes[4].legend()

    # -------------------------------------------------
    # 6) Empirical CDF vs Gaussian CDF
    # -------------------------------------------------
    emp_cdf = np.arange(1, n+1)/n
    axes[5].plot(np.sort(x), emp_cdf, marker='.', linestyle='none', color='red', label="Empirical CDF")
    axes[5].plot(grid, stats.norm.cdf(grid, mu, sigma), lw=2, color='black', label="Gaussian CDF")
    axes[5].set_title("Empirical CDF vs Gaussian CDF")
    axes[5].legend()
    axes[5].grid(True)

    # -------------------------------------------------
    # Layout & save
    # -------------------------------------------------
    plt.tight_layout()
    plt.savefig(out_prefix + "_diagnostics.png", dpi=150)
    plt.show()

    # Affichage estimation stable du tail index
    alpha_hat = hill_vals[len(hill_vals)//3 : len(hill_vals)//3*2]
    print(f"Approx. tail index α = {np.median(alpha_hat):.3f}")




def detect_jumps(logr, threshold_std=4.0):
    mu = logr.mean()
    sigma = logr.std()
    jumps = logr[np.abs((logr - mu) / sigma) > threshold_std]
    print(f"Detected {len(jumps)} jumps with threshold {threshold_std}σ")
    return jumps


PHI_FUN = {
    "VG": VG_methode.phi_vg,
    "Merton": M_methode.phi_merton,
}

def plot_all_vg_diagnostics(logr, sim_mle, sim_ecf, u,
                            params_mle, params_ecf, methode):
    


    plt.style.use("seaborn-v0_8")

    # Prepare grids
    x_grid = np.linspace(np.min(logr), np.max(logr), 500)
    sorted_data = np.sort(logr)
    sorted_mle  = np.sort(sim_mle)

    # Characteristic functions

    phi_fun = PHI_FUN[methode]

    phi_mle = phi_fun(u, *params_mle)
    phi_ecf = phi_fun(u, *params_ecf)
    
    if methode == "VG" : 
        phi_hat = VG_methode.ecf_empirical(u, logr)
    elif methode == "Merton" : 
        phi_hat = M_methode.ecf_empirical(u, logr)

    # -----------------------------
    # Subplots configuration : 2×3
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    # -------------------------------------------------------------------
    # 1) Histogram + KDE simulations  (top-left)
    # -------------------------------------------------------------------
    ax = axes[0, 0]
    sns.histplot(logr, bins=80, stat="density", alpha=0.5, label="data", ax=ax)
    sns.kdeplot(sim_mle, lw=2, label="MLE sim", ax=ax)
    sns.kdeplot(sim_ecf, lw=2, label="ECF sim", ax=ax)
    ax.set_title("Histogramme + KDE simulées")
    ax.legend()

    # -------------------------------------------------------------------
    # 2) QQ-plot vs normale (top-middle)
    # -------------------------------------------------------------------
    ax = axes[0, 1]
    stats.probplot(logr, dist="norm", plot=ax)
    ax.set_title("QQ-plot data vs N(μ,σ)")

    # -------------------------------------------------------------------
    # 3) QQ data vs MLE sim (top-right)
    # -------------------------------------------------------------------
    ax = axes[0, 2]
    ax.plot(sorted_mle, sorted_data, "b.", alpha=0.6)
    ax.plot([sorted_data.min(), sorted_data.max()],
            [sorted_data.min(), sorted_data.max()], "r--")
    ax.set_title("QQ-plot data vs simulation MLE")
    ax.set_xlabel("quantiles MLE")
    ax.set_ylabel("quantiles data")

    # -------------------------------------------------------------------
    # 4) ECF : Re(phi) (bottom-left)
    # -------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(u, np.real(phi_hat), "k.", label="Re(φ empirical)")
    ax.plot(u, np.real(phi_mle), "b-", lw=2, label="Re(φ MLE)")
    ax.plot(u, np.real(phi_ecf), "g--", lw=2, label="Re(φ ECF)")
    ax.set_title("ECF – Partie réelle")
    ax.legend()

    # -------------------------------------------------------------------
    # 5) ECF : |phi| (bottom-middle) — NEW GRAPHIC
    # -------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(u, np.abs(phi_hat), "k.", label="|φ empirical|")
    ax.plot(u, np.abs(phi_mle), "b-", lw=2, label="|φ MLE|")
    ax.plot(u, np.abs(phi_ecf), "g--", lw=2, label="|φ ECF|")
    ax.set_title("ECF – Module |φ(u)|")
    ax.legend()

    # -------------------------------------------------------------------
    # 6) PDF empirical + normal + densités simulées (bottom-right)
    # -------------------------------------------------------------------
    ax = axes[1, 2]
    sns.kdeplot(logr, lw=2, label="Empirical", ax=ax)
    sns.kdeplot(sim_mle, lw=1.5, label="MLE sim", ax=ax)
    sns.kdeplot(sim_ecf, lw=1.5, label="ECF sim", ax=ax)
    mu, sigma = np.mean(logr), np.std(logr)
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma),
            "r--", lw=2, label="Normal fit")
    ax.set_title("PDF empirique vs normale & simulations")
    ax.legend()


################################################################################
#######           Fourrier tranform (carr-madan -> FFT)               ########## 
################################################################################

# -----------------------------
# Fonction caractéristique VG
# -----------------------------
def phi_vg_car(u, theta, nu, sigma, T, S0, r, q, w):
    """Characteristic function of log(S_T) under VG"""
    cu = 1 - 1j*theta*nu*u + 0.5*sigma**2 * nu * u**2
    vg = cu ** (-T/nu)
    drift = np.exp(1j*u*(np.log(S0) + (r - q - w)*T))
    return drift * vg

# -----------------------------
# Carr-Madan FFT pricing
# -----------------------------
def price_fft(S0, r, T, phi, alpha, N, eta):
    """
    Carr–Madan FFT pricing for European calls
    S0: Spot
    r: risk-free rate
    T: maturity (years)
    phi: characteristic function of log(S_T)
    alpha: damping factor
    N: number of FFT points
    eta: spacing in frequency domain
    """
    # 1. frequences
    u = np.arange(N) * eta

    # 2. Lambda and shift
    lamb = 2*np.pi / (N*eta)
    b = np.log(S0) - N*lamb/2    # centered arounf the spot

    # 3. log-strikes and strikes
    k = b + lamb*np.arange(N)
    K = np.exp(k)

    # 4. Psi for FFT
    u_shift = u - 1j*(alpha+1)
    phi_vals = phi(u_shift)
    denom = (alpha**2 + alpha - u**2) + 1j*u*(2*alpha + 1)
    psi = np.exp(-r*T) * phi_vals / denom

    # 5. FFT input with Simpson ponderation
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    fft_in = np.exp(-1j * b * u) * psi * eta * w

    # 6. FFT
    fft_vals = fft(fft_in)
    C = np.exp(-alpha * k) * np.real(fft_vals) / np.pi

    df = pd.DataFrame({"K": K, "C": C})
    df = df[df["C"]>0] 
    return df.sort_values("K").reset_index(drop=True)

################################################################################
#######                    Analyse of result                          ########## 
################################################################################


def simulate_vg_price(n, S0, T, c, sigma, theta, nu, r, q=0.0, random_state=None):
    rng = np.random.default_rng(random_state)
    
    # Drift correction for martingale
    omega = (1.0/nu) * np.log(1 - theta*nu - 0.5*sigma**2*nu)
    
    # Gamma
    shape = T / nu
    scale = nu
    G = rng.gamma(shape=shape, scale=scale, size=n)
    
    # Bruit normal
    Z = rng.standard_normal(n)
    
    # Increments X_T
    X = theta * G + sigma * np.sqrt(G) * Z
    
    # Prix S_T corrected
    ST = S0 * np.exp((r - q) * T + omega * T + X)
    return ST




