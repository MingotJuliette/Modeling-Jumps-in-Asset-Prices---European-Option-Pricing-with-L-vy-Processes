# Specific Models
## Variance Gamma (VG)

VG is a Lévý process built by **subordinating Brownian motion with a Gamma process**:

$$
X_t = \theta G_t + \sigma W_{G_t}, \quad G_t \sim \Gamma(t/\nu, \nu)
$$

- $\theta$: drift, $\sigma$: Brownian volatility, $\nu$: jump activity.  
- Independent and stationary increments; càdlàg paths.  

**Characteristic function:**

$$
\phi_{X_t}(u) = \left( 1 - \nu \left( i u \theta - \frac{1}{2} \sigma^2 u^2 \right) \right)^{-t/\nu}
$$  

**Parameters:**  
- $\nu$: jump frequency  
- $\sigma$: jump size  
- $\theta$: skewness  
- As $\nu \to 0$, VG → Brownian motion with drift $\theta$ and volatility $\sigma$.

---

## Merton Model

Merton extends Black–Scholes with **log-normal jumps**:

$$
X_t = \mu - \frac{\sigma^2}{2} - \lambda k t + \sigma W_t + \sum_{i=1}^{N_t} Y_i
$$

- $N_t \sim \text{Poisson}(\lambda t)$, $Y_i \sim N(\mu_J, \sigma_J^2)$  
- Decomposition: $X_t = X_t^{\text{cont}} + X_t^{\text{jumps}}$  
- Lévý triplet: $(\mu - \frac{\sigma^2}{2} - \lambda k, \sigma^2, \nu(dy))$  

**Characteristic function:**

$$
\phi_{X_t}(u) = \exp \Big( i u (\mu - \frac{\sigma^2}{2} - \lambda k) t - \frac{1}{2} \sigma^2 u^2 t + \lambda t ( e^{i u \mu_J - \frac{1}{2} \sigma_J^2 u^2} - 1) \Big)
$$  

---

# Parameter Estimation

Methods: **Maximum Likelihood Estimation (MLE)** and **Empirical Characteristic Function (ECF)**.

### Maximum Likelihood Estimation (MLE)

For sample $(x_1,\dots,x_n)$:

$$
L(x_1,\dots,x_n;\Theta) = \prod_{i=1}^{n} f(x_i;\Theta), \quad \hat{\Theta} = \arg\max \ell(\Theta)
$$

- VG: marginal density involves Bessel function $K_\gamma$.  
- Merton: marginal density sums over Poisson-distributed jumps.  
- MLE is solved numerically.

###  Empirical Characteristic FunctionMethod (ECF)

Empirical CF:

$$
\hat{\phi}(u) = \frac{1}{n} \sum_{j=1}^n e^{i u r_j}
$$

Minimize weighted error:

$$
J(\Theta) = \sum_k w_k |\hat{\phi}(u_k) - \phi(u_k;\Theta)|^2, \quad \hat{\Theta} = \arg\min_\Theta J(\Theta)
$$

- Weights: $w(u) = 1 / (1 + u^2)$  
- VG and Merton CF as above.  
- Initial values from robust moments and jump detection.

### Optimization

- Minimize $f(\Theta)$ (MLE or ECF).  
- Gradient-free methods (e.g., Nelder–Mead) used for multimodal, nonlinear functions.  

### Fit metrics:  
  - **Kolmogorov–Smirnov:** $D_n = \sup_x |F_n(x) - F(x;\hat{\Theta})|$  
  - **Wasserstein (order 1):** $W_1(\mu, \nu) = \inf E[d(X,Y)]$
