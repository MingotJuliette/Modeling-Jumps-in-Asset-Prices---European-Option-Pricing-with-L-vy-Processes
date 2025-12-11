# Pricing of a European Call

## European Call Definition

A European call with strike $K$ and maturity $T$ has payoff:

$$
\text{Payoff} = (S_T - K)^+ = \max(S_T - K, 0)
$$

**Pricing principle:** In an arbitrage-free market, the discounted price is the expected payoff under a **risk-neutral measure $Q$**:

$$
C(S_0, K, T) = e^{-rT} \mathbb{E}^Q[(S_T - K)^+]
$$

Martingale condition for the discounted underlying:

$$
e^{-(r-q)t} S_t \text{ is a martingale under } Q \quad \Leftrightarrow \quad \mathbb{E}^Q[S_T \mid \mathcal{F}_t] = e^{(r-q)(T-t)} S_t
$$

---

## Carr–Madan Fourier Method

**Motivation:** For Lévý models (e.g., VG), the density $f^Q_{S_T}$ is not available in closed form, but the characteristic function of log-price $X_T = \ln S_T$ is:

$$
\phi_{X_T}(u) = \mathbb{E}^Q[e^{i u X_T}]
$$

**Fourier pricing:**  
- Transform the payoff with damping parameter $\alpha > 0$:

$$
C_\alpha(k) = e^{\alpha k} C(k), \quad k = \ln K
$$

- Fourier transform:

$$
\psi_\alpha(\nu) = \int_{-\infty}^{+\infty} e^{i \nu k} C_\alpha(k) \, dk = e^{-rT} \phi_{X_T}(\nu - i(\alpha+1))
$$

- Inverse transform recovers the damped price:

$$
C_\alpha(k) = \frac{1}{2\pi} \int_{-\infty}^{+\infty} e^{-i\nu k} \psi_\alpha(\nu) \, d\nu
$$

---

## Variance Gamma Model: Risk-Neutral Measure

**VG characteristic function:**

$$
\phi_{X_t}(u) = \left( 1 - i \theta \nu u + \frac{1}{2} \sigma^2 \nu u^2 \right)^{-t/\nu}
$$

**Exponential moment condition:**

$$
\mathbb{E}[e^{X_t}] \text{ exists if } 1 - \theta \nu + \frac{1}{2} \sigma^2 \nu > 0
$$

**Risk-neutral adjustment:** Introduce deterministic correction $w_t$:

$$
\ln S_t = \ln S_0 + (r-q)t + X_t - w_t, \quad \mathbb{E}^Q[e^{X_t - w_t}] = 1
$$

- For VG, the cumulant exponent $k(u)$ gives:

$$
k(u) = -\frac{1}{\nu} \ln \left( 1 - \theta \nu u + \frac{1}{2} \sigma^2 \nu u^2 \right)
$$

- Correction term:

$$
w = k(1) = -\frac{1}{\nu} \ln \left( 1 - \theta \nu + \frac{1}{2} \sigma^2 \nu \right)
$$

**Interpretation:**  
- Ensures martingale property under $Q$ without changing VG jump parameters.  
- Alternative: Esscher transform, modifies Lévý measure but more complex.

**Damping parameter $\alpha$ condition:**  

$$
1 - \theta \nu (\alpha + 1) + \frac{1}{2} \sigma^2 \nu (\alpha + 1)^2 > 0
$$

- Guarantees convergence of the Fourier transform and numerical stability.
