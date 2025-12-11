# Lévy Processes

Lévy processes extend the Black–Scholes model by combining a Brownian component, jumps, and deterministic drift. Unlike the classical model with continuous log-normal returns and constant volatility, Lévý processes capture features like sudden price jumps, heavy tails, and skewed return distributions. They allow more realistic option valuation and calibration to implied volatility surfaces. Based on Cont & Tankov.

## Definition

A stochastic process $(X_t)_{t \ge 0}$ is a Lévý process if it satisfies:

1. **Independent increments**: $X_{t_2} - X_{t_1}, \dots, X_{t_n} - X_{t_{n-1}}$ are independent.
2. **Stationary increments**: $X_t - X_s \stackrel{d}{=} X_{t-s}$ for all $0 \le s < t$.
3. **Càdlàg paths**: $t \to X_t$ is right-continuous with left limits.

**Implications:**

- Continuity in probability: $\lim_{h \to 0} P(|X_{t+h} - X_t| > \epsilon) = 0$
- No deterministic jumps: $P(X_t = X_{t^-}) = 1$
- Trajectories generally include jumps; only continuous paths imply Brownian motion with drift.

## Characteristic Function and Lévý Exponent

The characteristic function of $X_t$ is central:

$$E[e^{iu X_t}] = E[e^{iu X_1}]^t = e^{t \psi(u)}, \quad \psi(u) := \log E[e^{iu X_1}]$$

where \(\psi(u)\) is the **Lévý exponent**.

## Lévy–Itô Decomposition

Any Lévý process can be decomposed into:

$$X_1 = \gamma + \sigma W_1 + \int_{|x|<1} x \tilde{N}(1,dx) + \int_{|x|>1} x N(1,dx)$$

- $W$ is standard Brownian motion  
- $N(t,dx)$ is a Poisson random measure with intensity $t \nu(dx)$  
- $\tilde{N}(t,dx) = N(t,dx) - t\nu(dx)$ compensates small jumps  
- $\gamma \in \mathbb{R}$, $\sigma \ge 0$  
- $\nu$ is the Lévy measure on $\mathbb{R} \setminus {0}$, with $\int (1 \wedge x^2) \nu(dx) < \infty$  

**Notes:**

- Large jumps $(|x|>1)\$ follow a compound Poisson process with i.i.d. amplitudes.  
- Small jumps $(|x|<1)$ may be infinite; compensation ensures convergence and càdlàg paths.

## Lévy–Khintchine Formula

The characteristic function can be written as:

$$\phi_{X_t}(u) = E[e^{iu X_t}] = \exp \left\{ t \left( i u \gamma - \frac{1}{2}\sigma^2 u^2 + \int_{\mathbb{R}} \big( e^{iux} - 1 - iux \mathbf{1}_{|x|<1} \big) \nu(dx) \right) \right\}$$

- **Drift**: $i u \gamma$ 
- **Brownian diffusion**: $-\frac{1}{2} \sigma^2 u^2$  
- **Jump part**: $\int ( e^{iux} -1 - iux 1_{|x|<1}) \nu(dx)$  

The triplet $(\gamma, \sigma, \nu)$ fully characterizes the process.
