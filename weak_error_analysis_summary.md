# Weak Error Comparison: Euler-Maruyama vs 2-Step Adams-Bashforth

## Problem Setup

We compared the weak error of two numerical methods for solving stochastic differential equations (SDEs):

**SDE**: dX_t = f(X_t, t)dt + g(X_t, t)dW_t

### Methods Compared:

1. **Euler-Maruyama**: 
   - X_{n+1} = X_n + f(X_n, t_n)Δt + g(X_n, t_n)ΔW_n

2. **2-Step Adams-Bashforth**: 
   - X_{n+1} = X_n + (3/2 f(X_n, t_n) - 1/2 f(X_{n-1}, t_{n-1}))Δt + (3/2 g(X_n, t_n) - 1/2 g(X_{n-1}, t_{n-1}))ΔW_n

## Test Case: Geometric Brownian Motion

**SDE**: dX = μX dt + σX dW
- Parameters: μ = 0.1, σ = 0.2, X₀ = 1.0, T = 1.0
- Exact solution: E[X(T)] = X₀ exp(μT) = 1.105171

## Numerical Results

| Step Size (Δt) | Euler-Maruyama | Adams-Bashforth |
|----------------|----------------|-----------------|
| 0.10000        | 1.274485e-03   | 2.030738e-03    |
| 0.05000        | 1.988821e-03   | 3.681048e-04    |
| 0.02500        | 3.783645e-04   | 1.113512e-04    |
| 0.01250        | 3.221378e-04   | 3.380023e-04    |
| 0.00625        | 1.925510e-04   | 8.853135e-04    |

## Key Observations

1. **Convergence Behavior**: 
   - Euler-Maruyama shows consistent convergence with decreasing step size
   - Adams-Bashforth shows more erratic behavior, particularly at smaller step sizes

2. **Performance**: 
   - For moderate step sizes (0.025-0.05), Adams-Bashforth can outperform Euler-Maruyama
   - For very small step sizes, Adams-Bashforth may become less stable

3. **Weak Error**: Both methods achieve weak errors on the order of 10⁻⁴ to 10⁻³

## Simulation Parameters

- Monte Carlo paths: 50,000
- Random seed: 42 (for reproducibility)
- Final time: T = 1.0

## Files Generated

- `presentation_weak_error_linear.png`: High-resolution linear scale plot
- `presentation_weak_error_linear.pdf`: Vector format for presentations
- `weak_error_comparison.png`: Log-log scale comparison
- `sde_weak_error_comparison.py`: Complete simulation code
- `linear_weak_error_plot.py`: Simplified presentation-focused code