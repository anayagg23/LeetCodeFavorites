import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Set style for presentation-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SDESolver:
    """Class to solve SDEs using different numerical methods"""
    
    def __init__(self, f, g, x0, T, exact_solution=None):
        """
        Initialize SDE solver
        f: drift function f(x, t)
        g: diffusion function g(x, t)
        x0: initial condition
        T: final time
        exact_solution: function to compute exact solution (if known)
        """
        self.f = f
        self.g = g
        self.x0 = x0
        self.T = T
        self.exact_solution = exact_solution
    
    def euler_maruyama(self, dt, num_paths=10000):
        """Euler-Maruyama method"""
        N = int(self.T / dt)
        dt_actual = self.T / N
        
        # Initialize paths
        X = np.zeros((num_paths, N + 1))
        X[:, 0] = self.x0
        
        # Time grid
        t = np.linspace(0, self.T, N + 1)
        
        # Generate Wiener increments
        dW = np.sqrt(dt_actual) * np.random.randn(num_paths, N)
        
        # Euler-Maruyama scheme
        for i in range(N):
            X[:, i + 1] = (X[:, i] + 
                          self.f(X[:, i], t[i]) * dt_actual + 
                          self.g(X[:, i], t[i]) * dW[:, i])
        
        return X, t
    
    def adams_bashforth_2step(self, dt, num_paths=10000):
        """2-Step Adams-Bashforth method"""
        N = int(self.T / dt)
        dt_actual = self.T / N
        
        # Initialize paths
        X = np.zeros((num_paths, N + 1))
        X[:, 0] = self.x0
        
        # Time grid
        t = np.linspace(0, self.T, N + 1)
        
        # Generate Wiener increments
        dW = np.sqrt(dt_actual) * np.random.randn(num_paths, N)
        
        # First step using Euler-Maruyama (we need X_1 to start Adams-Bashforth)
        X[:, 1] = (X[:, 0] + 
                  self.f(X[:, 0], t[0]) * dt_actual + 
                  self.g(X[:, 0], t[0]) * dW[:, 0])
        
        # Store previous values for Adams-Bashforth
        f_prev = self.f(X[:, 0], t[0])
        g_prev = self.g(X[:, 0], t[0])
        
        # Adams-Bashforth 2-step scheme
        for i in range(1, N):
            f_curr = self.f(X[:, i], t[i])
            g_curr = self.g(X[:, i], t[i])
            
            X[:, i + 1] = (X[:, i] + 
                          (3/2 * f_curr - 1/2 * f_prev) * dt_actual + 
                          (3/2 * g_curr - 1/2 * g_prev) * dW[:, i])
            
            # Update previous values
            f_prev = f_curr
            g_prev = g_curr
        
        return X, t
    
    def compute_weak_error(self, method, dt_values, num_paths=10000, test_function=None):
        """Compute weak error for a given method"""
        if test_function is None:
            test_function = lambda x: x  # Default: E[X_T]
        
        weak_errors = []
        
        for dt in dt_values:
            if method == 'euler':
                X, t = self.euler_maruyama(dt, num_paths)
            elif method == 'adams_bashforth':
                X, t = self.adams_bashforth_2step(dt, num_paths)
            
            # Compute numerical expectation
            numerical_expectation = np.mean(test_function(X[:, -1]))
            
            # Compute exact expectation (if available)
            if self.exact_solution is not None:
                exact_value = self.exact_solution(self.T)
                if callable(test_function):
                    # For simple test functions, we can compute this analytically
                    # For the geometric Brownian motion example, E[X_T] = x0 * exp(mu * T)
                    exact_expectation = exact_value
                else:
                    exact_expectation = exact_value
            else:
                # Use very fine discretization as reference
                X_ref, _ = self.euler_maruyama(dt/100, num_paths*10)
                exact_expectation = np.mean(test_function(X_ref[:, -1]))
            
            weak_error = abs(numerical_expectation - exact_expectation)
            weak_errors.append(weak_error)
        
        return np.array(weak_errors)


def geometric_brownian_motion_example():
    """Example: Geometric Brownian Motion dX = mu*X*dt + sigma*X*dW"""
    
    # Parameters
    mu = 0.1    # drift parameter
    sigma = 0.2 # volatility parameter
    x0 = 1.0    # initial value
    T = 1.0     # final time
    
    # Define drift and diffusion functions
    def f(x, t):
        return mu * x
    
    def g(x, t):
        return sigma * x
    
    # Exact solution: X(t) = x0 * exp((mu - sigma^2/2)*t + sigma*W(t))
    # E[X(T)] = x0 * exp(mu * T)
    def exact_solution(t):
        return x0 * np.exp(mu * t)
    
    return SDESolver(f, g, x0, T, exact_solution)


def ornstein_uhlenbeck_example():
    """Example: Ornstein-Uhlenbeck process dX = -theta*X*dt + sigma*dW"""
    
    # Parameters
    theta = 1.0  # mean reversion rate
    sigma = 0.5  # volatility
    x0 = 1.0     # initial value
    T = 1.0      # final time
    
    # Define drift and diffusion functions
    def f(x, t):
        return -theta * x
    
    def g(x, t):
        return sigma * np.ones_like(x)
    
    # Exact solution: E[X(T)] = x0 * exp(-theta * T)
    def exact_solution(t):
        return x0 * np.exp(-theta * t)
    
    return SDESolver(f, g, x0, T, exact_solution)


def main():
    """Main function to run the comparison"""
    
    print("Comparing weak error of Euler-Maruyama vs 2-Step Adams-Bashforth")
    print("=" * 60)
    
    # Choose SDE example
    solver = geometric_brownian_motion_example()
    # solver = ornstein_uhlenbeck_example()  # Alternative example
    
    # Step sizes to test
    dt_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    num_paths = 50000  # Number of Monte Carlo paths
    
    print(f"Using {num_paths} Monte Carlo paths")
    print(f"Step sizes: {dt_values}")
    print()
    
    # Compute weak errors
    print("Computing weak errors for Euler-Maruyama...")
    euler_errors = solver.compute_weak_error('euler', dt_values, num_paths)
    
    print("Computing weak errors for 2-Step Adams-Bashforth...")
    adams_errors = solver.compute_weak_error('adams_bashforth', dt_values, num_paths)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot weak errors
    plt.loglog(dt_values, euler_errors, 'o-', linewidth=2, markersize=8, 
               label='Euler-Maruyama', alpha=0.8)
    plt.loglog(dt_values, adams_errors, 's--', linewidth=2, markersize=8, 
               label='2-Step Adams-Bashforth', alpha=0.8)
    
    # Add reference lines for convergence rates
    plt.loglog(dt_values, dt_values, ':', color='gray', alpha=0.7, 
               label='Order 1 (dt)')
    plt.loglog(dt_values, dt_values**2, ':', color='lightgray', alpha=0.7, 
               label='Order 2 (dt²)')
    
    plt.xlabel('Step Size (Δt)', fontsize=14, fontweight='bold')
    plt.ylabel('Weak Error', fontsize=14, fontweight='bold')
    plt.title('Weak Error Comparison: Euler-Maruyama vs 2-Step Adams-Bashforth\n' + 
              'Geometric Brownian Motion', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('weak_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('weak_error_comparison.pdf', bbox_inches='tight')
    
    # Also create a linear scale version for presentation
    plt.figure(figsize=(12, 8))
    
    plt.plot(dt_values, euler_errors, 'o-', linewidth=3, markersize=10, 
             label='Euler-Maruyama', alpha=0.8)
    plt.plot(dt_values, adams_errors, 's--', linewidth=3, markersize=10, 
             label='2-Step Adams-Bashforth', alpha=0.8)
    
    plt.xlabel('Step Size (Δt)', fontsize=14, fontweight='bold')
    plt.ylabel('Weak Error', fontsize=14, fontweight='bold')
    plt.title('Weak Error Comparison (Linear Scale)\n' + 
              'Euler-Maruyama vs 2-Step Adams-Bashforth', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save linear scale plot
    plt.savefig('weak_error_comparison_linear.png', dpi=300, bbox_inches='tight')
    plt.savefig('weak_error_comparison_linear.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print numerical results
    print("\nNumerical Results:")
    print("-" * 40)
    print(f"{'Step Size':<12} {'Euler-Maruyama':<15} {'Adams-Bashforth':<15}")
    print("-" * 40)
    for i, dt in enumerate(dt_values):
        print(f"{dt:<12.5f} {euler_errors[i]:<15.6e} {adams_errors[i]:<15.6e}")
    
    # Compute convergence rates
    print("\nConvergence Rates:")
    print("-" * 30)
    
    def compute_convergence_rate(errors, dt_values):
        # Use last few points to estimate convergence rate
        log_errors = np.log(errors[-3:])
        log_dt = np.log(dt_values[-3:])
        rate = np.polyfit(log_dt, log_errors, 1)[0]
        return rate
    
    euler_rate = compute_convergence_rate(euler_errors, dt_values)
    adams_rate = compute_convergence_rate(adams_errors, dt_values)
    
    print(f"Euler-Maruyama: {euler_rate:.2f}")
    print(f"Adams-Bashforth: {adams_rate:.2f}")
    
    print(f"\nPlots saved as:")
    print("- weak_error_comparison.png (log-log scale)")
    print("- weak_error_comparison_linear.png (linear scale)")
    print("- PDF versions also saved")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()