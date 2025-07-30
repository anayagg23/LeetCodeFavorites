import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for presentation-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SDESolver:
    """Simplified SDE solver for weak error comparison"""
    
    def __init__(self, f, g, x0, T, exact_solution=None):
        self.f = f
        self.g = g
        self.x0 = x0
        self.T = T
        self.exact_solution = exact_solution
    
    def euler_maruyama(self, dt, num_paths=50000):
        """Euler-Maruyama method: X_{n+1} = X_n + f(X_n, t_n)Δt + g(X_n, t_n)ΔW_n"""
        N = int(self.T / dt)
        dt_actual = self.T / N
        
        X = np.zeros((num_paths, N + 1))
        X[:, 0] = self.x0
        
        t = np.linspace(0, self.T, N + 1)
        dW = np.sqrt(dt_actual) * np.random.randn(num_paths, N)
        
        for i in range(N):
            X[:, i + 1] = (X[:, i] + 
                          self.f(X[:, i], t[i]) * dt_actual + 
                          self.g(X[:, i], t[i]) * dW[:, i])
        
        return X[:, -1]  # Return only final values
    
    def adams_bashforth_2step(self, dt, num_paths=50000):
        """2-Step Adams-Bashforth method as specified in the problem"""
        N = int(self.T / dt)
        dt_actual = self.T / N
        
        X = np.zeros((num_paths, N + 1))
        X[:, 0] = self.x0
        
        t = np.linspace(0, self.T, N + 1)
        dW = np.sqrt(dt_actual) * np.random.randn(num_paths, N)
        
        # First step using Euler-Maruyama
        X[:, 1] = (X[:, 0] + 
                  self.f(X[:, 0], t[0]) * dt_actual + 
                  self.g(X[:, 0], t[0]) * dW[:, 0])
        
        # Store previous values
        f_prev = self.f(X[:, 0], t[0])
        g_prev = self.g(X[:, 0], t[0])
        
        # Adams-Bashforth 2-step scheme:
        # X_{n+1} = X_n + (3/2 f(X_n, t_n) - 1/2 f(X_{n-1}, t_{n-1}))Δt + 
        #                 (3/2 g(X_n, t_n) - 1/2 g(X_{n-1}, t_{n-1}))ΔW_n
        for i in range(1, N):
            f_curr = self.f(X[:, i], t[i])
            g_curr = self.g(X[:, i], t[i])
            
            X[:, i + 1] = (X[:, i] + 
                          (3/2 * f_curr - 1/2 * f_prev) * dt_actual + 
                          (3/2 * g_curr - 1/2 * g_prev) * dW[:, i])
            
            f_prev = f_curr
            g_prev = g_curr
        
        return X[:, -1]  # Return only final values


def geometric_brownian_motion():
    """Geometric Brownian Motion: dX = μX dt + σX dW"""
    mu, sigma, x0, T = 0.1, 0.2, 1.0, 1.0
    
    def f(x, t): return mu * x
    def g(x, t): return sigma * x
    def exact_solution(t): return x0 * np.exp(mu * t)
    
    return SDESolver(f, g, x0, T, exact_solution)


def create_linear_weak_error_plot():
    """Create the linear scale weak error comparison plot"""
    
    print("Generating weak error comparison plot...")
    
    # Initialize solver
    solver = geometric_brownian_motion()
    
    # Step sizes
    dt_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    num_paths = 50000
    
    # Exact expectation
    exact_expectation = solver.exact_solution(solver.T)
    
    # Compute weak errors
    euler_errors = []
    adams_errors = []
    
    print("Computing Euler-Maruyama errors...")
    for dt in dt_values:
        X_final = solver.euler_maruyama(dt, num_paths)
        numerical_expectation = np.mean(X_final)
        weak_error = abs(numerical_expectation - exact_expectation)
        euler_errors.append(weak_error)
    
    print("Computing Adams-Bashforth errors...")
    for dt in dt_values:
        X_final = solver.adams_bashforth_2step(dt, num_paths)
        numerical_expectation = np.mean(X_final)
        weak_error = abs(numerical_expectation - exact_expectation)
        adams_errors.append(weak_error)
    
    euler_errors = np.array(euler_errors)
    adams_errors = np.array(adams_errors)
    
    # Create the linear scale plot
    plt.figure(figsize=(12, 8))
    
    # Plot with distinct markers and colors
    plt.plot(dt_values, euler_errors, 'o-', linewidth=3, markersize=12, 
             label='Euler-Maruyama', color='#1f77b4', alpha=0.8)
    plt.plot(dt_values, adams_errors, 's--', linewidth=3, markersize=12, 
             label='2-Step Adams-Bashforth', color='#ff7f0e', alpha=0.8)
    
    # Formatting for presentation
    plt.xlabel('Step Size (Δt)', fontsize=16, fontweight='bold')
    plt.ylabel('Weak Error', fontsize=16, fontweight='bold')
    plt.title('Weak Error Comparison: Euler-Maruyama vs 2-Step Adams-Bashforth\n' + 
              'Geometric Brownian Motion (Linear Scale)', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Legend and grid
    plt.legend(fontsize=14, framealpha=0.9, loc='upper right')
    plt.grid(True, alpha=0.3, linewidth=1)
    
    # Tick formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Scientific notation for y-axis if needed
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save high-quality versions
    plt.savefig('presentation_weak_error_linear.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('presentation_weak_error_linear.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Print results
    print("\nWeak Error Results:")
    print("=" * 50)
    print(f"{'Step Size':<12} {'Euler-Maruyama':<15} {'Adams-Bashforth':<15}")
    print("=" * 50)
    for i, dt in enumerate(dt_values):
        print(f"{dt:<12.5f} {euler_errors[i]:<15.6e} {adams_errors[i]:<15.6e}")
    
    print(f"\nExact expectation E[X(T)]: {exact_expectation:.6f}")
    print(f"Number of Monte Carlo paths: {num_paths}")
    
    print("\nPlots saved:")
    print("- presentation_weak_error_linear.png (high-res PNG)")
    print("- presentation_weak_error_linear.pdf (vector PDF)")
    
    return euler_errors, adams_errors, dt_values


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    create_linear_weak_error_plot()