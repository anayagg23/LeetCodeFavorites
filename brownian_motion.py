import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random

# Set up the figure and axis with pink background matching the slide
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')

# Color scheme matching the slide
bg_color = '#F4C2C2'  # Light pink/salmon background
particle_color = '#8B4513'  # Dark brown for small particles
brownian_particle_color = '#FFD700'  # Gold/yellow for the main particle
trail_color = '#FF6B6B'  # Slightly darker pink for trail

ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)

# Remove axes for cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Parameters
num_small_particles = 800  # Many small particles (molecules)
brownian_particle_size = 150  # Large particle size
small_particle_size = 15   # Small particle size
simulation_steps = 2000    # Extended duration
dt = 0.1

# Initialize small particles (gas molecules) - stationary background
small_particles_x = np.random.uniform(5, 95, num_small_particles)
small_particles_y = np.random.uniform(5, 95, num_small_particles)

# Initialize the Brownian particle (dust particle)
brownian_x = 50.0
brownian_y = 50.0

# Storage for the trail
trail_x = [brownian_x]
trail_y = [brownian_y]
max_trail_length = 200  # Show recent path

# Create scatter plots
small_particles_plot = ax.scatter(small_particles_x, small_particles_y, 
                                s=small_particle_size, c=particle_color, 
                                alpha=0.6, edgecolors='none')

brownian_plot = ax.scatter([brownian_x], [brownian_y], 
                         s=brownian_particle_size, c=brownian_particle_color, 
                         alpha=0.9, edgecolors='black', linewidth=2)

# Trail line
trail_line, = ax.plot([], [], color=trail_color, alpha=0.7, linewidth=2)

# Add title
plt.title('Brownian Motion Simulation\nLarge Particle Moving Through Gas Molecules', 
          fontsize=16, fontweight='bold', color='#2C1810', pad=20)

def animate(frame):
    global brownian_x, brownian_y, trail_x, trail_y
    
    # Brownian motion: random walk with Gaussian steps
    dx = np.random.normal(0, 0.8)  # Random displacement in x
    dy = np.random.normal(0, 0.8)  # Random displacement in y
    
    # Update position
    brownian_x += dx
    brownian_y += dy
    
    # Boundary conditions - bounce off walls
    if brownian_x < 5 or brownian_x > 95:
        brownian_x = np.clip(brownian_x, 5, 95)
    if brownian_y < 5 or brownian_y > 95:
        brownian_y = np.clip(brownian_y, 5, 95)
    
    # Update trail
    trail_x.append(brownian_x)
    trail_y.append(brownian_y)
    
    # Keep trail length manageable
    if len(trail_x) > max_trail_length:
        trail_x.pop(0)
        trail_y.pop(0)
    
    # Update plots
    brownian_plot.set_offsets([[brownian_x, brownian_y]])
    trail_line.set_data(trail_x, trail_y)
    
    # Occasionally move some small particles to show they're gas molecules
    if frame % 10 == 0:  # Every 10 frames
        # Move a few random particles slightly
        indices = np.random.choice(num_small_particles, size=50, replace=False)
        small_particles_x[indices] += np.random.normal(0, 0.3, 50)
        small_particles_y[indices] += np.random.normal(0, 0.3, 50)
        
        # Keep them in bounds
        small_particles_x[indices] = np.clip(small_particles_x[indices], 2, 98)
        small_particles_y[indices] = np.clip(small_particles_y[indices], 2, 98)
        
        small_particles_plot.set_offsets(np.column_stack((small_particles_x, small_particles_y)))
    
    return brownian_plot, trail_line, small_particles_plot

# Create animation with longer duration
anim = animation.FuncAnimation(fig, animate, frames=simulation_steps, 
                             interval=50, blit=True, repeat=True)

# Add some text explanation
textstr = 'Yellow particle: Large dust particle\nBrown dots: Gas molecules\nRed line: Brownian motion path'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, color='#2C1810')

plt.tight_layout()
plt.show()

# Optionally save as GIF (uncomment the line below)
# anim.save('brownian_motion.gif', writer='pillow', fps=20, dpi=100)