import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----- Parameters you can tweak -----
h0 = 0.2           # strain amplitude (exaggerated so effect is visible)
f = 1            # wave frequency [Hz] (arbitrary units)
omega = 2 * np.pi * f
duration = 10      # seconds of animation
fps = 20           # frames per second
grid_n = 10        # grid points per axis (total points = grid_n^2)
extent = 1.0       # half-size of the square (from -extent to +extent)

# Create a square grid of test masses in the z=0 plane
xs = np.linspace(-extent, extent, grid_n)
ys = np.linspace(-extent, extent, grid_n)
X0, Y0 = np.meshgrid(xs, ys)
x0 = X0.flatten()
y0 = Y0.flatten()

# Time array for animation frames
nframes = duration * fps
ts = np.linspace(0, duration, nframes)

# Figure
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.2 * extent, 1.2 * extent)
ax.set_ylim(-1.2 * extent, 1.2 * extent)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gravitational Wave: + Polarization (propagating along +z)')

# Un-deformed reference grid (light markers)
ref = ax.scatter(x0, y0, s=10, alpha=0.2)

# Deformed positions (animated)
sc = ax.scatter(x0, y0, s=20)

def positions(t):
    h = h0 * np.cos(omega * t)
    x = x0 * (1 + h / 2.0)
    y = y0 * (1 - h / 2.0)
    return x, y

def init():
    sc.set_offsets(np.c_[x0, y0])
    return (sc,)

def update(frame):
    t = ts[frame]
    x, y = positions(t)
    sc.set_offsets(np.c_[x, y])
    ax.set_title(f'h0={h0}, f={f} Hz, t={t:4.2f}s')
    return (sc,)

anim = FuncAnimation(fig, update, frames=nframes, init_func=init, blit=True, interval=1000/fps)

# Save as GIF so you can download/view later
gif_path = "./gw_plus_grid.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps))
