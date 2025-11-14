import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp

# ------------------------
# Physics & simulation params
# ------------------------
G = 1.0
M = 2.0            # Schwarzschild mass (G=c=1)
E = 0.5            # photon energy scale; p_t = -E
R = 10.0           # "screen" x-position (launch from x=R >> 2M)
b_vals = np.linspace(-12.0, 12.0, 25)   # impact parameters (y-offsets)

# Time grid for integration / playback
N  = 10000                     # number of output samples
dt = 0.01                     # affine-time step for display pacing
T  = N * dt                   # total affine-time duration
t_span = (0.0, T)
t_eval = np.linspace(0.0, T, N)

# ODE tolerances (looser is faster; these are fine for visuals)
RTOL = 1e-6
ATOL = 1e-8
MAX_STEP = dt

# ------------------------
# Geodesic Hamiltonian RHS (equatorial, null)
# State y = [r, phi, p_r]; constants: E, L
# ------------------------
def geodesic_rhs(lam, y, L):
    r, phi, pr = y
    if r <= 2.0*M:                    # guard inside horizon
        return [0.0, 0.0, 0.0]
    f = 1.0 - 2.0*G*M/r               # Schwarzschild factor
    # inverse metric components g^{μν}
    gtt = -1.0/f
    grr = f
    gphph = 1.0/(r*r)
    # radial derivatives
    df_dr = 2.0*G*M/(r*r)
    dgtt_dr = df_dr/(f*f)
    dgrr_dr = df_dr
    dgphph_dr = -2.0/(r**3)
    # Hamilton equations (H=0, null)
    dr   = grr * pr
    dphi = gphph * L
    dpr  = -0.5*(dgtt_dr*(E**2) + dgrr_dr*(pr**2) + dgphph_dr*(L**2))
    return [dr, dphi, dpr]

def initial_pr_from_constraint(r0, L, inward=True):
    f = 1.0 - 2.0*M/r0
    gtt = -1.0/f
    grr = f
    gphph = 1.0/(r0*r0)
    val = (-gtt*(E**2) - gphph*(L**2))/grr   # pr^2 from H=0
    val = max(val, 0.0)
    pr0 = np.sqrt(val)
    return -pr0 if inward else pr0

def event_horizon(lam, y):
    # stop if we hit r = 2M
    return y[0] - (2.0*M + 1e-6)
event_horizon.terminal  = True
event_horizon.direction = -1.0

# ------------------------
# Build initial conditions: parallel rays from x=R along -x
# For each b: (x0, y0)=(R, b) => (r0, phi0) = (hypot(R,b), atan2(b,R))
# Set L=b, choose inward pr0 from constraint
# ------------------------
ics = []
for b in b_vals:
    r0   = np.hypot(R, b)
    phi0 = np.arctan2(b, R)
    pr0  = initial_pr_from_constraint(r0, L=b, inward=True)
    ics.append((r0, phi0, pr0, b))

# ------------------------
# Integrate all rays (once)
# ------------------------
rays_r, rays_phi = [], []
for (r0, phi0, pr0, b) in ics:
    y0 = [r0, phi0, pr0]
    sol = solve_ivp(lambda lam, y: geodesic_rhs(lam, y, L=b),
                    t_span, y0, t_eval=t_eval, max_step=MAX_STEP,
                    rtol=RTOL, atol=ATOL, events=event_horizon)
    rays_r.append(sol.y[0])
    rays_phi.append(sol.y[1])

# Convert to Cartesian for plotting
X_list = [r*np.cos(phi) for r, phi in zip(rays_r, rays_phi)]
Y_list = [r*np.sin(phi) for r, phi in zip(rays_r, rays_phi)]
max_len = max(len(x) for x in X_list)

# ------------------------
# Figure & static artists (tails as static lines)
# ------------------------
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

# Axis limits from all points
xs = np.concatenate([x for x in X_list if len(x)>0]) if X_list else np.array([0])
ys = np.concatenate([y for y in Y_list if len(y)>0]) if Y_list else np.array([0])
pad = 0.1*max(xs.max()-xs.min() if xs.size else 1.0,
              ys.max()-ys.min() if ys.size else 1.0)
ax.set_xlim(xs.min()-pad, xs.max()+pad)
ax.set_ylim(ys.min()-pad, ys.max()+pad)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Photon null geodesics (Schwarzschild) — bundle from +x")

# Horizon (static)
theta = np.linspace(0, 2*np.pi, 400)
rh = 2.0*M
ax.plot(rh*np.cos(theta), rh*np.sin(theta), linestyle="--", linewidth=1.0, color='k', alpha=0.7)

# **Static tails**: draw full trajectories once (no per-frame updates)
for X, Y in zip(X_list, Y_list):
    ax.plot(X, Y, lw=1.0, alpha=0.6)

# Animated heads: one scatter with N_rays points
points = ax.scatter([], [], s=18, zorder=3, animated=True)

def init_anim():
    points.set_offsets(np.empty((0, 2)))
    return [points]

def update_anim(frame):
    # Heads only; use NaN to hide rays that ended before 'frame'
    xs_frame = np.full(len(X_list), np.nan)
    ys_frame = np.full(len(Y_list), np.nan)
    for i, (X, Y) in enumerate(zip(X_list, Y_list)):
        if frame < len(X):
            xs_frame[i] = X[frame]
            ys_frame[i] = Y[frame]
    points.set_offsets(np.column_stack([xs_frame, ys_frame]))
    return [points]

# Interval in ms; since we only move heads, this is very light to draw
interval_ms = max(1, int(1000*dt))

ani = animation.FuncAnimation(
    fig, update_anim, frames=max_len, init_func=init_anim,
    interval=interval_ms, blit=True, repeat=True
)

plt.show()
