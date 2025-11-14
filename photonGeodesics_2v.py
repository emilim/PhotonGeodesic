import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

# ------------------------
# Physics & simulation params
# ------------------------
G = 1.0
M = 1.0            # Schwarzschild mass (G=c=1)
E = 1.0            # photon energy scale; p_t = -E
bs = np.arange(5.19, 5.20, 0.001)
print(bs)
Ls = bs * E
r0 = 10.0          # initial radius
phi0 = 0.0         # initial angle

T_max = 40.0       # max affine-time to integrate
dt_max = 1e-3      # largest step far from the BH
dt_min = 1e-8      # safety floor near horizon
gamma  = 1.0       # how aggressively dt shrinks as r -> 2M
chi    = 0.05      # max allowed fractional radial change per step
r_soft = 1e-3      # softening scale in the CFL limiter

rh = 2.0*M

def Veff(r, L, M):      # photon effective potential
    return (L*L / r**2) * (1.0 - 2.0*M/r)

def dVeff_dr(r, L, M):
    # d/dr [ L^2 (1 - 2M/r) / r^2 ] = -2 L^2 / r^3 + 6 M L^2 / r^4
    return (-2.0 * L*L) / (r**3) + (6.0 * M * L*L) / (r**4)

def a_r(r, L, M):
    # radial "acceleration" in the effective 1D problem (null geodesic)
    return -0.5 * dVeff_dr(r, L, M)

def choose_dt(r, rdot):
    # 1) horizon-based shrink: goes to dt_min as r -> 2M
    hdist = max(r - rh, 0.0)
    fh = (hdist / (r0 - rh + 1e-12))**gamma
    dt_h = dt_min + (dt_max - dt_min) * np.clip(fh, 0.0, 1.0)

    # 2) CFL-like bound: limit |Δr| per step
    #    require |rdot| * dt <= chi * max(r - 2M, r_soft)
    dr_allow = chi * max(hdist, r_soft)
    dt_cfl = dt_max if abs(rdot) < 1e-20 else dr_allow / abs(rdot)

    # final dt
    return float(np.clip(min(dt_h, dt_cfl), dt_min, dt_max))

X, Y, Tcoord = [], [], []

for L, b in zip(Ls, bs):
    xs, ys, tlist = [], [], []

    # initial conditions
    r   = np.sqrt(r0**2 + b**2)
    phi = np.arctan2(b, r0) + phi0  
    tcoord = 0.0      # Schwarzschild coordinate time (for an observer at infinity)

    # inward-going radial speed from E^2 = rdot^2 + Veff
    rdot = -np.sqrt(max(E*E - Veff(r, L, M), 0.0))

    tau = 0.0         # affine parameter accumulator
    while tau < T_max and r > rh:
        # pick adaptive dt
        dt = choose_dt(r, rdot)

        # velocity-Verlet with variable dt:
        # kick
        ar = a_r(r, L, M)
        rdot_half = rdot + 0.5 * dt * ar
        # drift
        r_new = r + dt * rdot_half

        # midpoint values for better phi,t updates
        r_mid = 0.5 * (r + r_new)

        # update phi using midpoint radius: phi' = L / r^2
        phi += dt * (L / (r_mid * r_mid))

        # coordinate time using midpoint: dt/dλ = E / (1 - 2M/r)
        tcoord += dt * E / (1.0 - 2.0*G*M / r_mid)

        # complete kick with new acceleration
        ar_new = a_r(r_new, L, M)
        rdot_new = rdot_half + 0.5 * dt * ar_new

        # accept step
        r, rdot = r_new, rdot_new
        tau += dt

        # store for plotting
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xs.append(x); ys.append(y); tlist.append(tcoord)
        #print(f"tau={tau:.3f} r={r:.3f} phi={phi:.3f} t={tcoord:.3f} dt={dt:.3e}")
        # optional early exit if photon clearly escapes outward
        if r > r0 + 5.0 and rdot > 0:
            break

    X.append(xs); Y.append(ys); Tcoord.append(tlist)

# --- animated 2D plotting (time as animation parameter) ---
from matplotlib import animation

# prepare arrays
Xs = [np.array(xs) for xs in X]
Ys = [np.array(ys) for ys in Y]
Ts = [np.array(ts) for ts in Tcoord]
t_end = max(ts[-1] for ts in Ts if len(ts) > 0)

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-r0, r0)
ax.set_ylim(-r0, r0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Photon trajectory (animated by Schwarzschild time)")

# horizon circle
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(rh*np.cos(theta), rh*np.sin(theta), linestyle="--", linewidth=1.0, color='k', alpha=0.7)

# one line per trajectory
lines = [ax.plot([], [], lw=1)[0] for _ in Xs]

# time label
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")

def init():
    for ln in lines:
        ln.set_data([], [])
    time_text.set_text("")
    return (*lines, time_text)

def update(frame_t):
    # draw up to current coordinate time
    for ln, x, y, t in zip(lines, Xs, Ys, Ts):
        if len(t) == 0:
            ln.set_data([], [])
            continue
        idx = np.searchsorted(t, frame_t, side="right")
        ln.set_data(x[:idx], y[:idx])
    time_text.set_text(f"t = {frame_t:.3f}")
    return (*lines, time_text)

# choose number of frames and timing
n_frames = 300
frame_times = np.linspace(0.0, t_end, n_frames)

anim = animation.FuncAnimation(
    fig, update, frames=frame_times, init_func=init, blit=True, interval=20
)

plt.show()
anim.save("radial_photon.gif", writer="imagemagick", fps=30)
''' --- 2D plotting --- 
fig, (ax_xy, ax_rt) = plt.subplots(1, 2, figsize=(12, 6))
ax_xy.set_aspect('equal', adjustable='box')

# left: trajectory in the orbital plane
for xs, ys in zip(X, Y):
    ax_xy.scatter(xs, ys, s=1)

theta = np.linspace(0, 2*np.pi, 400)
ax_xy.plot(rh*np.cos(theta), rh*np.sin(theta),
           linestyle="--", linewidth=1.0, color='k', alpha=0.7)
ax_xy.set_xlim(-rh - r0, rh + r0)
ax_xy.set_ylim(-rh - r0, rh + r0)
ax_xy.set_xlabel("x")
ax_xy.set_ylabel("y")
ax_xy.set_title("Photon trajectory in orbital plane")

# right: radius vs Schwarzschild coordinate time
for xs, ys, ts in zip(X, Y, Tcoord):
    rvals = np.sqrt(np.array(xs)**2 + np.array(ys)**2)
    ax_rt.plot(ts, rvals, lw=1)
ax_rt.axhline(rh, ls="--", color="k", alpha=0.7)
ax_rt.set_xlabel("t (Schwarzschild coordinate time)")
ax_rt.set_ylabel("r")
ax_rt.set_title("Radial coordinate vs time")

plt.tight_layout()
plt.show()
'''
 
''' --- 3D plotting ---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for xs, ys, ts in zip(X, Y, Tcoord):
    ax.plot(xs, ys, ts, lw=1)

# horizon as a cylinder at r = 2M
theta = np.linspace(0, 2*np.pi, 200)
z = np.linspace(0, max(max(ts) for ts in Tcoord), 200)
Theta, Z = np.meshgrid(theta, z)
Xh = rh * np.cos(Theta)
Yh = rh * np.sin(Theta)
Zh = Z
ax.plot_surface(Xh, Yh, Zh, color='k', alpha=0.2, linewidth=0)
ax.set_xlim(-r0, r0)
ax.set_ylim(-r0, r0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("t (Schwarzschild coord. time)")
ax.set_title("Photon trajectory in 3D (t as vertical axis)")

plt.tight_layout()
plt.show()
'''