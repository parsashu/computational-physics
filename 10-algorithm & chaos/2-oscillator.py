import numpy as np
import matplotlib.pyplot as plt


k = 1
m = 1
T = 500
h = 0.5

x0 = 1
v0 = 0

n_steps = int(T / h)
t = np.linspace(0, T, n_steps)


def A(x):
    return -k / m * x


# Euler
def euler(t, h=h, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        a = A(x[i])
        v[i + 1] = v[i] + a * h
        x[i + 1] = x[i] + v[i] * h

    if plot:
        plt.plot(t, x, label="Euler")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Euler (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Euler (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v


# Euler-Cromer
def euler_cromer(t, h=h, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        a = A(x[i])
        v[i + 1] = v[i] + a * h
        x[i + 1] = x[i] + v[i + 1] * h

    if plot:
        plt.plot(t, x, label="Euler-Cromer")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Euler-Cromer (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Euler-Cromer (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v


# Frog jump
def frog_jump(t, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v_half = np.zeros(n_steps)

    # First step using Euler
    x[0] = x0
    a0 = A(x[0])
    v_half[0] = 0.5 * a0 * h

    for i in range(n_steps - 1):
        x[i + 1] = x[i] + v_half[i] * h
        a_new = A(x[i + 1])
        v_half[i + 1] = v_half[i] + a_new * h

    if plot:
        plt.plot(t, x, label="Frog Jump")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Frog Jump (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v_half, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Frog Jump (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v_half


# Verlet
def verlet(t, h=h, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    # Second step using Euler
    x[1] = x[0] + v[0] * h
    a1 = A(x[1])
    v[1] = v[0] + a1 * h

    for i in range(1, n_steps - 1):
        a = A(x[i])
        x[i + 1] = 2 * x[i] - x[i - 1] + a * h**2
        v[i + 1] = (x[i + 1] - x[i]) / (2 * h)

    if plot:
        plt.plot(t, x, label="Verlet")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Verlet (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Verlet (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v


# Velocity Verlet
def velocity_verlet(t, h=h, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        a = A(x[i])
        x[i + 1] = x[i] + v[i] * h + 0.5 * a * h**2

        a_new = A(x[i + 1])
        v[i + 1] = v[i] + 0.5 * (a + a_new) * h

    if plot:
        plt.plot(t, x, label="Velocity Verlet")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Velocity Verlet (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Velocity Verlet (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v

# Biman
def biman(t, h=h, plot=True):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    # First step using Euler
    x[1] = x[0] + v[0] * h
    a1 = A(x[1])
    v[1] = v[0] + a1 * h

    for i in range(1, n_steps - 1):
        a_old = A(x[i - 1])
        a = A(x[i])
        x[i + 1] = x[i] + v[i] * h + (4 * a - a_old) * h**2 / 6

        a_new = A(x[i + 1])
        v[i + 1] = v[i] + (2 * a_new + 5 * a - a_old) * h / 6

    if plot:
        plt.plot(t, x, label="Biman")
        plt.xlabel("Time (s)")
        plt.ylabel("X (m)")
        plt.title(f"Harmonic Oscillator using Biman (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(x, v, label="Phase Space")
        plt.xlabel("Position (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Phase Space Biman (h = {h}s)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, v

# euler(t)
# euler_cromer(t)
# frog_jump(t)
# verlet(t)
# velocity_verlet(t)
# biman(t)


# Algorithm Errors ----------------------------------------------------------------
def theoretical_phase_space(t):
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    omega = np.sqrt(k / m)

    for i in range(len(t)):
        x[i] = x0 * np.cos(omega * t[i])
        v[i] = -x0 * omega * np.sin(omega * t[i])
    return x, v


t = np.linspace(10_000, 20_000, 10_000)
x_true, v_true = theoretical_phase_space(t)
x_euler, v_euler = euler(t, plot=False)
x_euler_cromer, v_euler_cromer = euler_cromer(t, plot=False)
x_frog_jump, v_frog_jump = frog_jump(t, plot=False)
x_verlet, v_verlet = verlet(t, plot=False)
x_velocity_verlet, v_velocity_verlet = velocity_verlet(t, plot=False)
x_biman, v_biman = biman(t, plot=False)

def phase_space_distance(x_true, v_true, x_sim, v_sim):
    dx = x_true - x_sim
    dv = v_true - v_sim
    dist = np.sqrt(dx**2 + dv**2)
    return dist

dist_euler = phase_space_distance(x_true, v_true, x_euler, v_euler)
dist_euler_cromer = phase_space_distance(x_true, v_true, x_euler_cromer, v_euler_cromer)
dist_frog_jump = phase_space_distance(x_true, v_true, x_frog_jump, v_frog_jump)
dist_verlet = phase_space_distance(x_true, v_true, x_verlet, v_verlet)
dist_velocity_verlet = phase_space_distance(x_true, v_true, x_velocity_verlet, v_velocity_verlet)
dist_biman = phase_space_distance(x_true, v_true, x_biman, v_biman)

sample_rate = 100
t_sampled = t[::sample_rate]
dist_euler_cromer_sampled = dist_euler_cromer[::sample_rate]
dist_frog_jump_sampled = dist_frog_jump[::sample_rate]
dist_verlet_sampled = dist_verlet[::sample_rate]
dist_velocity_verlet_sampled = dist_velocity_verlet[::sample_rate]
dist_biman_sampled = dist_biman[::sample_rate]

plt.figure(figsize=(12, 6))
plt.plot(t_sampled, dist_euler_cromer_sampled, label='Euler-Cromer')
plt.plot(t_sampled, dist_frog_jump_sampled, label='Frog Jump')
plt.plot(t_sampled, dist_verlet_sampled, label='Verlet')
plt.plot(t_sampled, dist_velocity_verlet_sampled, label='Velocity Verlet')
plt.plot(t_sampled, dist_biman_sampled, label='Biman')
plt.xlabel('Time (s)')
plt.ylabel('Phase Space Distance')
plt.title('Algorithm Errors')
plt.legend()
plt.grid(True)
plt.show()
