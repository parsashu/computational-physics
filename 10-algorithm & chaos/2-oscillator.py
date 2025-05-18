import numpy as np
import matplotlib.pyplot as plt


k = 1
m = 1
T = 10000
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
def frog_jump(t, h=h, plot=True):
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


h = np.linspace(1.5, 0.001, 100)
x_true, v_true = theoretical_phase_space(t)

def phase_space_dist(x_true, v_true, x_sim, v_sim):
    dx = np.linalg.norm(x_true - x_sim)
    dv = np.linalg.norm(v_true - v_sim)
    return dx + dv

dist_euler = []
dist_euler_cromer = []
dist_frog_jump = []
dist_verlet = []
dist_velocity_verlet = []
dist_biman = []

for h_i in h:
    x_euler, v_euler = euler(t, h=h_i, plot=False)
    x_euler_cromer, v_euler_cromer = euler_cromer(t, h=h_i, plot=False)
    x_frog_jump, v_frog_jump = frog_jump(t, h=h_i, plot=False)
    x_verlet, v_verlet = verlet(t, h=h_i, plot=False)
    x_velocity_verlet, v_velocity_verlet = velocity_verlet(t, h=h_i, plot=False)
    x_biman, v_biman = biman(t, h=h_i, plot=False)
    
    dist_euler.append(phase_space_dist(x_true, v_true, x_euler, v_euler))
    dist_euler_cromer.append(phase_space_dist(x_true, v_true, x_euler_cromer, v_euler_cromer))
    dist_frog_jump.append(phase_space_dist(x_true, v_true, x_frog_jump, v_frog_jump))
    dist_verlet.append(phase_space_dist(x_true, v_true, x_verlet, v_verlet))
    dist_velocity_verlet.append(phase_space_dist(x_true, v_true, x_velocity_verlet, v_velocity_verlet))
    dist_biman.append(phase_space_dist(x_true, v_true, x_biman, v_biman))

plt.figure(figsize=(12, 6))
# plt.plot(h, dist_euler, label='Euler')
plt.plot(h, dist_euler_cromer, label='Euler-Cromer')
plt.plot(h, dist_frog_jump, label='Frog Jump')
plt.plot(h, dist_verlet, label='Verlet')
plt.plot(h, dist_velocity_verlet, label='Velocity Verlet')
plt.plot(h, dist_biman, label='Biman')
plt.xlabel('h')
plt.ylabel('Error')
plt.title('Algorithm Errors vs Time')
plt.legend()
plt.grid(True)
plt.show()
