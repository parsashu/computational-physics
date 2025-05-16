import numpy as np
import matplotlib.pyplot as plt


k = 1
m = 1
T = 100
h = 0.0001
n_steps = int(T / h)


# Euler
def euler():
    t = np.linspace(0, T, n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = 1
    v[0] = 0

    for i in range(n_steps - 1):
        a = -k / m * x[i]
        v[i + 1] = v[i] + a * h
        x[i + 1] = x[i] + v[i] * h

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
    plt.title(f"Phase Space (h = {h}s)")
    plt.legend()
    plt.grid(True)
    plt.show()


euler()
