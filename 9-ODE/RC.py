import numpy as np
import matplotlib.pyplot as plt

R = 1e3
C = 1e-6
Q0 = 5
tau = R * C

T = 0.01
dt = 0.0001
n_steps = int(T / dt)

t = np.linspace(0, T, n_steps)
Q = np.zeros(n_steps)
Q[0] = Q0

for i in range(n_steps - 1):
    dQ = -Q[i] / tau * dt
    Q[i + 1] = Q[i] + dQ

plt.plot(t, Q, label="RC Charge")
plt.xlabel("Time (s)")
plt.ylabel("Charge (C)")
plt.title(f"RC Charge (dt = {dt}s)")
plt.legend()
plt.grid(True)
plt.show()
