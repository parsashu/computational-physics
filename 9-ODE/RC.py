import numpy as np
import matplotlib.pyplot as plt

R = 1e3
C = 1e-6
V0 = 5
tau = R * C

T = 0.01
dt = 0.0001
n_steps = int(T / dt)

t = np.linspace(0, T, n_steps)
V = np.zeros(n_steps)
V[0] = V0

for i in range(n_steps - 1):
    dV = -V[i] / tau * dt
    V[i + 1] = V[i] + dV

plt.plot(t, V, label="RC Voltage")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title(f"RC Voltage (dt = {dt}s)")
plt.legend()
plt.grid(True)
plt.show()
