import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 2
real_value = -6


def f(x):
    return x**3 - 5 * x


def shelep(N):
    Ns = 0
    ym = f(np.sqrt(5 / 3))

    for _ in range(N):
        x = np.random.uniform(a, b)
        y = np.random.uniform(0, ym)

        if y > f(x):
            Ns += 1

        I = (b - a) * ym * Ns / N
    return I


N = np.arange(1, 1000, 10)
ensemble_size = 100
I_values = []

for n in N:
    print(n)
    estimates = [shelep(int(n)) for _ in range(ensemble_size)]
    average_estimate = np.mean(estimates)
    I_values.append(abs(average_estimate - real_value))

plt.figure(figsize=(10, 6))
plt.plot(N, I_values, "b-", label="Shelep Integration (Ensemble Average)")
plt.xlabel("Number of Points (N)")
plt.ylabel("Integral Estimate")
plt.title(
    f"Difference between Shelep Integration and real value vs. Number of Points (Ensemble Size: {ensemble_size})"
)
plt.grid(True)
plt.legend()
plt.show()
