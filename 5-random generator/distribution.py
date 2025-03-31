import random
import matplotlib.pyplot as plt
import numpy as np

N = 10000
n_ensemble = 1000


def call_rnd(N):
    numbers = []
    for _ in range(N):
        random_number = random.randint(0, 9)
        numbers.append(random_number)
    return numbers


numbers = call_rnd(N)

plt.figure(figsize=(10, 6))
plt.hist(numbers, bins=range(11), align="left", rwidth=0.8)
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.title(f"Distribution of {N} Random Numbers (0-9)")
plt.xticks(range(10))
plt.grid(axis="y", alpha=0.75)
plt.show()


def sigma(N):
    numbers = call_rnd(N)
    sigma = np.zeros(10)

    for i in range(10):
        num_count = numbers.count(i)
        sigma[i] = num_count

    sigma -= N / 10
    sigma = sigma**2
    sigma = np.sum(sigma)
    sigma = np.sqrt(sigma) / N
    return sigma


N_range = np.linspace(100, 10000, 20)
sigma_list = []
for n in N_range:

    sigma_avg = []
    for _ in range(n_ensemble):
        sigma_avg.append(sigma(int(n)))

    sigma_avg = np.mean(sigma_avg)
    sigma_list.append(sigma_avg)

plt.figure(figsize=(10, 6))
plt.plot(N_range, sigma_list, "o-", markersize=3, label="Measured sigma")
coeff, _ = np.polyfit(1 / np.sqrt(N_range), sigma_list, 1)
theoretical_curve = coeff * (1 / np.sqrt(N_range))
plt.plot(N_range, theoretical_curve, "r--", label=f"{coeff:.4f}/√N")

plt.xlabel("N (Sample Size)")
plt.ylabel("Sigma")
plt.title("Sigma vs Sample Size N with 1/√N Fit")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
