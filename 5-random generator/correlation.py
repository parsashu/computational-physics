import random
import matplotlib.pyplot as plt
import numpy as np
from pseudorandom import rand_LCG

N = 1000000
n_ensemble = 100


def before4_rnd(N):
    numbers = []
    cache = []

    while len(numbers) < N:
        random_number = random.randint(0, 9)

        if random_number == 4 and len(cache) > 0:
            numbers.append(cache[-1])
            cache = []
        elif random_number == 4 and len(cache) == 0:
            numbers.append(4)
        else:
            cache.append(random_number)

    return numbers


def before4_LCG(N):
    init_numbers = rand_LCG(N)
    numbers = []

    for i in range(1, len(init_numbers)):
        if init_numbers[i] == 4:
            numbers.append(init_numbers[i - 1])

    return numbers


# numbers = before4_rnd(N)
numbers = before4_LCG(N)

plt.figure(figsize=(10, 6))
plt.hist(numbers, bins=range(11), align="left", rwidth=0.8)
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.title(f"Distribution of {N} Numbers Before 4 (0-9)")
plt.xticks(range(10))
plt.grid(axis="y", alpha=0.75)
plt.show()


def sigma(N):
    # numbers = before4_rnd(N)
    numbers = before4_LCG(N)
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

for i, n in enumerate(N_range):
    print(f"Processing {i+1}/{len(N_range)}: N = {int(n)}")
    sigma_avg = []
    for j in range(n_ensemble):
        if j % 100 == 0 and j > 0:
            print(f"  Completed {j}/{n_ensemble} ensembles")
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
plt.title(f"Sigma vs Sample Size N with 1/√N Fit - Before 4 (Ensembles: {n_ensemble})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
