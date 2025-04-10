import numpy as np
import matplotlib.pyplot as plt
import time
import tabulate

a = 0
b = 2
REAL_VALUE = 0.8820813907624215


def f(x):
    return np.exp(-(x**2))


def g(x):
    return np.exp(-x)


def random_gen_g(a, b):
    x = np.random.random()
    y = -np.log(x)
    return y


def simple_sampling(a, b, N):
    mean_f = 0
    mean_sqr_f = 0

    for _ in range(N):
        x = np.random.uniform(a, b)
        mean_f += f(x)
        mean_sqr_f += f(x) ** 2

    mean_f /= N
    mean_sqr_f /= N

    I = mean_f * (b - a)
    sigma = np.sqrt(mean_sqr_f - mean_f**2)
    delta = sigma / np.sqrt(N)
    real_value_difference = abs(I - REAL_VALUE)

    return float(I), float(sigma), float(delta), float(real_value_difference)


def importance_sampling(a, b, N):
    mean_f_over_g = 0
    mean_sqr_f_over_g = 0
    norm_const = np.exp(-a) - np.exp(-b)

    for _ in range(N):
        x = -1
        while not (a <= x <= b):
            x = random_gen_g(a, b)

        mean_f_over_g += f(x) / g(x)
        mean_sqr_f_over_g += ((f(x) / g(x))) ** 2

    mean_f_over_g /= N
    mean_sqr_f_over_g /= N

    I = mean_f_over_g * norm_const
    sigma = np.sqrt(mean_sqr_f_over_g - mean_f_over_g**2)
    delta = sigma / np.sqrt(N)
    real_value_difference = abs(I - REAL_VALUE)

    return float(I), float(sigma), float(delta), float(real_value_difference)


# Compare the two methods table
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
results = []

headers = [
    "N",
    "I_s",
    "I_i",
    "sigma_s",
    "sigma_i",
    "delta_s",
    "delta_i",
    "error_s",
    "error_i",
    "runtime_s",
    "runtime_i",
]


for N in sample_sizes:
    print(f"Running with N = {N}...")

    start_time = time.time()
    I_s, sigma_s, delta_s, error_s = simple_sampling(a, b, N)
    runtime_s = time.time() - start_time

    start_time = time.time()
    I_i, sigma_i, delta_i, error_i = importance_sampling(a, b, N)
    runtime_i = time.time() - start_time

    results.append(
        [
            N,
            I_s,
            I_i,
            sigma_s,
            sigma_i,
            delta_s,
            delta_i,
            error_s,
            error_i,
            runtime_s,
            runtime_i,
        ]
    )

print(tabulate.tabulate(results, headers=headers, floatfmt=".6f", tablefmt="grid"))


# Generate histogram of random numbers
N = 100000
samples = [random_gen_g(a, b) for _ in range(N)]

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, color="blue")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("Histogram of random numbers with distribution g(x)")
plt.grid(True, alpha=0.3)
plt.show()
