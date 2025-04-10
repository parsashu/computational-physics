import numpy as np
import matplotlib.pyplot as plt
import time

a = 0
b = 2


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

    return float(I), float(sigma), float(delta)


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

    return float(I), float(sigma), float(delta)


def compare_methods_with_plots():
    # Different sample sizes to test
    sample_sizes = [100, 1000, 10000, 100000, 1000000]

    # Arrays to store results
    simple_I = []
    simple_sigma = []
    simple_delta = []
    simple_times = []

    importance_I = []
    importance_sigma = []
    importance_delta = []
    importance_times = []

    # Collect data for each sample size
    for N in sample_sizes:
        print(f"Running with N = {N}...")

        # Simple sampling
        start_time = time.time()
        I, sigma, delta = simple_sampling(a, b, N)
        end_time = time.time()
        simple_I.append(I)
        simple_sigma.append(sigma)
        simple_delta.append(delta)
        simple_times.append(end_time - start_time)

        # Importance sampling
        start_time = time.time()
        I, sigma, delta = importance_sampling(a, b, N)
        end_time = time.time()
        importance_I.append(I)
        importance_sigma.append(sigma)
        importance_delta.append(delta)
        importance_times.append(end_time - start_time)

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparison of Monte Carlo Methods", fontsize=16)

    # Plot I vs N
    ax1.plot(sample_sizes, simple_I, "bo-", label="Simple Sampling")
    ax1.plot(sample_sizes, importance_I, "ro-", label="Importance Sampling")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Samples (N)")
    ax1.set_ylabel("Integral Estimate (I)")
    ax1.set_title("Integral Estimate vs Sample Size")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot sigma vs N
    ax2.plot(sample_sizes, simple_sigma, "bo-", label="Simple Sampling")
    ax2.plot(sample_sizes, importance_sigma, "ro-", label="Importance Sampling")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Samples (N)")
    ax2.set_ylabel("Standard Deviation (σ)")
    ax2.set_title("Standard Deviation vs Sample Size")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot delta vs N
    ax3.plot(sample_sizes, simple_delta, "bo-", label="Simple Sampling")
    ax3.plot(sample_sizes, importance_delta, "ro-", label="Importance Sampling")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Number of Samples (N)")
    ax3.set_ylabel("Error Estimate (Δ)")
    ax3.set_title("Error Estimate vs Sample Size")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot runtime vs N
    ax4.plot(sample_sizes, simple_times, "bo-", label="Simple Sampling")
    ax4.plot(sample_sizes, importance_times, "ro-", label="Importance Sampling")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Number of Samples (N)")
    ax4.set_ylabel("Runtime (seconds)")
    ax4.set_title("Runtime vs Sample Size")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig("monte_carlo_comparison.png", dpi=300)
    plt.show()


# Run the comparison
compare_methods_with_plots()

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
