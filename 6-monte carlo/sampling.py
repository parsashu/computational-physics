import numpy as np
import matplotlib.pyplot as plt

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

    for _ in range(N):
        x = np.random.uniform(a, b)
        mean_f_over_g += f(x) / g(x)
        mean_sqr_f_over_g += (f(x) / g(x)) ** 2

    mean_f_over_g /= N
    mean_sqr_f_over_g /= N

    I = mean_f_over_g * (b - a)
    sigma = np.sqrt(mean_sqr_f_over_g - mean_f_over_g**2)
    delta = sigma / np.sqrt(N)

    return float(I), float(sigma), float(delta)


# print(simple_sampling(a, b, 1000000))
# print(importance_sampling(a, b, 1000000))

N = 100000
samples = [random_gen_g(a, b) for _ in range(N)]

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, color="blue")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("Histogram of random numbers with distribution g(x)")
plt.grid(True, alpha=0.3)
plt.show()
