import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit


N = 20000
L = 100

surface = np.zeros(L)
particle_colors = np.zeros((1, L), dtype=int)
w_array = np.zeros(N)


def add_particle(position, color_value):
    """Function to add a particle and track its color"""
    global particle_colors
    height = int(surface[position])

    if height >= particle_colors.shape[0]:
        additional_height = particle_colors.shape[0]  # Double the size
        zeros_to_append = np.zeros((additional_height, L), dtype=int)
        particle_colors = np.vstack((particle_colors, zeros_to_append))

    # Periodic boundaries
    if position == L - 1:  # position = 199
        previous = position - 1
        next = 0
    else:
        previous = position - 1
        next = position + 1

    if surface[next] < surface[position]:
        height = int(surface[next])
        particle_colors[height, next] = color_value
        surface[next] += 1

    elif surface[previous] < surface[position]:
        height = int(surface[previous])
        particle_colors[height, previous] = color_value
        surface[previous] += 1

    else:
        particle_colors[height, position] = color_value
        surface[position] += 1


def calculate_width():
    mean_height = np.mean(surface)
    mean_height_squared = np.mean(surface**2)
    w = (mean_height_squared - mean_height**2) ** 0.5
    return w


for i in range(N):
    random_position = np.random.randint(0, L)

    if (i // (N / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    add_particle(random_position, color)
    w_array[i] = calculate_width()


#--------------------------------
nonzero_indices = np.where(w_array > 0)[0]
time_data = nonzero_indices + 1  # +1 to avoid log(0)
width_data = w_array[nonzero_indices]

log_time = np.log10(time_data)
log_width = np.log10(width_data)

plt.plot(log_time, log_width, "b.", alpha=0.5, label="Data")

constant_value = np.mean(log_width)

x_fit = np.linspace(log_time[0], log_time[-1], 100)
y_fit = np.full_like(x_fit, constant_value)

plt.plot(
    x_fit,
    y_fit,
    "r-",
    label=f"Constant Fit: y = {constant_value:.3f}",
)

plt.xlabel("log(Number of Deposited Particles)")
x_min, x_max = plt.xlim()
plt.xticks(np.linspace(x_min, x_max, 15))

plt.ylabel("log(Surface Width)")

y_min, y_max = plt.ylim()
plt.yticks(np.linspace(y_min, y_max, 15))

plt.title(f"Log-Log Plot of Surface Width vs Number of Particles (L={L}, N={N})")

plt.grid(True)
plt.legend()
plt.show()


#--------------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(N), w_array, "r-", alpha=0.7)
plt.xlabel("Number of Deposited Particles")
plt.ylabel("Surface Width (w)")
plt.title(f"Surface Width Evolution in Random Deposition (Particles: {N})")
plt.grid(True)


def power_law(t, A, beta):
    return A * t**beta


fit_start = N // 10  # Start fitting from 10% of the data
x_data = np.arange(fit_start, N)
y_data = w_array[fit_start:]

params, covariance = curve_fit(power_law, x_data, y_data)
A_fit, beta_fit = params
beta_error = np.sqrt(np.diag(covariance))[1]

y_fit = power_law(x_data, A_fit, beta_fit)

plt.plot(x_data, y_fit, "b--", label=f"Fitted: t^{beta_fit:.3f}±{beta_error:.3f}")
plt.legend()

print(f"Fitted growth exponent (beta): {beta_fit:.4f} ± {beta_error:.4f}")
print(f"Amplitude (A): {A_fit:.6e}")
plt.show()


#--------------------------------
max_height = int(np.max(surface))

'''Create a custom colormap: 0=white, 1=blue, 2=light blue'''
colors = ["white", "blue", "skyblue"]
cmap = ListedColormap(colors)

plt.figure(figsize=(10, 6))

particle_colors_trimmed = particle_colors[:max_height, :]

plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
plt.xlabel("Position")
plt.ylabel("Height")

num_ticks = 5
plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

plt.title(f"Buttom-up Deposition Model (Particles: {N})")
plt.show()
