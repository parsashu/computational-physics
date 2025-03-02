import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit


time = 100000
L = 200

# Create an array of 200 zeros for height tracking
surface = np.zeros(L)
# Create a 3D array to track particle colors (0: no particle, 1: blue, 2: light blue)
particle_colors = np.zeros((1, L), dtype=int)

# Create arrays to track width over time
w_array = np.zeros(time)


# Function to add a particle and track its color
def add_particle(position, color_value):
    global particle_colors
    height = int(surface[position])

    # Expand particle_colors array if needed
    if height >= particle_colors.shape[0]:
        additional_height = particle_colors.shape[0]  # Double the size
        zeros_to_append = np.zeros((additional_height, L), dtype=int)
        particle_colors = np.vstack((particle_colors, zeros_to_append))

    # Wrap boundaries
    if position == L - 1:  # position = 199
        previous = position - 1
        next = 0
    else:
        previous = position - 1
        next = position + 1

    # Add the particle with its color
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


# Deposit particles with alternating colors
for i in range(time):
    # Choose a random integer between 0 and 199
    random_position = np.random.randint(0, L)

    # Determine color based on deposition time
    if (i // (time / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    # Add the particle
    add_particle(random_position, color)

    w_array[i] = calculate_width()


# Create a log-log plot of width vs time
plt.figure(figsize=(10, 6))
# Filter out any zero values to avoid log(0) errors
nonzero_indices = w_array > 0
log_time = np.log10(np.arange(1, time + 1)[nonzero_indices])
log_width = np.log10(w_array[nonzero_indices])
plt.plot(log_time, log_width, "bo", alpha=0.5, markersize=2)
plt.xlabel("log(Number of Deposited Particles)")
plt.ylabel("log(Surface Width)")
plt.title("Log-Log Plot of Surface Width vs Time")
plt.grid(True)
plt.show()


# Define the power law function: w(t) = A * t^beta
def power_law(t, A, beta):
    return A * t**beta


# Use only the latter part of the data for fitting (after initial transient)
# fit_start = time // 10  # Start fitting from 10% of the data
# x_data = np.arange(fit_start, time)
# y_data = w_array[fit_start:]

# # Perform the curve fitting
# params, covariance = curve_fit(power_law, x_data, y_data)
# A_fit, beta_fit = params
# beta_error = np.sqrt(np.diag(covariance))[1]  # Extract the error in beta

# # Generate the fitted curve
# y_fit = power_law(x_data, A_fit, beta_fit)

# # Plot the fitted curve
# plt.plot(x_data, y_fit, "b--", label=f"Fitted: t^{beta_fit:.3f}±{beta_error:.3f}")
# plt.legend()

# print(f"Fitted growth exponent (beta): {beta_fit:.4f} ± {beta_error:.4f}")
# print(f"Amplitude (A): {A_fit:.6e}")
plt.show()

# Create a visualization
max_height = int(np.max(surface))

# Create a custom colormap: 0=white, 1=blue, 2=light blue
colors = ["white", "blue", "skyblue"]
cmap = ListedColormap(colors)

# Plot the pixel-based surface
plt.figure(figsize=(10, 6))

# Trim the particle_colors array to only include the heights we need
particle_colors_trimmed = particle_colors[:max_height, :]

plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
plt.xlabel("Position")
plt.ylabel("Height")

# Set y-ticks to show actual heights (no need for custom calculation now)
num_ticks = 5  # Adjust this for more or fewer ticks
plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

plt.title("Buttom-up Deposition Model")
plt.show()
