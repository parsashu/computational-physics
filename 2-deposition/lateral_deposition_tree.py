import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit


N = 100000
L = 500

# Create an array of 200 zeros for height tracking
surface = np.zeros(L)
surface[L // 2] = 1

# Create a 3D array to track particle colors (0: no particle, 1: blue, 2: light blue)
particle_colors = np.zeros((1, L), dtype=int)
particle_colors[0, L // 2] = 1

# Create arrays to track width over time
w_array = np.zeros(N)


# Function to add a particle and track its color
def add_particle(position, color_value):
    global particle_colors
    height = int(surface[position])

    # Wrap boundaries
    if position == L - 1:  # position = 199
        previous = position - 1
        next = 0
    else:
        previous = position - 1
        next = position + 1

    # Determine the maximum height needed
    max_height_needed = height
    if surface[next] > surface[position]:
        max_height_needed = int(surface[next])
    elif surface[previous] > surface[position]:
        max_height_needed = int(surface[previous])

    # Expand particle_colors array if needed
    while max_height_needed >= particle_colors.shape[0]:
        additional_height = particle_colors.shape[0]  # Double the size
        zeros_to_append = np.zeros((additional_height, L), dtype=int)
        particle_colors = np.vstack((particle_colors, zeros_to_append))

    # Add the particle with its color
    if surface[next] > surface[position]:
        max_height = int(surface[next])
        particle_colors[max_height, position] = color_value
        surface[position] += max_height - height

    elif surface[previous] > surface[position]:
        max_height = int(surface[previous])
        particle_colors[max_height, position] = color_value
        surface[position] += max_height - height

    else:
        if height != 0:
            particle_colors[height, position] = color_value
            surface[position] += 1


def calculate_width():
    i_last = np.where(surface != 0)[0][-1]
    i_first = np.where(surface != 0)[0][0]
    return i_last - i_first


# Deposit particles with alternating colors
for i in range(N):
    # Choose a random integer between 0 and 199
    random_position = np.random.randint(0, L)

    # Determine color based on deposition time
    if (i // (N / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    # Add the particle
    add_particle(random_position, color)

    w_array[i] = calculate_width()


# Plot the width evolution over time to find, beta N = 2000----------------------
plt.figure(figsize=(10, 6))
plt.plot(range(N), w_array, "r-", alpha=0.7)
plt.xlabel("Number of Deposited Particles")
plt.ylabel("Surface Width (w)")
plt.title(f"Surface Width Evolution in Lateral Deposition (N={N}, L={L})")
plt.grid(True)


# Define the linear function: w(t) = m*t + b
def linear_func(t, m, b):
    return m * t + b


# Use only the latter part of the data for fitting (after initial transient)
fit_start = N // 10  # Start fitting from 10% of the data
x_data = np.arange(fit_start, N)
y_data = w_array[fit_start:]

# Perform the linear fitting
params, covariance = curve_fit(linear_func, x_data, y_data)
m_fit, b_fit = params
m_error, b_error = np.sqrt(np.diag(covariance))

# Generate the fitted curve
y_fit = linear_func(x_data, m_fit, b_fit)

# Plot the fitted line
plt.plot(
    x_data, y_fit, "b--", label=f"Fitted: {m_fit:.3f}±{m_error:.3f}*t + {b_fit:.1f}"
)
plt.legend()

print(f"Fitted slope (m): {m_fit:.4f} ± {m_error:.4f}")
print(f"Y-intercept (b): {b_fit:.4f} ± {b_error:.4f}")
plt.show()


# Create a visualization-------------------------------------------------
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

plt.title(f"Lateral Deposition Model (N={N}, L={L})")
plt.show()
