import numpy as np
import matplotlib.pyplot as plt

L = 100
p = 0.618

grid = np.zeros((L, L))


def turn_on(grid, p):
    """Open sites with probability p"""
    random_values = np.random.random((L, L))
    grid = np.where(random_values < p, 1, 0)
    return grid


def renormalize_block(block):
    """Apply renormalization rule to a 2x2 block based on examples"""
    percolating_patterns = [
        np.array([[0, 1], [1, 1]]),
        np.array([[1, 0], [1, 1]]),
        np.array([[1, 1], [0, 1]]),
        np.array([[1, 1], [1, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 1], [1, 1]]),
    ]

    for pattern in percolating_patterns:
        if np.array_equal(block, pattern):
            return 1

    return 0


def renormalize(grid):
    """Renormalize the grid by applying the rule based on pattern matching"""
    L_old = grid.shape[0]
    L_new = L_old // 2
    new_grid = np.zeros((L_new, L_new))

    for i in range(L_new):
        for j in range(L_new):
            block = grid[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            new_grid[i, j] = renormalize_block(block)

    return new_grid


def probability(grid):
    """Calculate the probability of 1 in the grid"""
    L = grid.shape[0]
    return np.sum(grid) / (L * L)


def find_fixed_point(p_values):
    """Find the fixed point where p = p_prime"""
    p_primes = []
    for p in p_values:
        grid = np.zeros((L, L))
        grid_on = turn_on(grid, p)
        renormalized_grid = renormalize(grid_on)
        p_prime = probability(renormalized_grid)
        p_primes.append(p_prime)

    differences = p_values - p_primes

    fixed_points = []
    for i in range(len(differences) - 1):
        if differences[i] * differences[i + 1] <= 0:
            if differences[i] == differences[i + 1] == 0:
                fixed_points.append((p_values[i] + p_values[i + 1]) / 2)
            else:
                t = abs(differences[i]) / (
                    abs(differences[i]) + abs(differences[i + 1])
                )
                fixed_point = p_values[i] + t * (p_values[i + 1] - p_values[i])
                fixed_points.append(fixed_point)

    return fixed_points, p_values, p_primes


grid_on = turn_on(grid, p)
renormalized_grid = renormalize(grid_on)
p_prime = probability(renormalized_grid)

p_values = np.linspace(0, 1, 10)
fixed_points, all_p, all_p_prime = find_fixed_point(p_values)

plt.figure(figsize=(8, 6))
plt.plot(all_p, all_p_prime, "b-", label="p'(p)")
plt.plot(all_p, all_p, "r--", label="p = p'")

for i, fp in enumerate(fixed_points):
    idx = np.abs(all_p - fp).argmin()
    plt.scatter(
        [fp],
        [all_p_prime[idx]],
        color=["green", "orange", "purple"][i % 3],
        s=100,
        label=f"Fixed point {i+1}: p ≈ {fp:.5f}",
    )

plt.xlabel("p")
plt.ylabel("p'")
plt.legend()
plt.title("Renormalization Flow: Finding Fixed Points")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
colors_with_alpha = [(0.3, 0.3, 0.3, 1.0), (0, 0, 0.5, 1.0)]
cmap = plt.matplotlib.colors.ListedColormap(colors_with_alpha)
plt.imshow(grid_on, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Original Grid (L={L}, p={p})")

plt.subplot(1, 2, 2)
plt.imshow(renormalized_grid, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L // 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L // 2, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Renormalized Grid (L={L//2}, p'={p_prime:.3f})")
plt.tight_layout()
plt.show()
