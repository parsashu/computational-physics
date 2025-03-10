import numpy as np
import matplotlib.pyplot as plt
L = 10

grid = np.zeros((L, L))
grid[0, :] = 1

plt.figure(figsize=(6, 6))
colors = ['red', 'blue']
colors_with_alpha = [(1, 0, 0, 0.7), (0, 0, 1, 0.7)]  
cmap = plt.matplotlib.colors.ListedColormap(colors_with_alpha)
plt.imshow(grid, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)  
ax = plt.gca()
ax.set_xticks(np.arange(-.5, L, 1), minor=True)
ax.set_yticks(np.arange(-.5, L, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
plt.show()






