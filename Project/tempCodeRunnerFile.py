
colors = plt.cm.rainbow(np.linspace(0, 1, max_color))
colors[0] = (0.3, 0.3, 0.3, 1.0)
colors[1] = (0, 0, 0, 0)
cmap = ListedColormap(colors)
im2 = ax2.imshow(color_grid, cmap=cmap, vmin=0, vmax=max_color - 1)