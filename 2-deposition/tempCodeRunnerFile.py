# Create a visualization
# max_height = int(np.max(surface))

# # Create a custom colormap: 0=white, 1=blue, 2=light blue
# colors = ["white", "blue", "skyblue"]
# cmap = ListedColormap(colors)

# # Plot the pixel-based surface
# plt.figure(figsize=(10, 6))

# # Trim the particle_colors array to only include the heights we need
# particle_colors_trimmed = particle_colors[:max_height, :]

# plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
# plt.xlabel("Position")
# plt.ylabel("Height")

# # Set y-ticks to show actual heights (no need for custom calculation now)
# num_ticks = 5  # Adjust this for more or fewer ticks
# plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

# plt.title("Buttom-up Deposition Model")
# plt.show()