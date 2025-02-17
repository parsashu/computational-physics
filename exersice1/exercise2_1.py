import numpy as np
import matplotlib.pyplot as plt

# Generate 100 x-values evenly spaced between 0 and 10
x = np.linspace(0, 10, 100)

# Define a linear function; here, y = 2x + 1
y = 2 * x + 1

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, color='blue')

plt.axis("off")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the triangle vertices
x = [0, 2, 1]  # x coordinates of the three points
y = [0, 0, 2]  # y coordinates of the three points

plt.fill(x, y, color='lightblue')  # Creates a filled triangle
plt.axis('off')  # Removes axes
plt.show()

#test

