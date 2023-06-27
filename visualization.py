import numpy as np
import matplotlib.pyplot as plt


def visualize(f, hist, name=""):
    # Define the range of values for x and y
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = np.linspace(-3, 3, 1000)

    # Create a 2D meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the function values over the meshgrid
    Z = np.squeeze(f([X, Y]))

    # Plot the level lines of the function
    plt.contour(X, Y, Z, levels=30)

    # Plot the path taken by gradient descent
    plt.plot(*zip(*hist), 'o-', color='r')

    # Add a title to the plot
    plt.title(name)

    # Show the plot
    plt.show()
