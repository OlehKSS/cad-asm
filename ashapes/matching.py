import numpy as np
import ashapes.PCA.pca as pca

# Path to aligned data
path = "./data/hand/shapes/shapes_norm.txt"

shapes_norm = np.loadtxt(path, np.float32)

no_points = 56
no_shapes = 40

# Initialise the shape parameters, b, to zero (the mean shape)
b = np.zeros((112, 1))

# Generate the model point positions using x = x_bar + P * b
x_bar, largest_evals, P = pca(shapes_norm)
b = np.transpose(P) @ (shapes_norm[:, 0] - x_bar)

# should be done with protocol 1 pg 9
