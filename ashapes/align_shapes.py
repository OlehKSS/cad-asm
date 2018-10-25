"""Active Shapes Method in Python"""

import numpy as np
import matplotlib.pyplot as plt

# Path to hand data
path = "./data/hand/shapes/shapes.txt"

shapes = np.loadtxt(path, np.float32) * 600

no_points = 56
no_shapes = 40

# Plotting shapes (unaligned)
plt.figure()
plt.plot(shapes[:no_points, :], shapes[no_points:, :])

# Normalize shapes to have origin at COM
x_mean = np.mean(shapes[:no_points, :], axis=0).reshape([1, 40])
norm = np.linalg.norm(shapes, axis=0).reshape([1, 40])
y_mean = np.mean(shapes[no_points:, :], axis=0).reshape([1, 40])
shapes_norm = np.zeros(shapes.shape)
shapes_norm[:no_points, :] = (shapes[:no_points, :] - x_mean) / norm
shapes_norm[no_points:, :] = (shapes[no_points:, :] - y_mean) / norm

# Plotting shapes (COM)
plt.figure()
plt.plot(shapes_norm[:no_points, :], shapes_norm[no_points:, :])

# Find mean shape for starting point iterations
mean_shape = np.mean(shapes_norm, axis=1).reshape([112])

x2 = mean_shape/(np.linalg.norm(mean_shape))
# x2 = (shapes_norm[:, 0]/np.linalg.norm(mean_shape)).reshape([112])

done = False
iteration_no = 0
# Iterative part
while done is False:
    # x1 current, x2 mean
    shapes_new = np.zeros(shapes_norm.shape)
    for i in range(0, no_shapes):
        x1 = shapes_norm[:, i]

        a = (np.dot(x1, x2)) / ((np.linalg.norm(x1)) ** 2)
        b = (x1[:no_points] * x2[no_points:]) - (x1[no_points:] * x2[:no_points])
        b = np.sum(b) / ((np.linalg.norm(x1)) ** 2)

        s = np.sqrt(a ** 2 + b ** 2)
        theta = np.arctan(b / a)

        T = np.zeros((2, 2))
        T[0, 0] = s * np.cos(theta)
        T[1, 1] = T[0, 0]
        T[0, 1] = -s * np.sin(theta)
        T[1, 0] = s * np.sin(theta)

        for j in range(0, no_points):
            shapes_new[j, i], shapes_new[j + no_points, i] = T @ np.array([x1[j], x1[j + no_points]])

    new_mean_shape = shapes_norm[:, 0]
    new_mean_shape = np.mean(shapes_new, axis=1).reshape([112])
    #new_mean_shape = new_mean_shape / (np.linalg.norm(new_mean_shape))

    # Checking time
    mean_diff = new_mean_shape - x2
    norm = np.linalg.norm(mean_diff)
    print("Iter. No.: {}, Norm: {}".format(iteration_no, norm))
    done = True if norm < 0.001 else False

    # Reassignment for next iteration
    shapes_norm = shapes_new
    x2 = new_mean_shape
    iteration_no += 1

# Plotting shapes (aligned)
plt.figure()
plt.plot(shapes_norm[:no_points, :], shapes_norm[no_points:, :])
plt.show()

np.savetxt("./data/hand/shapes/shapes_norm.txt", shapes_norm)