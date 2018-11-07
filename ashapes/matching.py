import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.animation as animation
from matplotlib import style

from PCA import pca

# Path to aligned data
path = "./data/hand/shapes/shapes_norm.txt"
path_image = "./data/hand/images/0000.jpg"

shapes_norm = np.loadtxt(path, np.float32)
img = cv2.imread(path_image,0)

no_points = 56
no_shapes = 40
scale = 20
dimens = 2

def display_normals(normals, mean, img, scale=15):
    """
    Helper function for displaying normals.

    Input:
        normals (numpy.ndarray): array of normals for each mean.
        mean (numpy.ndarray): array of mean coordinates of landmarks.
        img (numpy.ndarray): object image.
        scale (int): determines how long should normal be on the plot
    """
    x1 = mean[:no_points, 0] + scale * normals[:, 0]
    x2 = mean[:no_points, 0] - scale * normals[:, 0]
    y1 = mean[no_points:, 0] + scale * normals[:, 1]
    y2 = mean[no_points:, 0] - scale * normals[:, 1]

    plt.imshow(img, cmap = "gray")
    plt.plot(mean[:no_points,0], mean[no_points:,0], 'o', color='#F82A80')
    plt.plot((x1, x2), (y1, y2))
    plt.show()

def transform_shape(corr_pos, R, t=(0, 0)):
    """
    Apply affine transformation to a shape.
    :param corr_pos: shape to be transformed.
    :param R: Rotation and scaling matrix
    :param tx: translation vector.
    :return:
    """
    y = np.zeros_like(corr_pos)
    t = np.array(t)
    for j in range(0, no_points):
        y[j, 0], y[j + no_points, 0] = (R @ np.array([corr_pos[j, 0], corr_pos[j + no_points, 0]])) + t
    return y

def move_to_origin(x):
    """

    :param x:
    :return: x, tx, ty:

    """
    # Correct x so it is centered at (0,0)
    tx = np.mean(x[:no_points, :])
    ty = np.mean(x[no_points:, :])
    x[:no_points, :] = (x[:no_points, :] - tx)
    x[no_points:, :] = (x[no_points:, :] - ty)
    return x, tx, ty

def update_line(hl, new_data):
    hl.set_xdata(new_data[:no_points])
    hl.set_ydata(new_data[no_points:])
    plt.pause(0.05)
    plt.draw()

#Image gradient, we can try 
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
sobely = cv2.Sobel(img,cv2.CV_32F, 0, 1, ksize=5)
grad_mag = np.sqrt(sobelx**2 + sobely**2)
# normalize
grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

# plt.imshow(grad_mag, cmap="gray")
# plt.show()

# Generate the model point positions using x = x_bar + P * b
x_mean, largest_evals, P = pca(shapes_norm)
x_mean_scaled = transform_shape(x_mean, 30000 * np.eye(2), t=(320, 350))
# Initialise the shape parameters, b, to zero (the mean shape)
b = np.zeros((len(largest_evals), 1))
P = np.transpose(P)

plt.imshow(img, cmap="gray")
hl, = plt.plot([], [])

for i in range (0,70):
    # should be done with protocol 1 pg 9
    # Norm Calculation
    normals = np.zeros((no_points, dimens))

    for i in range(0, no_points):
        # the first and the last points vectors should
        # be calculated using one neighbor
        if i == 0:
            ux = x_mean_scaled[i + 1] - x_mean_scaled[i]
            uy = x_mean_scaled[i + 1 + no_points] - x_mean_scaled[i + no_points]
        elif i == no_points - 1:
            ux = x_mean_scaled[i] - x_mean_scaled[i - 1]
            uy = x_mean_scaled[i + no_points] - x_mean_scaled[i + no_points - 1]
        else:
            # points that have two neighbors
            ux = x_mean_scaled[i + 1] - x_mean_scaled[i - 1]
            uy = x_mean_scaled[i + 1 + no_points] - x_mean_scaled[i - 1 + no_points]
        # nx, ny for each point
        normals[i, 0] = -uy / sqrt(ux * ux + uy * uy)
        normals[i, 1] = ux / sqrt(ux * ux + uy * uy)

    # PLOT normals
    # display_normals(normals, x_mean_scaled, img, scale)

    # finding correct positions on the model, Y
    corr_pos = np.zeros(x_mean_scaled.shape)

    for i in range(0, no_points):
        # landmark point coordinates
        px = x_mean_scaled[i]
        py = x_mean_scaled[i + no_points]
        nx, ny = normals[i, :]
        max_x, max_y = grad_mag.shape
        check_pixels = []
        for t in range(-scale, +scale):
            lx = px + t * nx
            ly = py + t * ny
            # check boundaries
            if lx >= max_x:
                lx = max_x -1
            elif lx < 0:
                lx = 0

            if ly >= max_y:
                ly = max_y - 1
            elif ly < 0:
                ly = 0

            check_pixels.append((lx, ly))

        check_pixels = np.array(check_pixels, dtype=int)

        grad_values = grad_mag[check_pixels[:, 0], check_pixels[:, 1]]
        mag_argmax = np.argmax(grad_values)
        corr_pos[i] = check_pixels[mag_argmax, 0]
        corr_pos[i + no_points] = check_pixels[mag_argmax, 1]

    # plt.imshow(img, cmap="gray")
    # plt.plot(corr_pos[:no_points], corr_pos[no_points:])
    # plt.show()

    # Find correct T
    # Correct x so it is centered at (0,0)
    x = x_mean + P @ b
    x, _, _ = move_to_origin(x)

    # tx, ty, scale, angle = find_pose_params(corr_pos, x)

    corr_pos, tx, ty = move_to_origin(corr_pos)
    a_t = float((np.transpose(x) @ corr_pos) / (np.transpose(x) @ x))
    b_t = float((np.dot(x[:no_points, 0], corr_pos[no_points:, 0]) - np.dot(x[no_points:, 0],corr_pos[:no_points, 0])) / (np.transpose(x) @ x))
    T = np.array([[a_t, -b_t], [b_t, a_t]])

    #Project Y (corr_pos) into the model co-ordinate frame by inverting T
    #y = transform_shape(T, corr_pos)
    y = transform_shape(corr_pos, inv(T))


    #Project y into the tangent plane to xbar (x_mean) by scaling
    # y_prime = y / (np.dot(y, x_mean))
    y_prime = y

    #Update model parameters to match to y_prime
    b = np.transpose(P) @ (y_prime - x_mean)

    #Apply contraints to the parameters of b to ensure plausible shapes
    for i in range(0,len(largest_evals)):
        if b[i] > 3 * sqrt(largest_evals[i]):
            b[i] = 3 * sqrt(largest_evals[i])
        elif b[i] < - 3 * sqrt(largest_evals[i]):
            b[i] = - 3 * sqrt(largest_evals[i])


    x_mean_scaled = transform_shape(x_mean + P @ b, T, (tx, ty))
    update_line(hl, x_mean_scaled)

plt.show()