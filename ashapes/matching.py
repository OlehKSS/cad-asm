import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from math import sqrt


from PCA import pca
from visualizer import display_normals, update_line, draggable_hand
from utils import transform_shape, move_to_origin, similarity_trans, affine_trans, no_points, no_shapes, find_current_points

# Path to aligned data
path = "./data/hand/shapes/shapes_norm.txt"
path_image = "./data/hand/images/0000.jpg"

# previously normalized shapes
shapes_norm = np.loadtxt(path, np.float32)
img = cv2.imread(path_image, 0)

normals_length = 25

#Image gradient, we can try
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
# normalize
grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

# approximate transformation of the mean shape into image coordinate system
trans_init = (300, 300)
x_mean, largest_evals, P = pca(shapes_norm)
x_mean_scaled = transform_shape(x_mean, 1400 * np.eye(2), trans_init)

# Ask user to improve alignment, return precisely aligned shape
dr = draggable_hand(grad_mag, x_mean_scaled)

x_mean_scaled = np.zeros_like(x_mean_scaled)
x_mean_scaled[:no_points] = dr.shape_x
x_mean_scaled[no_points:] = dr.shape_y

# Initialise b
b = np.zeros((len(largest_evals), 1))
# Change P indices for matrix multiplication
P = np.transpose(P)

# plt.subplot(121)
plt.imshow(grad_mag, cmap="gray")
hl, = plt.plot([], [])
nl, = plt.plot([], [])
#
# plt.subplot(122)
# plt.imshow(img, cmap="gray")
# gl, = plt.plot([], [])
# rl, = plt.plot([], [])
# ml, = plt.plot([], [])

for i in range(0, 50):
    # should be done with protocol 1 pg 9
    Y = find_current_points(grad_mag, x_mean_scaled, normals_length, display_normals=False, img=None)

    # Update x with new hand approximation
    x = x_mean + P @ b
    # find the transformation that maps points in fixed Y to moving x
    # return x and Y centered at the origin,
    # transformation matrix, and translation
    x, Y, T, tx, ty = similarity_trans(x, Y)
    #T, tx, ty = affine_trans(Y, x)

    # Project Y into the model co-ordinate frame by inverting T
    # y = transform_shape(T, Y)
    y = transform_shape(Y, inv(T))

    # Project y into the tangent plane to xbar (x_mean) by scaling
    y_prime = y / (np.dot(y[:, 0], x_mean))
    #y_prime = y

    # Update model parameters to match to y_prime
    b = np.transpose(P) @ (y_prime - x_mean)

    # Apply constraints to the parameters of b to ensure plausible shapes
    for i in range(0, len(largest_evals)):
        if b[i] > 3 * sqrt(largest_evals[i]):
            b[i] = 3 * sqrt(largest_evals[i])
        elif b[i] < - 3 * sqrt(largest_evals[i]):
            b[i] = - 3 * sqrt(largest_evals[i])

    x_temp = x_mean + P @ b
    # move new shape to the origin of the coordinate system
    x_temp, _, _ = move_to_origin(x_temp)
    x_mean_scaled = transform_shape(x_temp, T, (tx, ty))

    update_line(hl, x_mean_scaled)
    update_line(nl, transform_shape(Y, np.eye(2),(tx, ty)))
    #
    # update_line(gl, x)
    # update_line(rl, y)
    # update_line(ml, x_mean)

plt.show()
