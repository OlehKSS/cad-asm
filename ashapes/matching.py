import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from math import sqrt

from PCA import pca

# Path to aligned data
path = "./data/hand/shapes/shapes_norm.txt"
path_image = "./data/hand/images/0000.jpg"

shapes_norm = np.loadtxt(path, np.float32)
img = cv2.imread(path_image,0)

no_points = 56
no_shapes = 40
scale = 15
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
    x1 = mean[:no_points] + scale * normals[:, 0]
    x2 = mean[:no_points] - scale * normals[:, 0]
    y1 = mean[no_points:] + scale * normals[:, 1]
    y2 = mean[no_points:] - scale * normals[:, 1]

    plt.imshow(img, cmap = "gray")
    plt.plot(mean[:no_points], mean[no_points:], 'o', color='#F82A80')
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
        y[j], y[j + no_points] = (R @ np.array([corr_pos[j], corr_pos[j + no_points]])) + t
    return y


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
x_mean_scaled = 30000 * x_mean + 300
# Initialise the shape parameters, b, to zero (the mean shape)
b = np.zeros((len(largest_evals), 1))
P = np.transpose(P)

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

display_normals(normals, x_mean_scaled, img, scale)
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
        x = px + t * nx
        y = py + t * ny
        # check boundaries
        if x >= max_x:
            x = max_x -1
        elif x < 0:
            x = 0

        if y >= max_y:
            y = max_y - 1
        elif y < 0:
            y = 0

        check_pixels.append((x, y))

    check_pixels = np.array(check_pixels, dtype=int)
    
    grad_values = grad_mag[check_pixels[:, 0], check_pixels[:, 1]]
    mag_argmax = np.argmax(grad_values)
    corr_pos[i] = check_pixels[mag_argmax, 0]
    corr_pos[i + no_points] = check_pixels[mag_argmax, 1]

plt.imshow(img, cmap="gray")
plt.plot(corr_pos[:no_points], corr_pos[no_points:])
plt.show()

# Find correct T
# Correct x_mean_scaled so it is centered at (0,0)
x_mean_scaled = x_mean_scaled - 300
corr_pos -= 300
a_t = (x_mean @ corr_pos) / np.dot(x_mean, x_mean)
b_t = (np.dot(x_mean[:no_points], corr_pos[no_points:]) - np.dot(x_mean[no_points:],corr_pos[:no_points])) / np.dot(x_mean, x_mean)
T = np.array([[a_t, -b_t], [b_t, a_t]])

#Project Y (corr_pos) into the model co-ordinate frame by inverting T
#y = transform_shape(T, corr_pos)
y = transform_shape(corr_pos, inv(T))

#Project y into the tangent plane to xbar (x_mean) by scaling
y_prime = y / (np.dot(y, x_mean))

#Update model parameters to match to y_prime
b = np.transpose(P) @ (y_prime - x_mean)

print(b)
#Apply contraints to the parameters of b to ensure plausible shapes
for i in range(0,len(largest_evals)):
    if b[i] > 3 * sqrt(largest_evals[i]):
        b[i] = 3 * sqrt(largest_evals[i])
        print("too big")
    elif b[i] < - 3 * sqrt(largest_evals[i]):
        b[i] = - 3 * sqrt(largest_evals[i])
        print("too small")

t = P @ b
print(b)

x_mean_scaled = transform_shape(x_mean + P @ b, T, (300, 300))
plt.imshow(img, cmap="gray")
plt.plot(x_mean_scaled[:no_points], x_mean_scaled[no_points:])
plt.show()
