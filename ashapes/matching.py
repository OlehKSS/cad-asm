import numpy as np
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

#Image gradient, we can try 
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
sobely = cv2.Sobel(img,cv2.CV_32F, 0, 1, ksize=5)
grad_mag = np.sqrt(sobelx**2 + sobely**2)
# normalize
grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

# plt.imshow(grad_mag, cmap="gray")
# plt.show()
# Initialise the shape parameters, b, to zero (the mean shape)
b = np.zeros((112, 1))

# Generate the model point positions using x = x_bar + P * b
x_mean, largest_evals, P = pca(shapes_norm)
x_mean_scaled = 30000 * x_mean + 300

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
# finding correct positions on the
corr_pos = np.zeros(x_mean_scaled.shape) 

for i in range(0, no_points):
    # ladnmark point coordinates
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

# x' = x + t*nx
