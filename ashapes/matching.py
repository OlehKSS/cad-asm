import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from math import sqrt

from visualizer import display_normals, update_line, draggable_hand
from utils import transform_shape, move_to_origin, similarity_trans, affine_trans, no_points, no_shapes, \
    find_current_points, find_current_points_r

def match_average_shape(path_shapes_norm, path_test_img):
    # previously normalized shapes
    shapes_norm = np.loadtxt(path_shapes_norm, np.float32)
    img = cv2.imread(path_test_img, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    normals_length = 20

    # Image gradient, we can try
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # normalize
    grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

    # approximate transformation of the mean shape into image coordinate system
    trans_init = (350, 350)
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

    plt.subplot(121)
    plt.imshow(grad_mag, cmap="gray")
    hl, = plt.plot([], [], 'o-')
    plt.title("Current shape")

    plt.subplot(122)
    plt.imshow(grad_mag, cmap="gray")
    nl, = plt.plot([], [], 'o-')
    plt.title("Landmark points")

    it = 0
    min_diff = 0.5
    max_it = 100

    while True:
        # should be done with protocol 1 pg 9
        Y = find_current_points_r(grad_mag, x_mean_scaled, normals_length, display_normals=False, img=None)

        # Update x with new hand approximation
        x = x_mean + P @ b
        # find the transformation that maps points in fixed Y to moving x
        # return x and Y centered at the origin,
        # transformation matrix, and translation
        x, Y, T, tx, ty = similarity_trans(x, Y)
        # T, tx, ty = affine_trans(Y, x)

        # Project Y into the model co-ordinate frame by inverting T
        # y = transform_shape(T, Y)
        y = transform_shape(Y, inv(T))

        # Project y into the tangent plane to xbar (x_mean) by scaling
        y_prime = y / (np.dot(y[:, 0], x_mean))

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
        x_mean_scaled_new = transform_shape(x_temp, T, (tx, ty))

        update_line(hl, x_mean_scaled_new)
        update_line(nl, transform_shape(Y, np.eye(2), (tx, ty)))

        average_diff = np.linalg.norm(x_mean_scaled_new - x_mean_scaled) / sqrt(2 * no_points)

        print("Iteration {},\nAverage diff {}".format(it, average_diff))

        if average_diff < min_diff or it >= max_it:
            break

        it += 1
        x_mean_scaled = x_mean_scaled_new
    plt.show()


def match_average_shape_pyr(path_shapes_norm, path_test_img):
    # previously normalized shapes
    shapes_norm = np.loadtxt(path_shapes_norm, np.float32)
    img = cv2.imread(path_test_img, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    normals_length = 25

    # Image gradient, we can try
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # normalize
    grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

    # approximate transformation of the mean shape into image coordinate system
    trans_init = (350, 350)
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

    # minimum number of levels is 1
    no_levels = 3
    x_mean_scaled = x_mean_scaled / (2 ** (no_levels - 1))
    pyr_normals_length = int(normals_length / 2 ** (no_levels - 1))
    gaussian_pyr = [img]
    pyr_temp = img
    # create a pyramid
    for i in range(no_levels - 1):
        pyr_temp = cv2.pyrDown(pyr_temp)
        gaussian_pyr.append(pyr_temp)

    for pyr_index, pyr_temp in enumerate(reversed(gaussian_pyr)):
        # find gradient for a level
        sobelx = cv2.Sobel(pyr_temp, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(pyr_temp, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # normalize
        grad_mag = (grad_mag - grad_mag.min()) / grad_mag.max()

        plt.subplot(121)
        plt.imshow(grad_mag, cmap="gray")
        hl, = plt.plot([], [], 'o-')
        plt.title("Current shape")

        plt.subplot(122)
        plt.imshow(grad_mag, cmap="gray")
        nl, = plt.plot([], [], 'o-')
        plt.title("Landmark points")

        it = 0
        min_diff = 0.5
        max_it = 100

        while True:
            # should be done with protocol 1 pg 9
            Y = find_current_points_r(grad_mag, x_mean_scaled, pyr_normals_length, display_normals=False, img=None)

            # Update x with new hand approximation
            x = x_mean + P @ b
            # find the transformation that maps points in fixed Y to moving x
            # return x and Y centered at the origin,
            # transformation matrix, and translation
            x, Y, T, tx, ty = similarity_trans(x, Y)
            # T, tx, ty = affine_trans(Y, x)

            # Project Y into the model co-ordinate frame by inverting T
            # y = transform_shape(T, Y)
            y = transform_shape(Y, inv(T))

            # Project y into the tangent plane to xbar (x_mean) by scaling
            y_prime = y / (np.dot(y[:, 0], x_mean))

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
            x_mean_scaled_new = transform_shape(x_temp, T, (tx, ty))

            update_line(hl, x_mean_scaled_new)
            update_line(nl, transform_shape(Y, np.eye(2), (tx, ty)))

            average_diff = np.linalg.norm(x_mean_scaled_new - x_mean_scaled) / sqrt(2 * no_points)

            print("Iteration {},\nAverage diff {}".format(it, average_diff))

            if average_diff < min_diff or it >= max_it:
                break

            it += 1
            x_mean_scaled = x_mean_scaled_new
        plt.show()

        # scale up in order to go to the other level of the pyramid
        if pyr_index < (no_levels - 1):
            x_mean_scaled *= 2
            pyr_normals_length *= 2


def pca(shapes_norm):
    """
    Performs PCA on the normalized shapes.
    :param shapes_norm: array of coordinates of normalized shapes.
    :return:
    """
    # Mean and covariance
    mean_shape = np.mean(shapes_norm, axis=1)
    mean_shape = np.reshape(mean_shape,(len(mean_shape),1))
    cov_shape = np.cov(shapes_norm)

    # E-values E-vectors
    evals, evecs = np.linalg.eig(cov_shape)
    real_evals = np.real(evals)
    real_evecs = np.real(evecs)

    # total variance of the data
    v_t = np.sum(real_evals)
    #print(v_t)

    # the proportion of the total variation one wishes to explain
    f_v = 0.98

    # Choose e-values larger than f_v * v_t
    cumu_evals = real_evals[:20]
    cumu_evals = np.cumsum(cumu_evals)
    eval_stop = v_t * f_v
    i_largest_evals = np.argmax(cumu_evals >= eval_stop)

    largest_evals = real_evals[:i_largest_evals + 1]
    largest_evecs = real_evecs[:i_largest_evals + 1, :]

    return mean_shape, largest_evals, largest_evecs
