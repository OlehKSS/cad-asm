import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils import no_points, no_shapes, get_images


def align(path_shapes, path_images, path_shapes_norm=None, test_img_name=48):
    """
    Align all the shapes and normalize their coordinates excluding test image.
    :param path_shapes: a path to the file with shapes coordinates.
    :param path_images: a path to the folder that stores original images.
    :param path_shapes_norm: a full path to a file where normalized shapes coordinates should be stored.
    :param test_img_name: a number that is a name of the test image.
    :return:
    """
    # min difference, after which we stop iterations
    min_diff = 0.025

    shapes = np.loadtxt(path_shapes, np.float32) * 600

    imgs = get_images(path_images)
    imgs = sorted([int(name) for name in imgs])

    # set test image and remove from shapes array
    test_img_indx = imgs.index(test_img_name)
    test_img_shape = shapes[:, test_img_indx]
    test_img_path = os.path.join(path_images, str(test_img_name).zfill(4) + ".jpg")
    test_img = cv2.imread(test_img_path, 0)

    shapes = np.delete(shapes, test_img_indx, 1)

    # Plotting shapes (unaligned)
    # plt.figure()
    # plt.plot(shapes[:no_points, :], shapes[no_points:, :])

    # Normalize shapes to have origin at COM
    x_mean = np.mean(shapes[:no_points, :], axis=0).reshape([1, no_shapes])
    norm = np.linalg.norm(shapes, axis=0).reshape([1, no_shapes])
    y_mean = np.mean(shapes[no_points:, :], axis=0).reshape([1, no_shapes])
    shapes_norm = np.zeros(shapes.shape)
    shapes_norm[:no_points, :] = (shapes[:no_points, :] - x_mean) / norm
    shapes_norm[no_points:, :] = (shapes[no_points:, :] - y_mean) / norm

    # Plotting shapes (COM)
    # plt.figure()
    # plt.plot(shapes_norm[:no_points, :], shapes_norm[no_points:, :])

    # Find mean shape for starting point iterations
    mean_shape = np.mean(shapes_norm, axis=1).reshape([2 * no_points])

    x2 = mean_shape / (np.linalg.norm(mean_shape))
    # x2 = (shapes_norm[:, 0]/np.linalg.norm(mean_shape)).reshape([2 * no_points])

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
        new_mean_shape = np.mean(shapes_new, axis=1).reshape([2 * no_points])
        # new_mean_shape = new_mean_shape / (np.linalg.norm(new_mean_shape))

        # Checking time
        mean_diff = new_mean_shape - x2
        norm = np.linalg.norm(mean_diff)
        print("Iter. No.: {}, Norm: {}".format(iteration_no, norm))
        done = True if norm < min_diff else False

        # Reassignment for next iteration
        shapes_norm = shapes_new
        x2 = new_mean_shape
        iteration_no += 1
        break

    # Plotting shapes (aligned)
    # plt.figure()
    # plt.plot(shapes_norm[:no_points, :], shapes_norm[no_points:, :])
    # plt.show()

    if path_shapes_norm is not None:
        np.savetxt(path_shapes_norm, shapes_norm)
    
    return shapes_norm, test_img_shape
