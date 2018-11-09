import numpy as np
from math import sqrt

no_points = 56
no_shapes = 40


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
    Move a shape to its COM
    :param x: shape parameters
    :return x: shape paramters moved to origin
    tx: offset in x
    ty: offset in y

    """
    # Correct x so it is centered at (0,0)
    tx = np.mean(x[:no_points, :])
    ty = np.mean(x[no_points:, :])
    x[:no_points, :] = (x[:no_points, :] - tx)
    x[no_points:, :] = (x[no_points:, :] - ty)
    return x, tx, ty

def similarity_trans(x, corr_pos):
    """

    :param x:
    :param corr_pos: Y
    :return: x: x
             corr_pos:
             T: transformation matrix
             tx: translation in x
             ty: translation in y
    """
    x, _, _ = move_to_origin(x)
    corr_pos, tx, ty = move_to_origin(corr_pos)
    a_t = float((np.transpose(x) @ corr_pos) / (np.transpose(x) @ x))
    b_t = float(
        (np.dot(x[:no_points, 0], corr_pos[no_points:, 0]) - np.dot(x[no_points:, 0], corr_pos[:no_points, 0])) / (
                np.transpose(x) @ x))
    T = np.array([[a_t, -b_t], [b_t, a_t]])
    return x, corr_pos, T, tx, ty

def affine_trans(xf, xm):
    """

        :param x fixed
        :param x moving
        :return: x: x
                 corr_pos:
                 T: transformation matrix
                 tx: translation in x
                 ty: translation in y
        """
    xf, _, _ = move_to_origin(xf)
    S_xf = np.mean(xf[:no_points, 0])
    S_yf = np.mean(xf[no_points:, 0])

    S_xx = np.mean(xm[:no_points, 0] * xm[:no_points, 0])
    S_yy = np.mean(xm[no_points:, 0] * xm[no_points:, 0])

    S_xy = np.mean(xm[:no_points, 0] * xm[no_points:, 0])

    S_xxf = np.mean(xm[:no_points, 0] * xf[:no_points, 0])
    S_yxf = np.mean(xm[no_points:, 0] * xf[:no_points, 0])
    S_xyf = np.mean(xm[:no_points, 0] * xf[no_points:, 0])
    S_yyf = np.mean(xm[no_points:, 0] * xf[no_points:, 0])


    tx = S_xf
    ty = S_yf
    delta = (S_xx * S_yy) - S_xy**2
    T = 1/delta * (np.array([[S_xxf, S_yxf], [S_xyf, S_yyf]]) @ np.array([[S_yy, -S_xy], [-S_xy, S_xx]]))

    return T, tx, ty


def find_current_points(grad_mag, x_mean_scaled, normals_length, display_normals = False, img = None):
    """
    Calculate normals and then coordinates of current found points Y
    :param grad_mag: gradient magnitude image
    :param x_mean_scaled: mean hand image in image coordinate system
    :param normals_length: half-length of normal to look for max magnitude
    :param display_normals: boolean to display image with normals
    :param img: required to display normals, otherwise gradient image is used
    :return: Y: coordinates of current found points Y
    """
    if img is None:
        img = grad_mag

    # Norm Calculation
    normals = np.zeros((no_points, 2))
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
    if display_normals:
        display_normals(normals, x_mean_scaled, img, normals_length)

    # finding correct positions on the model, Y
    Y = np.zeros(x_mean_scaled.shape)

    for i in range(0, no_points):
        # landmark point coordinates
        px = float(x_mean_scaled[i])
        py = float(x_mean_scaled[i + no_points])
        nx, ny = normals[i, :]
        max_y, max_x = grad_mag.shape
        check_pixels = []
        for t in range(-normals_length, +normals_length):
            lx = px + t * nx
            ly = py + t * ny
            # check boundaries
            if lx >= max_x:
                lx = max_x - 1
            elif lx < 0:
                lx = 0

            if ly >= max_y:
                ly = max_y - 1
            elif ly < 0:
                ly = 0

            check_pixels.append((lx, ly))

        check_pixels = np.array(check_pixels, dtype=int)

        grad_values = grad_mag[check_pixels[:, 1], check_pixels[:, 0]]
        mag_argmax = np.argmax(grad_values)
        Y[i] = check_pixels[mag_argmax, 0]
        Y[i + no_points] = check_pixels[mag_argmax, 1]

    return Y