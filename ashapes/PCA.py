import numpy as np
import matplotlib.pyplot as plt


def pca(shapes_norm):
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
