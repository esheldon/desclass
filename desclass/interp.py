import numpy as np
from numba import njit


@njit
def smooth_data_hann3(y):
    window = np.array([0.5, 1., 0.5])
    num = window.size
    offsets = np.array([-1, 0, 1])

    sy = y.copy()

    for idata in range(1, y.size-1):
        ssum = 0.0
        ksum = 0.0

        for i in range(num):
            off = offsets[i]
            k = window[i]

            ssum += k*y[idata + off]
            ksum += k

        sy[idata] = ssum/ksum

    return sy


def fit_gp(x, y, yerr):
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import (
        Matern, WhiteKernel,
        # ConstantKernel,
    )

    # _, ystd = eu.stat.sigma_clip(smooth_data_hann3(y) - y)
    # ystd = (smooth_data_hann3(y) - y).std()

    spacing = x[1] - x[0]
    kernel = (
        # ConstantKernel() +
        Matern(length_scale=spacing, nu=5/2) +
        WhiteKernel(noise_level=yerr**2)
    )

    gp = gaussian_process.GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
    )

    X = x.reshape(-1, 1)
    gp.fit(X, y)

    return gp


def interp_gp(x, y, yerr, xinterp):

    gp = fit_gp(x, y, yerr)

    y_pred, sigma = gp.predict(
        xinterp.reshape(-1, 1),
        return_std=True,
    )

    return y_pred, sigma
