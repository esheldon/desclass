import numpy as np
from numba import njit
from . import cem
from .cem import gauss_set, gauss_eval_scalar
from .interp import smooth_data_hann3, interp_gp


def calculate_prob(data, rmag, conc):

    rmag = np.array(rmag, dtype='f8', copy=False)
    conc = np.array(conc, dtype='f8', copy=False)

    gmixes = data['gmix']
    ngauss = gmixes['mean'].shape[1]

    centers = data['rmag']

    gal_sums = np.zeros(rmag.size)
    star_sums = np.zeros(rmag.size)

    gauss = np.zeros(1, dtype=cem.GAUSS_DTYPE)[0]

    for igauss in range(ngauss):
        nums = gmixes['num'][:, igauss]
        means = gmixes['mean'][:, igauss]
        sigmas = gmixes['sigma'][:, igauss]

        ystd = (smooth_data_hann3(nums) - nums).std()
        nums_interp, _ = interp_gp(centers, nums, ystd, rmag)

        ystd = (smooth_data_hann3(means) - means).std()
        means_interp, _ = interp_gp(centers, means, ystd, rmag)

        if igauss < 2:
            ysend = sigmas
        else:
            ysend = smooth_data_hann3(sigmas)

        ystd = (smooth_data_hann3(sigmas) - sigmas).std()
        sigmas_interp, _ = interp_gp(centers, ysend, ystd, rmag)

        if igauss < 2:
            sum_interpolated_gauss(
                nums_interp, means_interp, sigmas_interp, conc, star_sums,
                gauss,
            )
        else:
            sum_interpolated_gauss(
                nums_interp, means_interp, sigmas_interp, conc, gal_sums,
                gauss,
            )

    tot_sums = gal_sums + star_sums

    w, = np.where(tot_sums > 0)
    prob_gal = np.zeros(rmag.size)
    prob_star = np.zeros(rmag.size)

    prob_gal[w] = gal_sums[w]/tot_sums[w]
    prob_star[w] = star_sums[w]/tot_sums[w]

    return prob_gal, prob_star


@njit
def sum_interpolated_gauss(nums, means, sigmas, conc, vals, gauss):

    for i in range(nums.size):
        gauss_set(
            gauss,
            nums[i],
            means[i],
            sigmas[i],
        )
        vals[i] += gauss_eval_scalar(gauss, conc[i])
