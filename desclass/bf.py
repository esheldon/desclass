import numpy as np
import esutil as eu
from esutil.numpy_util import between
import hickory


SLOPE = 2.0
OFF = 16


def predict(rmag, amp):

    bf_conc = amp*(rmag - OFF)**SLOPE
    try:
        _ = len(bf_conc)
        w, = np.where(rmag < OFF)
        bf_conc[w] = 0.0

        w, = np.where(rmag > 25)
        bf_conc[w] = amp*(25 - OFF)**SLOPE

    except TypeError:
        if rmag < OFF:
            bf_conc = 0.0
        elif rmag > 25:
            bf_conc = amp*(25 - OFF)**SLOPE

    return bf_conc


def get_amp(rmag, conc):
    pv = predict(rmag, 1.0)
    amp = (conc/pv).mean()

    return amp


def main():

    stars = eu.io.read('hsc/opt-griz-hsc-stars.fits')

    w, = np.where((stars['conc_flags'] == 0))

    stars = stars[w]

    rmag = stars['psf_mag'][:, 1]
    h, rev = eu.stat.histogram(rmag, min=19, max=23, nbin=10, rev=True)

    psf_mag_mean = []
    conc_mean = []
    conc_err = []

    for i in range(len(h)):
        if h[i] > 0:

            w = rev[rev[i]:rev[i+1]]

            mn, std, err, ind = eu.stat.sigma_clip(
                stars['conc'][w],
                get_err=True, get_indices=True,
            )

            wsub = np.where(
                between(stars['conc'][w], mn - 4*std, mn + 4*std)
            )

            psf_mag_mean.append(rmag[w].mean())
            # conc_mean.append(mn)
            conc_mean.append(np.median(stars['conc'][w[wsub]]))
            conc_err.append(err)

    psf_mag_mean = np.array(psf_mag_mean)
    conc_mean = np.array(conc_mean)
    conc_err = np.array(conc_err)

    amp = get_amp(psf_mag_mean, conc_mean)
    print('amp:', amp)

    plt = hickory.Plot(
        # figsize=(12, 12*0.618),
        xlabel='r mag',
        ylabel='conc',
        legend=True,
    )

    label = r'$%.2g \times \mathrm{mag}^{2}$' % amp
    plt.errorbar(psf_mag_mean, conc_mean, conc_err, label='stars')
    plt.curve(psf_mag_mean, predict(psf_mag_mean, amp), label=label)

    plt.savefig('bf.pdf')


if __name__ == '__main__':
    main()
