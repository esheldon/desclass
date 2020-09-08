import numpy as np
import esutil as eu
from esutil.numpy_util import between
from hickory import Table
import argparse

MAGOFF = 13.5
INDEX = 1.5


def get_amp(*, psf_mag, conc, show=False, output=None):

    # magmin, magmax = 16.0, 19.5
    # nbin = 7
    magmin, magmax = 16.5, 19.5
    nbin = 6

    cmin, cmax = -0.0005, 0.0003
    w, = np.where(
        between(psf_mag, magmin, magmax)
        &
        between(conc, cmin, cmax)
    )

    hd = eu.stat.histogram(
        psf_mag[w],
        min=magmin, max=magmax, nbin=nbin,
        more=True,
    )
    herr = np.sqrt(hd['hist'])

    pv = (hd['center'] - MAGOFF)**INDEX
    amp = (hd['hist']/pv).mean()
    print('amp:', amp)

    # alpha = 0.5
    plt = Table(2, 1, figsize=(8, 7))

    plt[0].plot(
        psf_mag,
        conc,
        marker='.',
        alpha=0.5,
    )
    plt[0].plot(
        psf_mag[w],
        conc[w],
        marker='.',
        markeredgecolor='black',
        alpha=0.5,
    )

    plt[0].set(
        xlim=(16, 20),
        ylim=(-0.002, 0.005)
    )

    plt[0].set(
        xlabel='psf mag r',
        ylabel='conc',
    )

    predicted = amp*pv
    plt[1].errorbar(
        hd['center'],
        hd['hist'],
        yerr=herr,
    )
    plt[1].curve(hd['center'], predicted)
    xvals = np.linspace(13.9, 21)
    plt[1].curve(
        xvals,
        amp*(xvals - MAGOFF)**INDEX
    )
    plt[1].set(
        xlabel='psf mag r',
        ylabel='Number',
        xlim=(13.5, 21),
    )

    if show:
        plt.show()

    if output is not None:
        output = output.replace('.png', '-fit.png')
        print('writing:', output)
        plt.savefig(output)

    return amp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flist', nargs='+', required=True)
    parser.add_argument('--mag-index', type=int, required=True)
    parser.add_argument('--output')
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    flist = args.flist

    flist.sort()
    t = eu.io.read(flist)
    w, = np.where(t['conc_flags'] == 0)
    print("kept: %d/%d  %g" % (w.size, t.size, w.size/t.size))
    t = t[w]

    amp = get_amp(
        psf_mag=t['psf_mag'][:, args.mag_index],
        conc=t['conc'],
        output=args.output,
        show=args.show,
    )
