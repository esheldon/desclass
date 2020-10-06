import numpy as np
import ngmix
from .constants import RADIUS, MAGZP


def replace_ext(*, fname, old_ext, new_ext):
    """
    replace the extension in the filename, ensuring that the result is a new
    string

    Parameters
    ----------
    fname: string
        File name
    old_ext: string
        Old string to replace
    new_ext: string
        New string for extension

    Returns
    -------
    new_fname: string
        Name with new extension
    """

    if old_ext == new_ext:
        raise ValueError("new and old extensions are the same: '%s'" % old_ext)

    el = len(old_ext)

    if fname[-el:] != old_ext:
        raise ValueError("extension '%s' not in filename" % old_ext)

    new_fname = fname[:-el] + new_ext
    assert new_fname != fname

    return new_fname


def rewrap_obslist(obslist_in):
    """
    rewrap an ObsList.  When the obs list is sliced, it is converted to a
    regular list (python behavior).  This rewraps to ObsList

    Parameters
    ----------
    observation list: [Observation]
        List of Observations

    Returns
    -------
    obslist: ngmix.ObsList
        converted list of Obs
    """
    obslist = ngmix.ObsList()

    for obs in obslist_in:
        obslist.append(obs)

    return obslist


def get_mag(flux):
    """
    convert flux to mag assuming zero point of 30

    Parameters
    ----------
    flux: float
        Flux value

    Returns
    -------
    mag: float
        Magnitude at zero point 30
    """
    return MAGZP - 2.5*np.log10(flux.clip(min=0.001))


def trim_obslist(*, obslist):
    """
    trim the images to the default radius, returning a new MultiBandObsList

    Parameters
    ----------
    obslist: ngmix.ObsList
        Observations to trim

    Returns
    ----------
    new_obslist: ngmix.ObsList
        Trimmed observations
    """

    new_obslist = ngmix.ObsList()

    for obs in obslist:
        try:
            new_obs = trim_obs(obs=obs)
            new_obslist.append(new_obs)
        except ngmix.GMixFatalError:
            pass

    return new_obslist


def trim_obs(*, obs):
    """
    trim the images to the default radius, returning a new Observation

    Parameters
    ----------
    obs: ngmix.Observation
        Observation to trim

    Returns
    ----------
    new_obs: ngmix.Observation
        Trimmed Observation
    """
    j = obs.jacobian
    row, col = j.cen
    irow, icol = int(row), int(col)

    row_start = irow - RADIUS
    row_end = irow + RADIUS + 1
    col_start = icol - RADIUS
    col_end = icol + RADIUS + 1

    new_image = obs.image[
        row_start:row_end,
        col_start:col_end,
    ]
    new_weight = obs.weight[
        row_start:row_end,
        col_start:col_end,
    ]
    if obs.has_bmask():
        new_bmask = obs.bmask[
            row_start:row_end,
            col_start:col_end,
        ]
    else:
        new_bmask = None

    new_rowcen = RADIUS + row-irow
    new_colcen = RADIUS + col-icol

    new_jac = j.copy()
    new_jac.set_cen(row=new_rowcen, col=new_colcen)

    if obs.has_psf():
        new_psf = trim_obs(obs=obs.psf)
    else:
        new_psf = None

    return ngmix.Observation(
        image=new_image,
        weight=new_weight,
        bmask=new_bmask,
        jacobian=new_jac,
        psf=new_psf,
        meta=obs.meta,
    )
