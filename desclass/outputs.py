import numpy as np


def get_struct(*, size, nband, injections=False):
    """
    get an output struct

    Parameters
    ----------
    nband: int
        Number of bands

    Returns
    -------
    array with fields
    """

    dt = []

    if not injections:
        dt += [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
        ]

    dt += [
        ('psf_flags', 'i4', (nband, )),
        ('psf_flux', 'f8', (nband, )),
        ('psf_flux_err', 'f8', (nband, )),
        ('psf_mag', 'f8', (nband, )),

        ('flags', 'i4'),

        ('nuse', 'i4'),
        ('flux1', 'f8'),
        ('flux2', 'f8'),
        ('pflux1', 'f8'),
        ('pflux2', 'f8'),
        ('conc', 'f8'),
    ]

    if injections:
        dt += [('true_mag', 'f8', (nband, ))]

    data = np.zeros(size, dtype=dt)

    # these we may not fill in later
    data['psf_flags'] = 1
    data['psf_flux'] = -9999
    data['psf_flux_err'] = 9999.0e9
    data['psf_mag'] = 9999e9

    return data
