import ngmix
from numba import njit


DESY5_BADPIX_MAP = {
    "BPM":          1,  # /* set in bpm (hot/dead pixel/column)        */
    "SATURATE":     2,  # /* saturated pixel                           */
    "INTERP":       4,  # /* interpolated pixel                        */
    "BADAMP":       8,  # /* Data from non-functional amplifier        */
    "CRAY":        16,  # /* cosmic ray pixel                          */
    "STAR":        32,  # /* bright star pixel                         */
    "TRAIL":       64,  # /* bleed trail pixel                         */
    "EDGEBLEED":  128,  # /* edge bleed pixel                          */
    "SSXTALK":    256,  # /* pixel potentially effected by xtalk from  */
                        # /*       a super-saturated source            */
    "EDGE":       512,  # /* pixel flag to exclude CCD glowing edges   */
    "STREAK":    1024,  # /* pixel associated with streak from a       */
                        # /*       satellite, meteor, ufo...           */
    "SUSPECT":   2048,  # /* nominally useful pixel but not perfect    */
    "FIXED":     4096,  # /* corrected by pixcorrect                   */
    "NEAREDGE":  8192,  # /* suspect due to edge proximity             */
    "TAPEBUMP": 16384,  # /* suspect due to known tape bump            */
}

IMAGE_FLAGNAMES_TO_MASK = [
    "BPM",
    "SATURATE",
    "INTERP",
    "BADAMP",
    "CRAY",
    "TRAIL",
    "EDGEBLEED",
    "EDGE",
    "STREAK",
    "NEAREDGE",
    "TAPEBUMP",
]


def get_flagvals(flagnames):
    """
    get logical or of all flags in the list of flag names
    """
    flagvals = 0
    for flagname in flagnames:
        flagvals |= get_flagval(flagname)

    return flagvals


def get_flagval(flagname):
    """
    get the value of the flag
    """
    if flagname not in DESY5_BADPIX_MAP:
        raise ValueError("bad flag name: '%s'" % flagname)

    return DESY5_BADPIX_MAP[flagname]


FLAGVALS_TO_MASK = get_flagvals(IMAGE_FLAGNAMES_TO_MASK)


@njit
def zero_masked_weight(bmask, weight, flagval):
    """
    zero the weight image for pixels that have the bits
    in flagval set

    Parameters
    ----------
    bmask: array
        bitmask image
    weight: array
        weight image
    flagval: int
        bitmask to check in mask
    """
    nrow, ncol = bmask.shape

    for row in range(nrow):
        for col in range(ncol):
            bmask_val = bmask[row, col]
            if (bmask_val & flagval) != 0:
                weight[row, col] = 0


def zero_masked_weights(*, obslist):
    """
    zero the weight image for pixels that have bad bits set, return
    a new ObsList

    Parameters
    ----------
    obslist: ngmix.ObsList
        Observations to check

    Returns
    -------
    new_obslist: ngmix.ObsList
        New version of obslist with weights set to zero.  Those with
        no non-zero weight pixels are removed
    """
    new_obslist = ngmix.ObsList()

    for obs in obslist:
        try:
            with obs.writeable():
                zero_masked_weight(
                    obs.bmask,
                    obs.weight,
                    FLAGVALS_TO_MASK,
                )
            new_obslist.append(obs)
        except ngmix.GMixFatalError:
            pass

    return new_obslist
