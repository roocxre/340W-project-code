"""Robust local mean decomposition (R-LMD).

Port of lmd_public.m by Zhiliang Liu, Deng Pan and Yaqiang Jin (MATLAB File
Exchange 66935, copied under reference/rlmd/), which implements Liu et al.,
"Time-frequency representation based on robust local mean decomposition for
multi-component AM-FM signal analysis", MSSP 95:468-487, 2017. Default options
only: moving-average smoothing, 'liu' span selection, 'liu' sifting stop,
mirror extension ratio 0.2.

Index vectors are kept 1-based, as in the MATLAB source, so the boundary
arithmetic of extend() can be compared line by line.
"""
import numpy as np

STOP_ENERGY_RATIO = 0.001


class ExtensionError(RuntimeError):
    """The mirror extension hit the error('bug') branch of the MATLAB code."""


def extr(x):
    """1-based indices of the local minima and maxima of x."""
    d = np.diff(x)
    d1, d2 = d[:-1], d[1:]
    prod = d1 * d2
    indmin = np.flatnonzero((prod < 0) & (d1 < 0)) + 2
    indmax = np.flatnonzero((prod < 0) & (d1 > 0)) + 2
    return indmin, indmax


def smooth(y, span):
    """MATLAB smooth(y, span): moving average with shrinking end windows."""
    n = len(y)
    span = min(int(span), n)
    width = span - 1 + span % 2
    if width == 1:
        return y.copy()
    mid = np.convolve(y, np.full(width, 1.0 / width), mode="valid")
    odd = np.arange(1, width - 1, 2)
    cbegin = np.cumsum(y[:width - 2])[::2] / odd
    cend = (np.cumsum(y[:n - width + 1:-1])[::2] / odd)[::-1]
    return np.concatenate([cbegin, mid, cend])


def _auto_bin_edges(v):
    """Bin edges of MATLAB histcounts(v) with the default 'auto' method.

    Re-implemented from the documented behaviour: unit-width integer bins for
    integer data with range <= 50, otherwise Scott's rule snapped to a "nice"
    width.
    """
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    if np.all(v == np.round(v)) and rng <= 50:
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.0, 1.0)
    raw = 3.5 * np.std(v, ddof=1) / len(v) ** (1 / 3)
    scale = max(abs(lo), abs(hi))
    raw = max(raw, np.spacing(scale))
    if rng <= max(np.sqrt(np.spacing(scale)), np.finfo(float).tiny):
        return np.array([lo - 0.5, lo + 0.5])
    p10 = 10.0 ** np.floor(np.log10(raw))
    rel = raw / p10
    if rel < 1.5:
        width = p10
    elif rel < 2.5:
        width = 2 * p10
    elif rel < 4:
        width = 3 * p10
    elif rel < 7.5:
        width = 5 * p10
    else:
        width = 10 * p10
    left = min(width * np.floor(lo / width), lo)
    nbins = max(1, int(np.ceil((hi - left) / width)))
    right = max(left + nbins * width, hi)
    edges = left + width * np.arange(nbins + 1)
    edges[-1] = right
    return edges


def best_span(ind_extr, x):
    """Moving-average span ('liu' rule) and the largest local-mean step."""
    x_extr = x[ind_extr - 1]
    eql = x_extr[2:] - x_extr[:-2]
    ind_extr = np.delete(ind_extr, np.flatnonzero(eql == 0) + 1)
    step = (ind_extr[1:] - ind_extr[:-1] + 1).astype(float)
    smax = step.max()
    edges = _auto_bin_edges(step)
    counts, _ = np.histogram(step, bins=edges)
    density = counts / counts.sum()
    centers = edges[:-1] + np.diff(edges) / 2
    span_c = np.sum(centers * density)
    span_std = np.sqrt(np.sum((centers - span_c) ** 2 * density))
    span = int(np.ceil(span_c + 3 * span_std))
    span = span + 1 - span % 2
    return span, smax


def itrma(x, span, smax):
    """Iterated moving average ('fixed' mode): smooth until no flat step is left."""
    x = smooth(x, span)
    nm = len(x)
    cntr = 0
    max_c = int(np.ceil(smax / span)) * 15
    k = (span + 1) // 2          # 1-based
    kmax = nm - (span - 1) // 2
    while k < kmax and cntr < max_c:
        # first 1-based index i in [k, kmax) with x(i) == x(i+1)
        flat = np.flatnonzero(x[k - 1:kmax - 1] == x[k:kmax])
        if len(flat) == 0:
            break
        k += int(flat[0])
        x = smooth(x, span)
        cntr += 1
    return x


def _sl(a, lo, hi):
    """MATLAB a(lo:hi) with 1-based inclusive bounds (empty if lo > hi)."""
    return a[lo - 1:hi] if hi >= lo else a[:0]


def extend(x, indmin, indmax, extd_r):
    """Mirror extension of x at both ends (after Rilling & Flandrin's emd)."""
    xlen = len(x)
    if extd_r == 0:
        return indmin, indmax, x, (1, xlen)
    nbsym = int(np.ceil(extd_r * len(indmax)))
    nmin, nmax = len(indmin), len(indmax)
    one = np.array([1])
    last = np.array([xlen])
    X = lambda i: x[i - 1]  # noqa: E731

    if indmax[0] < indmin[0]:
        if X(1) > X(indmin[0]):
            lmax = _sl(indmax, 2, min(nmax, nbsym + 1))[::-1]
            lmin = _sl(indmin, 1, min(nmin, nbsym))[::-1]
            lsym = indmax[0]
        else:
            lmax = _sl(indmax, 1, min(nmax, nbsym))[::-1]
            lmin = np.concatenate([_sl(indmin, 1, min(nmin, nbsym - 1))[::-1], one])
            lsym = 1
    else:
        if X(1) < X(indmax[0]):
            lmax = _sl(indmax, 1, min(nmax, nbsym))[::-1]
            lmin = _sl(indmin, 2, min(nmin, nbsym + 1))[::-1]
            lsym = indmin[0]
        else:
            lmax = np.concatenate([_sl(indmax, 1, min(nmax, nbsym - 1))[::-1], one])
            lmin = _sl(indmin, 1, min(nmin, nbsym))[::-1]
            lsym = 1

    if indmax[-1] < indmin[-1]:
        if X(xlen) < X(indmax[-1]):
            rmax = _sl(indmax, max(nmax - nbsym + 1, 1), nmax)[::-1]
            rmin = _sl(indmin, max(nmin - nbsym, 1), nmin - 1)[::-1]
            rsym = indmin[-1]
        else:
            rmax = np.concatenate([last, _sl(indmax, max(nmax - nbsym + 2, 1), nmax)[::-1]])
            rmin = _sl(indmin, max(nmin - nbsym + 1, 1), nmin)[::-1]
            rsym = xlen
    else:
        if X(xlen) > X(indmin[-1]):
            rmax = _sl(indmax, max(nmax - nbsym, 1), nmax - 1)[::-1]
            rmin = _sl(indmin, max(nmin - nbsym + 1, 1), nmin)[::-1]
            rsym = indmax[-1]
        else:
            rmax = _sl(indmax, max(nmax - nbsym + 1, 1), nmax)[::-1]
            rmin = np.concatenate([last, _sl(indmin, max(nmin - nbsym + 2, 1), nmin)[::-1]])
            rsym = xlen

    if min(len(lmin), len(lmax), len(rmin), len(rmax)) == 0:
        raise ExtensionError("empty mirror set")

    tlmin, tlmax = 2 * lsym - lmin, 2 * lsym - lmax
    trmin, trmax = 2 * rsym - rmin, 2 * rsym - rmax

    # in case symmetrized parts do not extend enough
    if tlmin[0] > 1 or tlmax[0] > 1:
        if lsym == indmax[0]:
            lmax = _sl(indmax, 1, min(nmax, nbsym))[::-1]
        else:
            lmin = _sl(indmin, 1, min(nmin, nbsym))[::-1]
        if lsym == 1:
            raise ExtensionError("left")
        lsym = 1
    if trmin[-1] < xlen or trmax[-1] < xlen:
        if rsym == indmax[-1]:
            rmax = _sl(indmax, max(nmax - nbsym + 1, 1), nmax)[::-1]
        else:
            rmin = _sl(indmin, max(nmin - nbsym + 1, 1), nmin)[::-1]
        if rsym == xlen:
            raise ExtensionError("right")
        rsym = xlen

    l_end = max(lmax.max(), lmin.max())
    r_end = min(rmax.min(), rmin.min())

    new_lmax = l_end + 1 - lmax
    new_lmin = l_end + 1 - lmin
    new_rmax = rsym - rmax
    new_rmin = rsym - rmin
    lx_length = l_end - lsym
    lx = _sl(x, lsym + 1, l_end)[::-1]
    rx = _sl(x, r_end, rsym - 1)[::-1]

    ext_x = np.concatenate([lx, _sl(x, lsym, rsym), rx])
    shift = lx_length - lsym + 1
    ext_indmin = np.concatenate([new_lmin, indmin + shift, new_rmin + shift + rsym])
    ext_indmax = np.concatenate([new_lmax, indmax + shift, new_rmax + shift + rsym])
    cut_index = (lx_length - lsym + 2, xlen + lx_length - lsym + 1)
    return ext_indmin, ext_indmax, ext_x, cut_index


def mean_amp(x, extd_r=0.2):
    """Smoothed local mean and local amplitude of x, and its extrema count."""
    indmin, indmax = extr(x)
    n_extr = len(indmin) + len(indmax)
    if n_extr < 3:
        return None, None, n_extr
    try:
        ext_indmin, ext_indmax, ext_x, cut = extend(x, indmin, indmax, extd_r)
    except ExtensionError:
        # The MATLAB code aborts here; fall back to no extension instead.
        ext_indmin, ext_indmax, ext_x, cut = extend(x, indmin, indmax, 0)
    ind = np.sort(np.concatenate([ext_indmin, ext_indmax]))
    ind = ind[(ind >= 1) & (ind <= len(ext_x))]
    v = ext_x[ind - 1]
    mean_k = 0.5 * (v[:-1] + v[1:])
    amp_k = 0.5 * np.abs(v[:-1] - v[1:])
    # Piecewise-constant fill: segment k covers [ind(k), ind(k+1)], and each
    # later segment overwrites the shared end point.
    m0 = np.zeros(len(ext_x))
    a0 = np.zeros(len(ext_x))
    reps = np.diff(ind)
    lo, hi = ind[0] - 1, ind[-1] - 1
    m0[lo:hi] = np.repeat(mean_k, reps)
    a0[lo:hi] = np.repeat(amp_k, reps)
    m0[hi], a0[hi] = mean_k[-1], amp_k[-1]
    span, smax = best_span(ind, ext_x)
    m = itrma(m0, span, smax)[cut[0] - 1:cut[1]]
    a = itrma(a0, span, smax)[cut[0] - 1:cut[1]]
    return m, a, n_extr


def _stop_lmd(x, x_energy):
    indmin, indmax = extr(x)
    return (len(indmin) + len(indmax)) < 3 or np.sum(x ** 2) / x_energy < STOP_ENERGY_RATIO


def _objective(a_j):
    """Sifting objective of Liu et al.: RMS + |excess kurtosis| of a_j - 1."""
    df = a_j - 1.0
    c = df - df.mean()
    var = np.mean(c ** 2)
    kurt = np.mean(c ** 4) / var ** 2 if var > 0 else np.nan
    return np.sqrt(np.mean(df ** 2)) + abs(kurt - 3)


def rlmd(x, max_pfs=10, max_iter=30, extd_r=0.2):
    """Decompose x into product functions.

    Returns (pfs, ams, fms): pfs has one row per product function followed by
    the residual; ams and fms are the amplitude and frequency modulation
    signals of each product function.
    """
    x = np.asarray(x, dtype=float).ravel()
    nx = len(x)
    x_energy = np.sum(x ** 2)
    pfs, ams, fms = [], [], []
    xs = x.copy()
    while len(pfs) < max_pfs and x_energy > 0 and not _stop_lmd(xs, x_energy):
        a_i = np.ones(nx)
        s = xs
        # Rows stay zero for an iteration that breaks early, as in the MATLAB
        # code, where the zero objective value then wins the argmin.
        s_j = np.zeros((max_iter, nx))
        a_ij = np.zeros((max_iter, nx))
        fv = np.zeros(max_iter)
        j = 0
        while j < max_iter:
            j += 1
            m_j, a_j, n_extr = mean_amp(s, extd_r)
            if n_extr < 3:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (s - m_j) / a_j
            a_i = a_i * a_j
            a_ij[j - 1] = a_i
            s_j[j - 1] = s
            fv[j - 1] = _objective(a_j)
            if not np.all(np.isfinite(s)):
                # a_j touched zero; MATLAB would carry NaN/Inf on. Stop here
                # and exclude this iteration.
                fv[j - 1] = np.inf
                break
            if j >= 3 and fv[j - 1] >= fv[j - 2] >= fv[j - 3]:
                break
        fvj = np.where(np.isnan(fv[:j]), np.inf, fv[:j])
        opt = int(np.argmin(fvj))
        am, fm = a_ij[opt], s_j[opt]
        if not np.all(np.isfinite(fm)):
            break
        pf = am * fm
        ams.append(am)
        fms.append(fm)
        pfs.append(pf)
        xs = xs - pf
    pfs.append(xs)
    return np.array(pfs), np.array(ams).reshape(-1, nx), np.array(fms).reshape(-1, nx)


def modes(x, n_modes=5, **kw):
    """The n_modes-level decomposition used in the paper: n_modes - 1 product
    functions plus the residual, zero-padded to (n_modes, len(x)) when the
    signal runs out of extrema earlier."""
    pfs, _, _ = rlmd(x, max_pfs=n_modes - 1, **kw)
    out = np.zeros((n_modes, len(pfs[0])))
    out[:len(pfs)] = pfs
    return out
