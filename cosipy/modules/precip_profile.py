"""Vertical precipitation profile for the 1D band setup (shared by COSIPY.py and LHS_COSIPY.py).

Two parameters, after Immerzeel et al. (2015, HESS 19, 4673): precipitation increases with `lapse` (1/m) up to
`peak_elevation` and decreases with the same rate above it. The factor is normalised by its N_Points-weighted
glacier mean, so the profile only redistributes precipitation with height; the amount stays with the precipitation
factors. lapse = 0 -> no profile (factor 1 everywhere, peak_elevation has no effect).
  peak above the glacier top  -> increase everywhere
  peak inside the glacier     -> maximum at the peak, decrease above
  peak below the tongue       -> decrease everywhere
"""
import numpy as np


def glacier_weights(DATA) -> np.ndarray:
    """N_Points (time mean if dynamic), else MASK, as float with NaN -> 0, shape of HGT."""
    w = DATA["N_Points"].values if "N_Points" in DATA else DATA["MASK"].values
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 3:
        w = np.nanmean(w, axis=0)
    return np.where(np.isfinite(w), w, 0.0)


def precip_profile_factor(hgt, weights, lapse: float, peak_elevation: float, floor: float = 0.01) -> np.ndarray:
    """Normalised precipitation factor per grid cell (same shape as hgt)."""
    hgt = np.asarray(hgt, dtype=np.float64)
    f = np.ones_like(hgt)
    if lapse < 0.0:
        raise ValueError("precip_lapse_rate must be >= 0; a decrease is set by precip_peak_elevation below the "
                         "glacier (decrease everywhere) or inside it (decrease above the peak)")
    if lapse == 0.0:
        return f
    ok = np.isfinite(hgt)
    z_min = np.nanmin(hgt)
    peak_value = 1.0 + lapse * (peak_elevation - z_min)
    raw = np.where(hgt <= peak_elevation, 1.0 + lapse * (hgt - z_min), peak_value - lapse * (hgt - peak_elevation))
    raw = np.where(ok, np.maximum(raw, floor), 1.0)
    w = np.where(ok, weights, 0.0)
    if w.sum() > 0:
        raw = raw / (np.sum(raw * w) / np.sum(w))
    return raw
