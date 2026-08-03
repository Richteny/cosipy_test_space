"""
Test suite comparing createHORAYZONLUT (old) vs createHORAYZONfields (new).

Run with:
    pytest test_horayzon_lut.py -v
or directly:
    python test_horayzon_lut.py

What is tested
--------------
1. sw_dir_cor is bitwise identical — the shadow computation is unchanged.
2. SVF (new field) is physically plausible: values in [0, 1], NaN outside
   glacier mask, mean in expected Alpine range.
3. Slope and aspect agree within a small tolerance (difference between
   slope_vector_meth used by old vs slope_plane_meth used by new).
4. N_Points, HGT, MASK are identical between old and new (1-D case).
5. SVF in the 1-D elevation-band output decreases toward lower elevations
   (physically expected: tongue is more enclosed by valley walls).

File layout expected
--------------------
Two runs, each in 2-D and 1-D mode, producing four NetCDF files:

    OLD_2D_FILE      : old script, 2-D spatial output (the HRZ file merged
                       with sw_dir_cor — contains elevation, slope, aspect,
                       surf_enl_fac, sw_dir_cor, MASK)
    NEW_2D_FILE      : new script, 2-D spatial output (same + svf)
    OLD_1D_FILE      : old script, elevation-profile output
                       (sw_dir_cor, HGT, SLOPE, ASPECT, MASK, N_Points)
    NEW_1D_FILE      : new script, elevation-profile output (same + svf)

Adjust the paths at the top of the CONFIG block below.
"""

import sys
import numpy as np
import xarray as xr
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — set paths to your actual output files
# ═══════════════════════════════════════════════════════════════════════════════
OLD_2D_FILE = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/HEF_HORAYZON-LUT-old_30m.nc"   # old script, 2-D output
NEW_2D_FILE = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/HEF_HORAYZON-LUT-new_30m.nc"   # new script, 2-D output
OLD_1D_FILE = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/HEF_HORAYZON-LUT-old_1D20m.nc"   # old script, 1-D elevation-band output
NEW_1D_FILE = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/HEF_HORAYZON-LUT-new_1D20m.nc"   # new script, 1-D elevation-band output

# Tolerances
SLOPE_TOL_DEG  = 5.0   # max mean absolute difference in slope [°]
                        # expected: small but non-zero (vector vs plane method)
ASPECT_TOL_DEG = 10.0  # max mean absolute difference in aspect [°]
                        # circular distance to handle 0°/360° wrap

SVF_PHYS_MIN   = 0.3   # SVF glacier minimum expected for enclosed tongues
SVF_PHYS_MAX   = 1.0   # SVF upper bound (flat/open terrain)
SVF_MEAN_MIN   = 0.6   # expected mean SVF for a typical Alpine glacier
SVF_MEAN_MAX   = 0.98  # expected mean SVF for a typical Alpine glacier

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def old_2d():
    return xr.open_dataset(OLD_2D_FILE)

@pytest.fixture(scope="module")
def new_2d():
    return xr.open_dataset(NEW_2D_FILE)

@pytest.fixture(scope="module")
def old_1d():
    return xr.open_dataset(OLD_1D_FILE)

@pytest.fixture(scope="module")
def new_1d():
    return xr.open_dataset(NEW_1D_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def circular_mean_abs_diff(a_deg, b_deg):
    """Mean absolute angular difference, wrapping at 360°."""
    diff = np.abs(a_deg - b_deg) % 360.0
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return float(np.nanmean(diff))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. sw_dir_cor — must be IDENTICAL (shadow computation unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def test_sw_dir_cor_2d_identical(old_2d, new_2d):
    """sw_dir_cor (2-D) must be bitwise equal: shadow logic was not changed."""
    np.testing.assert_array_equal(
        old_2d["sw_dir_cor"].values,
        new_2d["sw_dir_cor"].values,
        err_msg="sw_dir_cor differs between old and new in 2-D mode. "
                "The shadow computation should not have changed.",
    )


def test_sw_dir_cor_1d_identical(old_1d, new_1d):
    """sw_dir_cor (1-D elevation bands) must be bitwise equal."""
    np.testing.assert_array_equal(
        old_1d["sw_dir_cor"].values,
        new_1d["sw_dir_cor"].values,
        err_msg="sw_dir_cor differs between old and new in 1-D mode.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SVF — new field, physical validity checks
# ═══════════════════════════════════════════════════════════════════════════════

def test_svf_exists_in_new_output(new_2d, new_1d):
    """SVF field must be present in both new output files."""
    assert "svf" in new_2d, "svf missing from new 2-D output"
    assert "svf" in new_1d, "svf missing from new 1-D output"


def test_svf_range_2d(new_2d):
    """SVF values must lie in [0, 1] for all glacier cells."""
    svf   = new_2d["svf"].values
    valid = ~np.isnan(svf)
    assert valid.any(), "All SVF values are NaN — no glacier cells found."
    np.testing.assert_array_less(
        svf[valid] - 1.0, 1e-6,
        err_msg="SVF > 1 found — physically impossible.",
    )
    np.testing.assert_array_less(
        -svf[valid], 1e-6,
        err_msg="SVF < 0 found — physically impossible.",
    )


def test_svf_mask_consistency_2d(new_2d):
    """SVF must be NaN outside glacier mask and valid inside."""
    svf  = new_2d["svf"].values
    mask = new_2d["MASK"].values

    glacier     = (mask == 1)
    non_glacier = ~glacier & ~np.isnan(mask)  # exclude cells where MASK itself is NaN

    svf_glacier     = svf[glacier]
    svf_non_glacier = svf[non_glacier]

    assert not np.any(np.isnan(svf_glacier)), (
        f"{np.isnan(svf_glacier).sum()} glacier cells have NaN SVF."
    )
    assert np.all(np.isnan(svf_non_glacier)), (
        f"{(~np.isnan(svf_non_glacier)).sum()} non-glacier cells have non-NaN SVF."
    )


def test_svf_plausible_mean_2d(new_2d):
    """Mean SVF across glacier must be in the expected Alpine range."""
    svf  = new_2d["svf"].values
    mask = new_2d["MASK"].values
    mean_svf = float(np.nanmean(svf[mask == 1]))
    assert SVF_MEAN_MIN <= mean_svf <= SVF_MEAN_MAX, (
        f"Mean SVF = {mean_svf:.3f} is outside the expected Alpine range "
        f"[{SVF_MEAN_MIN}, {SVF_MEAN_MAX}]."
    )


def test_svf_range_1d(new_1d):
    """SVF in 1-D output must lie in [0, 1]."""
    svf   = new_1d["svf"].values
    valid = ~np.isnan(svf)
    assert valid.any(), "All 1-D SVF values are NaN."
    assert float(np.nanmin(svf)) >= 0.0 - 1e-6, "SVF < 0 in 1-D output."
    assert float(np.nanmax(svf)) <= 1.0 + 1e-6, "SVF > 1 in 1-D output."


def test_svf_decreases_toward_tongue_1d(new_1d):
    """
    SVF should be lower at the lowest elevation bands (enclosed tongue)
    than at the mid-glacier / accumulation zone.  Test that the mean SVF
    of the bottom 20 % of elevation bands is below the mean of the top 20 %.
    """
    svf  = new_1d["svf"].values.ravel()
    hgt  = new_1d["HGT"].values.ravel()

    valid = ~np.isnan(svf) & ~np.isnan(hgt)
    svf, hgt = svf[valid], hgt[valid]

    n20          = max(1, len(hgt) // 5)
    sort_idx     = np.argsort(hgt)
    svf_bottom   = svf[sort_idx[:n20]].mean()
    svf_top      = svf[sort_idx[-n20:]].mean()

    assert svf_bottom < svf_top, (
        f"SVF at the tongue ({svf_bottom:.3f}) is not lower than at the "
        f"upper glacier ({svf_top:.3f}). "
        "Expected lower SVF at enclosed lower elevations."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Slope and aspect — close but not identical (vector vs plane method)
# ═══════════════════════════════════════════════════════════════════════════════

def test_slope_close_2d(old_2d, new_2d):
    """
    Slope from new (slope_plane_meth) should agree with old (slope_vector_meth)
    within SLOPE_TOL_DEG on average.
    """
    s_old = old_2d["slope"].values
    s_new = new_2d["slope"].values
    valid = ~np.isnan(s_old) & ~np.isnan(s_new)

    mad = float(np.nanmean(np.abs(s_old[valid] - s_new[valid])))
    assert mad <= SLOPE_TOL_DEG, (
        f"Mean |Δslope| = {mad:.2f}° exceeds tolerance {SLOPE_TOL_DEG}°. "
        "slope_plane_meth and slope_vector_meth are producing very different results."
    )
    print(f"  Slope MAD between methods: {mad:.3f}°")


def test_aspect_close_2d(old_2d, new_2d):
    """
    Aspect from new should agree with old within ASPECT_TOL_DEG on average
    (circular distance to handle 0°/360° wrap).
    """
    a_old = old_2d["aspect"].values
    a_new = new_2d["aspect"].values

    mad = circular_mean_abs_diff(a_old, a_new)
    assert mad <= ASPECT_TOL_DEG, (
        f"Mean circular |Δaspect| = {mad:.2f}° exceeds tolerance "
        f"{ASPECT_TOL_DEG}°."
    )
    print(f"  Aspect MAD between methods: {mad:.3f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Static fields — must be identical between old and new (1-D case)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("var", ["HGT", "MASK", "N_Points"])
def test_static_1d_fields_identical(old_1d, new_1d, var):
    """HGT, MASK, N_Points must be bitwise identical: derived from same DEM."""
    np.testing.assert_array_equal(
        old_1d[var].values,
        new_1d[var].values,
        err_msg=f"{var} differs between old and new 1-D output.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. New output contains all expected variables
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("var", ["sw_dir_cor", "MASK", "elevation",
                                  "slope", "aspect", "surf_enl_fac", "svf"])
def test_new_2d_has_required_variables(new_2d, var):
    assert var in new_2d, f"Variable '{var}' missing from new 2-D output."


@pytest.mark.parametrize("var", ["sw_dir_cor", "HGT", "SLOPE", "ASPECT",
                                  "MASK", "N_Points", "svf"])
def test_new_1d_has_required_variables(new_1d, var):
    assert var in new_1d, f"Variable '{var}' missing from new 1-D output."


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER (without pytest)
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all tests manually without pytest, printing a summary."""
    old_2d = xr.open_dataset(OLD_2D_FILE)
    new_2d = xr.open_dataset(NEW_2D_FILE)
    old_1d = xr.open_dataset(OLD_1D_FILE)
    new_1d = xr.open_dataset(NEW_1D_FILE)

    tests = [
        ("sw_dir_cor 2D identical",       lambda: test_sw_dir_cor_2d_identical(old_2d, new_2d)),
        ("sw_dir_cor 1D identical",       lambda: test_sw_dir_cor_1d_identical(old_1d, new_1d)),
        ("SVF exists in new output",      lambda: test_svf_exists_in_new_output(new_2d, new_1d)),
        ("SVF range [0,1] 2D",            lambda: test_svf_range_2d(new_2d)),
        ("SVF mask consistency 2D",       lambda: test_svf_mask_consistency_2d(new_2d)),
        ("SVF plausible mean 2D",         lambda: test_svf_plausible_mean_2d(new_2d)),
        ("SVF range [0,1] 1D",            lambda: test_svf_range_1d(new_1d)),
        ("SVF decreases toward tongue 1D",lambda: test_svf_decreases_toward_tongue_1d(new_1d)),
        ("Slope close 2D",                lambda: test_slope_close_2d(old_2d, new_2d)),
        ("Aspect close 2D",               lambda: test_aspect_close_2d(old_2d, new_2d)),
        ("HGT identical 1D",              lambda: test_static_1d_fields_identical(old_1d, new_1d, "HGT")),
        ("MASK identical 1D",             lambda: test_static_1d_fields_identical(old_1d, new_1d, "MASK")),
        ("N_Points identical 1D",         lambda: test_static_1d_fields_identical(old_1d, new_1d, "N_Points")),
    ]

    # Also check required variables in new outputs
    for var in ["sw_dir_cor", "MASK", "elevation", "slope", "aspect",
                "surf_enl_fac", "svf"]:
        tests.append((f"new 2D has '{var}'",
                       lambda v=var: test_new_2d_has_required_variables(new_2d, v)))
    for var in ["sw_dir_cor", "HGT", "SLOPE", "ASPECT", "MASK", "N_Points", "svf"]:
        tests.append((f"new 1D has '{var}'",
                       lambda v=var: test_new_1d_has_required_variables(new_1d, v)))

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    old_2d.close(); new_2d.close(); old_1d.close(); new_1d.close()
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
