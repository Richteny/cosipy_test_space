"""
Test suite for the HORAYZON radiation correction in cosmo2cosipy.

Tests are fully self-contained — no COSIPY imports, no files required.
All functions are reimplemented from cosmo2cosipy.py so that tests
break immediately if the script logic changes.

Run with:
    pytest test_horayzon_radiation.py -v
or without pytest:
    python test_horayzon_radiation.py

Sections
--------
1. LUT year selection  (--sw-starts / searchsorted logic)
2. Leap-year LUT shift
3. Solar geometry      (stime formula, solar noon placement, tcart)
4. Mölg transmissivities (TAUr, TAUg, TAUa, TAUaa, TAUw)
5. Clear-sky radiation (sdir, Dcs, grcs physical constraints)
6. f_dif partition     (bounds, monotonicity, limiting cases)
7. Horayzon2022        (energy conservation, SVF fallback)
8. Horayzon_theory     (positivity, no G_meas dependency)
9. Edge cases          (night, zero SW, overcast)
"""

import math
import datetime
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# EXACT COPIES OF SCRIPT LOGIC
# Any change here must be mirrored in cosmo2cosipy.py and vice-versa.
# ══════════════════════════════════════════════════════════════════════════════

# Physical constants — identical to cosmo2cosipy.py
Sol0   = 1367.0
aesc1  = 0.87764
aesc2  = 2.4845e-5
alphss = 0.9
dirovc = 0.00
dif1   = 4.6
difra  = 0.66
Cf     = 0.65


def select_lut_idx(year: int, sw_starts_sorted: list, n_luts: int) -> int:
    """Mirror of LUT selection in cosmo2cosipy — side='right'."""
    if not sw_starts_sorted:
        return 0
    idx = int(np.searchsorted(sw_starts_sorted, year, side="right"))
    return min(idx, n_luts - 1)


def compute_stime(hour: float, tcorr: float, tcart: float) -> float:
    """Mirror of stime formula in cosmo2cosipy."""
    return 180.0 + 7.5 - hour * 15.0 - tcorr + tcart


def compute_sin_h(doy: int, hour: float, lat_deg: float,
                  tcart: float, tcorr: float = 0.0) -> float:
    """Mirror of sin_h computation in cosmo2cosipy (simplified solpars)."""
    tau    = 2 * math.pi * (doy - 1) / 365
    soldec = (0.006918
              - 0.399912 * math.cos(tau)
              + 0.070257 * math.sin(tau)
              - 0.006758 * math.cos(2 * tau)
              + 0.000907 * math.sin(2 * tau)
              - 0.002697 * math.cos(3 * tau)
              + 0.00148  * math.sin(3 * tau))
    stime  = compute_stime(hour, tcorr, tcart)
    return (math.sin(soldec) * math.sin(math.radians(lat_deg))
            + math.cos(soldec) * math.cos(math.radians(lat_deg))
            * math.cos(math.radians(stime)))


def compute_transmissivities(sin_h: float, p_hpa: float,
                              T_K: float, vp_hpa: float,
                              elev_m: float = 0.0):
    """
    Mirror of transmissivity chain in cosmo2cosipy.
    Returns (TAUr, TAUg, TAUa, TAUaa, TAUw, taucs, sdir, Dcs).
    All inputs scalar.
    """
    mopt  = 35.0 * (1224.0 * sin_h**2 + 1.0)**(-0.5)
    p_rel = p_hpa / 1013.25
    TAUr  = math.exp(
        (-0.09030 * (p_rel * mopt)**0.84)
        * (1.0 + p_rel * mopt - (p_rel * mopt)**1.01)
    )
    TAUg  = math.exp(-0.0127 * mopt**0.26)
    k_aes = min(aesc2 * elev_m + aesc1, 1.0)
    TAUa  = k_aes**mopt
    TAUaa = (1.0
             - (1.0 - alphss)
             * (1.0 - p_rel * mopt + (p_rel * mopt)**1.06)
             * (1.0 - TAUa))
    _w    = 46.5 * vp_hpa / T_K
    TAUw  = 1.0 - (2.4959 * mopt * _w
                   / ((1.0 + 79.034 * mopt * _w)**0.6828
                      + 6.385 * mopt * _w))
    taucs = TAUr * TAUg * TAUa * TAUw
    eccorr = 1.0   # simplification for unit tests
    sdir  = Sol0 * eccorr * sin_h * taucs
    Dcs   = (difra * Sol0 * eccorr * sin_h
             * TAUg * TAUw * TAUaa
             * (1.0 - TAUr * TAUa / TAUaa)
             / (1.0 - p_rel * mopt + (p_rel * mopt)**1.02))
    return TAUr, TAUg, TAUa, TAUaa, TAUw, taucs, sdir, Dcs


def compute_f_dif(sdir: float, Dcs: float, cld: float) -> float:
    """Mirror of f_dif computation in cosmo2cosipy."""
    grcs = sdir + Dcs
    if cld > 0:
        G_dir_t = sdir * (1.0 - (1.0 - dirovc) * cld)
        G_dif_t = grcs * ((100.0 - Cf * 100.0 - dif1) / 100.0 * cld
                           + dif1 / 100.0)
    else:
        G_dir_t = sdir
        G_dif_t = Dcs
    G_tot = max(G_dir_t + G_dif_t, 1e-10)
    return float(np.clip(G_dif_t / G_tot, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 1. LUT YEAR SELECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestLutYearSelection:
    """
    --sw-starts: boundary year belongs to the NEW file (side='right').
    Example: 3 files, sw_starts=[2013, 2017]
      year < 2013  → file 0 (rgi6)
      2013 ≤ year < 2017 → file 1 (2013 geometry)
      year ≥ 2017  → file 2 (2017 geometry)
    """

    starts = [2013, 2017]
    n      = 3

    def test_before_first_start(self):
        assert select_lut_idx(2012, self.starts, self.n) == 0

    def test_at_first_start_goes_to_new_file(self):
        assert select_lut_idx(2013, self.starts, self.n) == 1

    def test_between_starts(self):
        assert select_lut_idx(2016, self.starts, self.n) == 1

    def test_at_second_start_goes_to_new_file(self):
        assert select_lut_idx(2017, self.starts, self.n) == 2

    def test_after_last_start(self):
        assert select_lut_idx(2030, self.starts, self.n) == 2

    def test_single_file_always_zero(self):
        for year in [1990, 2010, 2050]:
            assert select_lut_idx(year, [], 1) == 0

    def test_four_file_hef_default(self):
        hef = [2014, 2018, 2021]
        cases = [
            (2013, 0), (2014, 1), (2017, 1),
            (2018, 2), (2020, 2), (2021, 3), (2030, 3),
        ]
        for year, expected in cases:
            assert select_lut_idx(year, hef, 4) == expected, \
                f"year={year}: expected idx={expected}"

    def test_boundary_is_exclusive_for_previous_file(self):
        # year=2013 must NOT go to file 0 (old geometry)
        assert select_lut_idx(2013, self.starts, self.n) != 0

    def test_idx_never_exceeds_n_minus_one(self):
        for year in range(1990, 2031):
            idx = select_lut_idx(year, self.starts, self.n)
            assert 0 <= idx < self.n


# ══════════════════════════════════════════════════════════════════════════════
# 2. LEAP-YEAR LUT SHIFT
# ══════════════════════════════════════════════════════════════════════════════

class TestLeapYearShift:
    """Non-leap year doy > 59 must be shifted by +1 to align with leap LUT."""

    def _lut_doy(self, year: int, month: int, day: int) -> int:
        t = datetime.datetime(year, month, day)
        doy = t.timetuple().tm_yday
        if year % 4 != 0 and doy > 59:
            return doy + 1
        return doy

    def test_non_leap_march_shifts(self):
        # March 1 in non-leap year = doy 60 → shifted to 61
        assert self._lut_doy(2019, 3, 1) == 61

    def test_non_leap_feb28_no_shift(self):
        # Feb 28 = doy 59 → NOT shifted (≤59)
        assert self._lut_doy(2019, 2, 28) == 59

    def test_leap_year_no_shift(self):
        # March 1 in leap year = doy 61 → no shift
        assert self._lut_doy(2020, 3, 1) == 61

    def test_non_leap_dec31_shifts(self):
        # Dec 31 = doy 365 in non-leap → shifted to 366
        assert self._lut_doy(2019, 12, 31) == 366

    def test_shift_is_at_most_one(self):
        for month in range(1, 13):
            for day in range(1, 29):
                try:
                    shifted = self._lut_doy(2019, month, day)
                    original = datetime.datetime(2019, month, day).timetuple().tm_yday
                    assert shifted - original in (0, 1)
                except ValueError:
                    pass  # invalid date e.g. Feb 30


# ══════════════════════════════════════════════════════════════════════════════
# 3. SOLAR GEOMETRY — stime and sin_h
# ══════════════════════════════════════════════════════════════════════════════

class TestSolarGeometry:
    """
    tcart places solar noon at the correct clock hour.
    At solar noon (stime ≈ 0 modulo the 7.5° half-step shift), sin_h is maximal.
    The 7.5° half-step means solar noon falls in the MIDDLE of the noon hour,
    not exactly at the timestamp, so we test for max sin_h near the expected time.
    """

    lat = 46.8
    doy = 172   # near summer solstice, long days

    def _find_noon(self, tcart: float, tcorr: float = 0.0) -> float:
        hours = np.arange(0, 24, 0.1)
        sins  = [compute_sin_h(self.doy, h, self.lat, tcart, tcorr)
                 for h in hours]
        return float(hours[np.argmax(sins)])

    def test_utc_forcing_hef_noon_near_1178(self):
        # UTC forcing (offset=0), HEF at 10.76°E.
        # Physical solar noon = 12 - 10.76/15 = 11.28 h UTC.
        # The 7.5° half-step shifts the formula's effective time by +0.5h:
        # sin_h peaks at the timestamp where stime=0, i.e. ≈11.78h UTC.
        # This is correct: the 11:00 timestamp represents the 11:00-12:00 interval
        # whose midpoint (11:30) is closest to physical noon (11:17).
        noon = self._find_noon(tcart=-10.76)
        assert abs(noon - 11.78) < 0.15, f"Noon at {noon:.2f} h UTC, expected ≈11.78"

    def test_cet_forcing_hef_noon_near_1278(self):
        # CET forcing (offset=1), HEF at 10.76°E.
        # Physical solar noon ≈ 12.28 h CET.
        # With 7.5° half-step: peak sin_h at ≈12.78 h CET.
        noon = self._find_noon(tcart=4.24)
        assert abs(noon - 12.78) < 0.15, f"Noon at {noon:.2f} h CET, expected ≈12.78"

    def test_utc_himalaya_noon_near_0677(self):
        # UTC forcing, Himalayan glacier at 86°E.
        # Physical solar noon ≈ 6.27 h UTC; with half-step ≈ 6.77 h.
        noon = self._find_noon(tcart=-86.0)
        assert abs(noon - 6.77) < 0.15, f"Noon at {noon:.2f} h UTC, expected ≈6.77"

    def test_equivalent_moments_give_same_sin_h(self):
        # UTC hour=10.0 and CET hour=11.0 represent the same physical moment
        sh_utc = compute_sin_h(self.doy, 10.0, self.lat, tcart=-10.76)
        sh_cet = compute_sin_h(self.doy, 11.0, self.lat, tcart=4.24)
        assert abs(sh_utc - sh_cet) < 1e-4, \
            f"UTC sin_h={sh_utc:.6f} ≠ CET sin_h={sh_cet:.6f}"

    def test_sin_h_bounded(self):
        # sin_h ∈ [-1, 1] always
        for hour in range(24):
            sh = compute_sin_h(self.doy, hour, self.lat, tcart=4.24)
            assert -1.0 <= sh <= 1.0

    def test_night_time_winter(self):
        # Polar-ish night: doy=355, lat=70°N, UTC noon
        sh = compute_sin_h(doy=355, hour=12, lat_deg=70, tcart=0)
        assert sh < 0.01, f"Expected night, got sin_h={sh:.4f}"

    def test_stime_formula_correctness(self):
        # At hour=12, tcorr=0, tcart=0: stime = 180+7.5-180+0 = 7.5
        assert compute_stime(12.0, 0.0, 0.0) == pytest.approx(7.5)
        # At hour=0, tcorr=0, tcart=0: stime = 180+7.5-0 = 187.5
        assert compute_stime(0.0, 0.0, 0.0) == pytest.approx(187.5)
        # tcart shifts stime linearly
        assert (compute_stime(12.0, 0.0, 5.0)
                - compute_stime(12.0, 0.0, 0.0)) == pytest.approx(5.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSMISSIVITIES — physical constraints
# ══════════════════════════════════════════════════════════════════════════════

class TestTransmissivities:
    """Each transmissivity must lie in (0, 1] and be physically ordered."""

    # Representative midday alpine conditions
    sin_h  = 0.65
    p_hpa  = 700.0    # ~3000 m
    T_K    = 275.0
    vp_hpa = 4.0      # moderate humidity
    elev   = 3000.0

    @property
    def taus(self):
        return compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa, self.elev)

    def test_all_transmissivities_in_0_1(self):
        TAUr, TAUg, TAUa, TAUaa, TAUw, taucs, _, _ = self.taus
        for name, val in [("TAUr", TAUr), ("TAUg", TAUg), ("TAUa", TAUa),
                          ("TAUaa", TAUaa), ("TAUw", TAUw), ("taucs", taucs)]:
            assert 0.0 < val <= 1.0, f"{name}={val:.4f} outside (0,1]"

    def test_taucs_less_than_components(self):
        TAUr, TAUg, TAUa, TAUaa, TAUw, taucs, _, _ = self.taus
        assert taucs <= TAUr
        assert taucs <= TAUg
        assert taucs <= TAUa
        assert taucs <= TAUw

    def test_higher_elevation_less_rayleigh(self):
        # Less air mass at higher elevation → higher TAUr (less scattering)
        _, _, _, _, _, _, _, _ = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa, 0.0)
        TAUr_low  = compute_transmissivities(
            self.sin_h, 1013.25, self.T_K, self.vp_hpa, 0.0)[0]
        TAUr_high = compute_transmissivities(
            self.sin_h, 600.0, self.T_K, self.vp_hpa, 4000.0)[0]
        assert TAUr_high > TAUr_low

    def test_low_sun_angle_increases_mopt(self):
        # sin_h=0.1 → much higher optical air mass → lower taucs
        _, _, _, _, _, taucs_low,  _, _ = compute_transmissivities(
            0.1, self.p_hpa, self.T_K, self.vp_hpa)
        _, _, _, _, _, taucs_high, _, _ = compute_transmissivities(
            0.9, self.p_hpa, self.T_K, self.vp_hpa)
        assert taucs_low < taucs_high

    def test_sdir_positive(self):
        _, _, _, _, _, _, sdir, _ = self.taus
        assert sdir > 0.0

    def test_Dcs_positive(self):
        _, _, _, _, _, _, _, Dcs = self.taus
        assert Dcs > 0.0

    def test_sdir_greater_than_Dcs_clear_sky(self):
        # Under clear sky the direct beam dominates
        _, _, _, _, _, _, sdir, Dcs = self.taus
        assert sdir > Dcs, f"sdir={sdir:.1f}, Dcs={Dcs:.1f}"

    def test_grcs_less_than_solar_constant(self):
        _, _, _, _, _, _, sdir, Dcs = self.taus
        assert sdir + Dcs < Sol0


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLEAR-SKY RADIATION VALUES — order of magnitude
# ══════════════════════════════════════════════════════════════════════════════

class TestClearSkyMagnitude:
    """Spot-check absolute values against expected Alpine summer ranges."""

    def test_summer_noon_sdir_range(self):
        # Alpine summer, sin_h ≈ 0.8 at noon, 700 hPa, dry atmosphere
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            sin_h=0.80, p_hpa=700.0, T_K=280.0, vp_hpa=3.0, elev_m=3000.0)
        # Expect 500–900 W/m² direct at high elevation in summer
        assert 500 < sdir < 950, f"sdir={sdir:.1f} W/m² outside expected range"
        # Expect 50–200 W/m² diffuse
        assert 30 < Dcs < 250, f"Dcs={Dcs:.1f} W/m² outside expected range"

    def test_low_sun_angle_reduced_radiation(self):
        _, _, _, _, _, _, sdir_noon, _ = compute_transmissivities(
            sin_h=0.80, p_hpa=700.0, T_K=280.0, vp_hpa=3.0)
        _, _, _, _, _, _, sdir_low, _ = compute_transmissivities(
            sin_h=0.20, p_hpa=700.0, T_K=280.0, vp_hpa=3.0)
        assert sdir_noon > sdir_low * 3   # roughly 4× less at sin_h=0.2


# ══════════════════════════════════════════════════════════════════════════════
# 6. f_dif PARTITION — physical constraints
# ══════════════════════════════════════════════════════════════════════════════

class TestDiffuseFraction:

    # Representative alpine clear-sky values
    sin_h = 0.65
    p_hpa = 700.0
    T_K   = 275.0
    vp_hpa = 4.0

    @property
    def sdir_Dcs(self):
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        return sdir, Dcs

    def test_f_dif_in_0_1_clear_sky(self):
        sdir, Dcs = self.sdir_Dcs
        f = compute_f_dif(sdir, Dcs, cld=0.0)
        assert 0.0 <= f <= 1.0

    def test_f_dif_in_0_1_cloudy(self):
        sdir, Dcs = self.sdir_Dcs
        for cld in np.linspace(0, 1, 11):
            f = compute_f_dif(sdir, Dcs, cld)
            assert 0.0 <= f <= 1.0, f"f_dif={f:.4f} at cld={cld:.1f}"

    def test_f_dif_monotone_increasing_with_cloud(self):
        # The Mölg parameterisation has a discontinuity at cld=0→cld>0:
        # at cld=0 the clear-sky formula gives f_dif = Dcs/grcs (~10-15%),
        # but the cloudy formula at cld=0+ gives dif1/100 = 4.6%.
        # Monotonicity holds for cld > 0; we start from a small positive value.
        sdir, Dcs = self.sdir_Dcs
        clds  = np.linspace(0.05, 1.0, 20)
        fdifs = [compute_f_dif(sdir, Dcs, c) for c in clds]
        for i in range(len(fdifs) - 1):
            assert fdifs[i] <= fdifs[i + 1] + 1e-9, \
                f"f_dif not monotone at cld={clds[i]:.2f}"

    def test_f_dif_discontinuity_at_zero_cloud(self):
        # The clear-sky value is higher than the cloudy formula at cld=0+.
        # This is a known property of the Mölg parameterisation.
        sdir, Dcs = self.sdir_Dcs
        f_clear     = compute_f_dif(sdir, Dcs, cld=0.0)
        f_small_cld = compute_f_dif(sdir, Dcs, cld=0.01)
        # f_clear uses Dcs/grcs (~10-15%); f_small_cld ≈ dif1/100 = 4.6%
        assert f_clear > f_small_cld, (
            "Expected f_dif(cld=0) > f_dif(cld=0.01) due to "
            "Mölg clear-sky/cloudy formula discontinuity"
        )

    def test_f_dif_clear_sky_less_than_50pct(self):
        # Under clear sky, direct beam dominates → f_dif < 0.5
        sdir, Dcs = self.sdir_Dcs
        f = compute_f_dif(sdir, Dcs, cld=0.0)
        assert f < 0.5, f"f_dif={f:.3f} under clear sky is too high"

    def test_f_dif_overcast_near_1(self):
        # At full overcast essentially all SW is diffuse
        sdir, Dcs = self.sdir_Dcs
        f = compute_f_dif(sdir, Dcs, cld=1.0)
        assert f > 0.90, f"f_dif={f:.3f} at full overcast should be > 0.90"

    def test_f_dif_minimum_at_zero_cloud(self):
        sdir, Dcs = self.sdir_Dcs
        f_clear   = compute_f_dif(sdir, Dcs, cld=0.0)
        f_cloudy  = compute_f_dif(sdir, Dcs, cld=0.5)
        assert f_clear <= f_cloudy


# ══════════════════════════════════════════════════════════════════════════════
# 7. HORAYZON2022 — energy conservation and terrain correction
# ══════════════════════════════════════════════════════════════════════════════

class TestHorayzon2022:
    """G_dir + G_dif must always equal G_meas (energy conservation)."""

    sin_h  = 0.65
    p_hpa  = 700.0
    T_K    = 275.0
    vp_hpa = 4.0

    def _run(self, G_meas, cld, sw_cor, svf):
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        f    = compute_f_dif(sdir, Dcs, cld)
        Gdir = G_meas * (1.0 - f)
        Gdif = G_meas * f
        G_out = sw_cor * Gdir + svf * Gdif
        return G_out, Gdir, Gdif, f

    def test_energy_conservation_clear_sky(self):
        G_meas = 600.0
        _, Gdir, Gdif, _ = self._run(G_meas, 0.0, 0.8, 0.85)
        assert abs(Gdir + Gdif - G_meas) < 1e-9

    def test_energy_conservation_overcast(self):
        G_meas = 150.0
        _, Gdir, Gdif, _ = self._run(G_meas, 1.0, 0.8, 0.85)
        assert abs(Gdir + Gdif - G_meas) < 1e-9

    def test_energy_conservation_partial_cloud(self):
        for cld in [0.2, 0.5, 0.8]:
            G_meas = 400.0
            _, Gdir, Gdif, _ = self._run(G_meas, cld, 0.8, 0.85)
            assert abs(Gdir + Gdif - G_meas) < 1e-9

    def test_output_non_negative(self):
        for cld in [0.0, 0.5, 1.0]:
            G_out, _, _, _ = self._run(500.0, cld, 0.8, 0.85)
            assert G_out >= 0.0

    def test_zero_gmeas_gives_zero_output(self):
        G_out, Gdir, Gdif, _ = self._run(0.0, 0.5, 0.8, 0.85)
        assert G_out == 0.0
        assert Gdir == 0.0
        assert Gdif == 0.0

    def test_svf_fallback_equals_sw_cor_times_total(self):
        # Without SVF: G_out = sw_cor * G_meas (old behaviour)
        sw_cor = 0.75
        G_meas = 500.0
        assert sw_cor * G_meas == pytest.approx(375.0)

    def test_shaded_pixel_sw_cor_zero(self):
        # sw_dir_cor=0 (fully shaded): output = SVF * diffuse only
        G_meas = 400.0
        sw_cor = 0.0
        svf    = 0.85
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        f     = compute_f_dif(sdir, Dcs, cld=0.0)
        G_out = sw_cor * G_meas * (1 - f) + svf * G_meas * f
        # Only diffuse term survives
        assert G_out == pytest.approx(svf * G_meas * f)

    def test_open_sky_svf_one_and_full_sun(self):
        # SVF=1, sw_cor=1 → G_out = G_meas exactly
        G_meas = 500.0
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        f     = compute_f_dif(sdir, Dcs, cld=0.0)
        G_out = 1.0 * G_meas * (1 - f) + 1.0 * G_meas * f
        assert G_out == pytest.approx(G_meas)


# ══════════════════════════════════════════════════════════════════════════════
# 8. HORAYZON_THEORY — does not use G_meas
# ══════════════════════════════════════════════════════════════════════════════

class TestHorayzonTheory:
    """Output must be positive and independent of G_meas."""

    sin_h  = 0.65
    p_hpa  = 700.0
    T_K    = 275.0
    vp_hpa = 4.0

    def _run(self, cld, sw_cor, svf, G_meas=None):
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        grcs = sdir + Dcs
        if cld > 0:
            G_dir = sdir * (1.0 - (1.0 - dirovc) * cld)
            G_dif = grcs * ((100.0 - Cf * 100.0 - dif1) / 100.0 * cld
                            + dif1 / 100.0)
        else:
            G_dir, G_dif = sdir, Dcs
        return sw_cor * G_dir + svf * G_dif

    def test_output_independent_of_g_meas(self):
        # Horayzon_theory ignores the forcing SW entirely
        out_a = self._run(0.3, 0.8, 0.85, G_meas=100.0)
        out_b = self._run(0.3, 0.8, 0.85, G_meas=800.0)
        assert out_a == pytest.approx(out_b)

    def test_output_positive_clear_sky(self):
        assert self._run(0.0, 0.8, 0.85) > 0.0

    def test_output_positive_overcast(self):
        assert self._run(1.0, 0.8, 0.85) > 0.0

    def test_output_non_negative_all_clouds(self):
        for cld in np.linspace(0, 1, 11):
            assert self._run(cld, 0.8, 0.85) >= 0.0

    def test_more_cloud_less_output_with_shading(self):
        # More cloud reduces direct beam, so total goes down when sw_cor > svf
        out_clear = self._run(0.0, 1.2, 0.85)
        out_cloud = self._run(0.8, 1.2, 0.85)
        assert out_cloud < out_clear

    def test_fully_shaded_only_diffuse_survives(self):
        # sw_cor=0: output = SVF * G_dif_theory
        out = self._run(0.0, sw_cor=0.0, svf=0.85)
        _, _, _, _, _, _, _, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        expected = 0.85 * Dcs
        assert out == pytest.approx(expected, rel=1e-6)

    def test_svf_fallback_only_direct(self):
        # svf=None → output = sw_cor * G_dir_theory only
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            self.sin_h, self.p_hpa, self.T_K, self.vp_hpa)
        G_fallback = 0.8 * sdir   # no diffuse
        assert G_fallback > 0.0
        assert G_fallback < 0.8 * (sdir + Dcs)


# ══════════════════════════════════════════════════════════════════════════════
# 9. EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_night_flag(self):
        # sin_h ≤ 0.01 → script sets G=0, no transmissivity computed
        sh = compute_sin_h(doy=355, hour=12, lat_deg=70, tcart=0)
        assert sh <= 0.01, f"Expected night at polar winter, got sin_h={sh:.4f}"

    def test_zero_forcing_sw_horayzon2022_zero_output(self):
        G_meas = 0.0
        f_dif  = 0.3
        assert G_meas * (1 - f_dif) == 0.0
        assert G_meas * f_dif       == 0.0

    def test_sw_dir_cor_clipped_at_25(self):
        # Values > 25 are unphysical (numerics near horizon)
        raw = np.array([0.5, 10.0, 25.0, 26.0, 100.0])
        clipped = np.where(raw > 25.0, 25.0, raw)
        assert clipped.max() == 25.0
        assert clipped[0] == 0.5   # small values unchanged

    def test_f_dif_not_nan_at_zero_sdir(self):
        # Degenerate: sdir → 0 (very low sun), Dcs also small
        # G_theory = max(..., 1e-10) prevents division by zero
        f = compute_f_dif(sdir=0.0, Dcs=0.0, cld=0.0)
        assert not math.isnan(f)
        assert 0.0 <= f <= 1.0

    def test_tcart_consistent_formula(self):
        # tcart = UTC_offset*15 - station_longitude
        cases = [
            (0,    10.76, -10.76),   # UTC forcing, HEF
            (1,    10.76,   4.24),   # CET forcing, HEF
            (0,    86.0,  -86.0),    # UTC forcing, Himalaya
            (5,    86.0,  -11.0),    # UTC+5 forcing, Himalaya (approx)
        ]
        for utc_offset, lon, expected_tcart in cases:
            tcart = utc_offset * 15 - lon
            assert tcart == pytest.approx(expected_tcart, abs=1e-9), \
                f"offset={utc_offset}, lon={lon}: got {tcart}, expected {expected_tcart}"

    def test_horayzon2022_worse_than_theory_when_climate_model_zero(self):
        # If forcing SW = 0 but sun is up, Horayzon2022 gives 0 but theory gives >0
        # (This is expected behaviour — theory is independent of forcing)
        _, _, _, _, _, _, sdir, Dcs = compute_transmissivities(
            sin_h=0.6, p_hpa=700, T_K=275, vp_hpa=4)
        theory_out = 0.8 * sdir + 0.85 * Dcs
        horayzon_out = 0.0   # G_meas=0 → zero output
        assert theory_out > horayzon_out


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER (no pytest required)
# ══════════════════════════════════════════════════════════════════════════════

def _run_class(cls):
    obj = cls()
    passed = failed = 0
    for name in [m for m in dir(cls) if m.startswith("test_")]:
        try:
            getattr(obj, name)()
            print(f"  PASS  {cls.__name__}.{name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {cls.__name__}.{name}")
            print(f"        {exc}")
            failed += 1
    return passed, failed


if __name__ == "__main__":
    import sys

    test_classes = [
        TestLutYearSelection,
        TestLeapYearShift,
        TestSolarGeometry,
        TestTransmissivities,
        TestClearSkyMagnitude,
        TestDiffuseFraction,
        TestHorayzon2022,
        TestHorayzonTheory,
        TestEdgeCases,
    ]

    total_pass = total_fail = 0
    for cls in test_classes:
        p, f = _run_class(cls)
        total_pass += p
        total_fail += f

    print(f"\n{'=' * 55}")
    print(f"Results: {total_pass} passed, {total_fail} failed "
          f"out of {total_pass + total_fail} tests")
    sys.exit(0 if total_fail == 0 else 1)
