"""
spice_reservoir.py -- extract the resonator bank's transfer matrix, noise, and
power from a CIRCUIT-LEVEL ngspice simulation, to replace the abstract closed-
form transfer function used in the paper's reservoir model.

For each of the n_modes=32 modes (log-spaced centre frequencies matching core.py)
we build a Tow-Thomas state-variable bandpass section (3 op-amps; the only
topology that holds the paper's nominal Q=40 against finite op-amp gain-
bandwidth), run an AC analysis with a single-pole op-amp macromodel parameterised
after a real low-power part (TI OPA376: GBW 5.5 MHz, A0 134 dB, en 7.5 nV/rtHz,
Icc 760 uA/ch), and read |H_m(f)|^2 onto the SAME 256-point linear frequency grid
the paper's reservoir uses. We also run a .noise analysis per section and account
the supply power from the op-amp count and datasheet Icc.

Outputs:
  spice_reservoir.npz   -- Hsq_spice[32,256] (row-normalised), freq grid, mode f0,
                           measured f0/Q/gain per mode, per-mode output noise
  spice_reservoir_results.json -- human-readable summary incl. the REAL power that
                           replaces the energy ledger's afe/bias/readout guesses
Run:  py spice_reservoir.py     (~1-2 min CPU)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from spice_common import (tow_thomas_section, run_ac, run_noise, measure_f0_Q)

ROOT = Path(__file__).resolve().parent

# ---- geometry MUST match core.Config / PiezoResonatorArray ----
F_LO, F_HI, N_FREQ, N_MODES, Q_NOM = 10.0, 1000.0, 256, 32, 40.0
GRID = np.linspace(F_LO, F_HI, N_FREQ)                       # core's freqs
MODE_F0 = np.logspace(np.log10(F_LO * 1.2), np.log10(F_HI * 0.9), N_MODES)  # core's wm/2pi
C_FIX = 100e-9                                               # 100 nF integrator caps

# ---- power model (TI OPA376, low-power precision CMOS) ----
ICC_PER_CH = 760e-6     # A, typical supply current per amplifier
VSUP = 3.3              # V, single 3.3 V supply (rail-to-rail)
OPAMPS_PER_SECTION = 3  # Tow-Thomas: summer-integrator + integrator + inverter
INPUT_BUFFER_OPAMPS = 1
READOUT_ADC_MW = 3.0    # low-rate decision-score ADC (kept from ledger estimate)


def extract_bank():
    t0 = time.time()
    Hsq = np.zeros((N_MODES, N_FREQ))
    meas = []
    noise_rms = np.zeros(N_MODES)
    for m, f0 in enumerate(MODE_F0):
        body = tow_thomas_section(float(f0), Q_NOM, C=C_FIX)
        # dense AC then sample onto the exact core grid (= exact H at each grid freq)
        f, H = _dense_ac(body)
        mag = np.abs(H)
        # |H(f)|^2 on the core grid via interpolation in the value (not undersampled)
        h_grid = np.interp(GRID, f, mag)
        Hsq[m] = h_grid ** 2
        f0m, Qm, g = measure_f0_Q(f, H)
        # per-section output noise integrated over the band
        try:
            fn, psd = run_noise(body, F_LO, F_HI, 30, out_node="bp", src="V1")
            vrms = float(np.sqrt(np.trapezoid(psd, fn)))
        except Exception as e:
            vrms = float("nan")
        noise_rms[m] = vrms
        meas.append({"mode": m, "f0_target": round(float(f0), 3),
                     "f0_meas": round(f0m, 3), "Q_meas": round(Qm, 2),
                     "gain_dB": round(20 * np.log10(max(g, 1e-9)), 2),
                     "noise_uVrms": round(vrms * 1e6, 3) if np.isfinite(vrms) else None})
        if m % 8 == 0:
            print(f"  mode {m:2d} f0={f0:7.2f} -> meas f0={f0m:7.2f} Q={Qm:5.1f} "
                  f"gain={20*np.log10(max(g,1e-9)):+5.2f}dB noise={vrms*1e6:.2f}uVrms", flush=True)
    # row-normalise like core: Hsq / Hsq.amax(dim=1)
    Hsq_norm = Hsq / (Hsq.max(axis=1, keepdims=True) + 1e-30)

    # ---- power that REPLACES the ledger's afe/bias/readout guesses ----
    n_opamps = N_MODES * OPAMPS_PER_SECTION + INPUT_BUFFER_OPAMPS
    p_opamps_mW = n_opamps * ICC_PER_CH * VSUP * 1e3
    p_analog_mW = p_opamps_mW + READOUT_ADC_MW

    Qm_all = np.array([d["Q_meas"] for d in meas if np.isfinite(d["Q_meas"])])
    summary = {
        "what": "Circuit-level (ngspice) extraction of the resonator reservoir bank",
        "topology": "Tow-Thomas 3-op-amp state-variable bandpass per mode",
        "opamp_model": "single-pole macromodel ~ TI OPA376 (GBW 5.5MHz, A0 134dB, en 7.5nV/rtHz, Icc 760uA/ch)",
        "n_modes": N_MODES, "Q_nominal": Q_NOM, "C_nF": C_FIX * 1e9,
        "mode_f0_target_Hz": [round(float(x), 2) for x in MODE_F0],
        "Q_meas_mean": round(float(Qm_all.mean()), 2),
        "Q_meas_min": round(float(Qm_all.min()), 2),
        "Q_meas_max": round(float(Qm_all.max()), 2),
        "Q_droop_pct_mean": round(float(100 * (Q_NOM - Qm_all.mean()) / Q_NOM), 1),
        "per_mode": meas,
        "noise_uVrms_mean": round(float(np.nanmean(noise_rms) * 1e6), 3),
        "POWER": {
            "topology_opamps_per_section": OPAMPS_PER_SECTION,
            "total_opamps": n_opamps,
            "icc_per_ch_uA": ICC_PER_CH * 1e6, "vsup_V": VSUP,
            "analog_frontend_mW_budgeted": round(p_analog_mW, 1),
            "ledger_guess_afe_bias_readout_mW": 13.0,
            "note": ("This REPLACES the energy ledger's ~13 mW guess (afe 6 + bias 4 + readout 3) "
                     f"for the analog front end with a circuit-grounded {p_analog_mW:.0f} mW: holding "
                     f"the paper's Q=40 across 32 modes needs 3 op-amps/section, and a real low-power "
                     f"op-amp draws ~{ICC_PER_CH*1e6:.0f} uA each. The energy advantage shrinks "
                     f"accordingly and motivates a leaner (fewer-mode / lower-Q) design."),
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    np.savez(ROOT / "spice_reservoir.npz", Hsq_spice=Hsq_norm, grid=GRID,
             mode_f0=MODE_F0, Q_meas=np.array([d["Q_meas"] for d in meas]),
             f0_meas=np.array([d["f0_meas"] for d in meas]), noise_rms=noise_rms,
             analog_frontend_mW=p_analog_mW)
    (ROOT / "spice_reservoir_results.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
    print(f"\n[spice_reservoir] Q meas mean {summary['Q_meas_mean']} "
          f"(droop {summary['Q_droop_pct_mean']}%), analog front end "
          f"{p_analog_mW:.0f} mW (ledger guessed 13 mW), {summary['wall_clock_s']}s")
    return Hsq_norm, summary


def _dense_ac(body):
    """Dense linear AC sweep over the band so interpolation onto the 256-grid is
    the true H value at each grid point (not an undersampled peak)."""
    from spice_common import run_ac_lin
    return run_ac_lin(body, F_LO, F_HI, npts=6000, probe="bp")


if __name__ == "__main__":
    extract_bank()
