"""
spice_common.py -- run ngspice (conda-forge MSVC build) in batch mode from Python
and parse its output. No PySpice dependency: we write a .cir, call ngspice_con.exe
-b, and read back wrdata ASCII tables. Robust on Windows / behind a flaky network.

The op-amp is a transparent single-pole macromodel whose parameters are taken from
a NAMED real low-power op-amp (default: TI OPA2376, GBW 5.5 MHz, A0 134 dB,
en 7.5 nV/rtHz, Icc 760 uA/ch) so every non-ideality (finite gain-bandwidth Q
droop, input voltage noise, output clipping, supply current -> power) is grounded
in a datasheet number, not invented.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np

# ---- locate the ngspice install (override with the NGSPICE_DIR env var) ----
# NGSPICE_DIR must point to the install's "Library" directory -- the one that
# contains bin/ngspice_con.exe and share/ngspice. The conda-forge ngspice build
# lays it out that way; see the README (spice branch) for how to obtain it.
NG_ROOT = Path(os.environ.get(
    "NGSPICE_DIR",
    r"E:\_7_Scientific_Research\0进行论文\DPNN\tools\ngspice\Library"))
NG_EXE = NG_ROOT / "bin" / "ngspice_con.exe"
NG_SHARE = NG_ROOT / "share" / "ngspice"


def _env():
    e = dict(os.environ)
    e["SPICE_LIB_DIR"] = str(NG_SHARE)
    e["SPICE_EXEC_DIR"] = str(NG_ROOT / "bin")
    return e


# ---------------------------------------------------------------------------
# Op-amp macromodel (single dominant pole) parameterised after a real part.
# GBW [Hz], A0 [V/V open-loop DC gain], VHI/VLO output swing [V], EN input-
# referred white voltage noise [V/rtHz] injected as the thermal noise of a
# series resistor Rn = EN^2 / (4kT).  4kT(300K) = 1.656e-20.
# ---------------------------------------------------------------------------
OPAMP_SUBCKT = r"""* single-pole op-amp macromodel (params ~ TI OPA2376)
.subckt opamp inp inn out PARAMS: GBW=5.5e6 A0=3.16e6 VHI=1.5 VLO=-1.5 EN=7.5e-9
Rn   inp  np  {EN*EN/1.656e-20}
G1   0 g  np inn 1
R1   g 0  {A0}
C1   g 0  {1/(6.283185307*GBW)}
Bout out 0 V = max({VLO}, min({VHI}, V(g)))
.ends"""


def run_ac(netlist_body: str, fstart: float, fstop: float, ndec: int,
           probe: str = "bp") -> tuple[np.ndarray, np.ndarray]:
    """Run an AC analysis; return (freq[Hz], complex H) for node `probe`.
    netlist_body must define the circuit; this adds the .control AC block."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ac.txt"
        cir = Path(td) / "c.cir"
        ctrl = f""".control
ac dec {ndec} {fstart} {fstop}
wrdata {out.as_posix()} v({probe})
.endc
.end
"""
        cir.write_text(netlist_body + "\n" + ctrl, encoding="utf-8")
        r = subprocess.run([str(NG_EXE), "-b", str(cir)], capture_output=True,
                           text=True, env=_env(), timeout=120)
        if not out.exists():
            raise RuntimeError("ngspice produced no AC output:\n" + r.stdout[-2000:] + r.stderr[-1000:])
        d = np.loadtxt(out)
    # wrdata complex: columns = freq, real, imag
    f = d[:, 0]
    H = d[:, 1] + 1j * d[:, 2]
    return f, H


def run_ac_lin(netlist_body: str, fstart: float, fstop: float, npts: int,
               probe: str = "bp") -> tuple[np.ndarray, np.ndarray]:
    """Run a LINEAR AC sweep (npts points fstart..fstop); return (freq, complex H)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ac.txt"
        cir = Path(td) / "c.cir"
        ctrl = f""".control
ac lin {npts} {fstart} {fstop}
wrdata {out.as_posix()} v({probe})
.endc
.end
"""
        cir.write_text(netlist_body + "\n" + ctrl, encoding="utf-8")
        r = subprocess.run([str(NG_EXE), "-b", str(cir)], capture_output=True,
                           text=True, env=_env(), timeout=180)
        if not out.exists():
            raise RuntimeError("ngspice produced no AC output:\n" + r.stdout[-2000:] + r.stderr[-1000:])
        d = np.loadtxt(out)
    return d[:, 0], d[:, 1] + 1j * d[:, 2]


def run_noise(netlist_body: str, fstart: float, fstop: float, ndec: int,
              out_node: str, src: str) -> tuple[np.ndarray, np.ndarray]:
    """Run a .noise analysis; return (freq, output-referred noise PSD [V^2/Hz])."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "n.txt"
        cir = Path(td) / "c.cir"
        ctrl = f""".control
noise v({out_node}) {src} dec {ndec} {fstart} {fstop}
setplot noise1
wrdata {out.as_posix()} onoise_spectrum
.endc
.end
"""
        cir.write_text(netlist_body + "\n" + ctrl, encoding="utf-8")
        r = subprocess.run([str(NG_EXE), "-b", str(cir)], capture_output=True,
                           text=True, env=_env(), timeout=120)
        if not out.exists():
            raise RuntimeError("ngspice produced no noise output:\n" + r.stdout[-2000:] + r.stderr[-1000:])
        d = np.loadtxt(out)
    return d[:, 0], d[:, 1]


def measure_f0_Q(f: np.ndarray, H: np.ndarray) -> tuple[float, float, float]:
    """From a bandpass AC response, return (f0_meas, Q_meas, peak_gain).
    f0 = freq of |H| peak; Q = f0 / (-3dB bandwidth) via interpolation."""
    mag = np.abs(H)
    k = int(np.argmax(mag))
    peak = mag[k]
    f0 = f[k]
    half = peak / np.sqrt(2.0)
    # walk left/right to the -3 dB crossings, linear-interp in log-f
    def cross(idx_range):
        prev = k
        for i in idx_range:
            if mag[i] < half:
                # interp between i and prev in log-f
                lf0, lf1 = np.log10(f[prev]), np.log10(f[i])
                m0, m1 = mag[prev], mag[i]
                t = (half - m0) / (m1 - m0 + 1e-30)
                return 10 ** (lf0 + t * (lf1 - lf0))
            prev = i
        return None
    fl = cross(range(k - 1, -1, -1))
    fh = cross(range(k + 1, len(f)))
    if fl is None or fh is None or fh <= fl:
        return float(f0), float("nan"), float(peak)
    Q = f0 / (fh - fl)
    return float(f0), float(Q), float(peak)


def tow_thomas_section(f0: float, Q: float, C: float = 100e-9,
                       opamp_params: str = "") -> str:
    """Return a netlist body (input source V1 + one Tow-Thomas bandpass section,
    BP output = node 'bp') for centre freq f0 [Hz], quality Q.
    omega0 = 1/(R C) -> R = 1/(2 pi f0 C);  Q = RQ/R -> RQ = Q R;  midband gain
    = RQ/R1; set R1=RQ for unity midband gain (avoids clipping)."""
    R = 1.0 / (2 * np.pi * f0 * C)
    RQ = Q * R
    R1 = RQ
    RR = 10e3
    op = f"opamp {opamp_params}".strip()
    return f"""* Tow-Thomas bandpass section  f0={f0:.3f}Hz Q={Q:.1f}
{OPAMP_SUBCKT}
V1 in 0 AC 1 SIN(0 0.2 {f0})
* A1 lossy integrator -> BP
XA1 0 x1 bp {op}
Ri  in x1 {R1:.6g}
Rqr bp x1 {RQ:.6g}
Cf  bp x1 {C:.6g}
Rlp v3 x1 {R:.6g}
* A2 integrator -> LP
XA2 0 x2 lp {op}
Rb  bp x2 {R:.6g}
Ci  lp x2 {C:.6g}
* A3 unity inverter -> v3 = -LP
XA3 0 x3 v3 {op}
Rin3 lp x3 {RR:.6g}
Rf3  v3 x3 {RR:.6g}
"""


if __name__ == "__main__":
    # smoke test: one mid-band section, ideal vs measured f0/Q
    f0_t, Q_t = 100.0, 40.0
    body = tow_thomas_section(f0_t, Q_t)
    f, H = run_ac(body, 10, 1000, 200, probe="bp")
    f0_m, Q_m, g = measure_f0_Q(f, H)
    print(f"target  f0={f0_t} Hz  Q={Q_t}")
    print(f"ngspice f0={f0_m:.2f} Hz  Q={Q_m:.2f}  peak_gain={g:.3f} ({20*np.log10(g):.2f} dB)")
