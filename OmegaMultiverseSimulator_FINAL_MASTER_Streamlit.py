import os
import datetime
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio

from fpdf import FPDF

# Optional OpenAI (won't crash if missing key)
try:
    import openai
except Exception:
    openai = None


# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(page_title="Omega Multiverse Simulator", layout="wide")
st.title("Omega Multiverse Simulator")

# Divider compatibility (older Streamlit)
if not hasattr(st, "divider"):
    def _divider_fallback():
        st.markdown("---")
    st.divider = _divider_fallback


# =========================
# Globals / Helpers
# =========================
OUTPUT_DIR = "pdf_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# filename -> caption text (value dependent)
FIGURE_CAPTIONS = {}

st.sidebar.header("Controls")

disable_3d = st.sidebar.checkbox(
    "Disable 3D (recommended on iPhone if plots appear blank)",
    value=False
)

auto_save_plots = st.sidebar.checkbox(
    "Auto-save plots for PDF (can slow app)",
    value=True
)

# 3D resolution (lower = faster/more reliable on iPhone)
res_3d = st.sidebar.slider(
    "3D resolution (lower = faster / more reliable on iPhone)",
    min_value=25, max_value=120, value=60, step=5
)


# =========================
# Caption helpers (value-dependent)
# =========================
def _latin1_safe(s: str) -> str:
    return (s or "").encode("latin-1", "replace").decode("latin-1")

def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _cap_header(constants: dict) -> str:
    S = constants["Strong Force Multiplier"]
    EM = constants["Electromagnetic Force Multiplier"]
    W = constants["Weak Force Multiplier"]
    G = constants["Gravitational Constant Multiplier"]
    DE = constants["Dark Energy Multiplier"]
    T = constants["Temperature Multiplier"]
    P = constants["Pressure Multiplier"]
    deviation = sum(abs(v - 1.0) for v in constants.values())
    return (
        f"Run constants: S={S:.2f}, EM={EM:.2f}, W={W:.2f}, G={G:.2f}, DE={DE:.2f}, T={T:.2f}, P={P:.2f} "
        f"(total deviation={deviation:.2f})."
    )

def _surface_stats(z, x=None, y=None):
    z = np.asarray(z, dtype=float)
    zf = z[np.isfinite(z)]
    if zf.size == 0:
        return {"min": np.nan, "mean": np.nan, "max": np.nan, "peak": None}
    out = {
        "min": float(np.min(zf)),
        "mean": float(np.mean(zf)),
        "max": float(np.max(zf)),
        "peak": None
    }
    if x is not None and y is not None and z.shape == (len(y), len(x)):
        iy, ix = np.unravel_index(np.nanargmax(z), z.shape)
        out["peak"] = (float(x[ix]), float(y[iy]), float(out["max"]))
    return out

def _line_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return {"min": np.nan, "mean": np.nan, "max": np.nan, "peak": None, "end": np.nan}
    xm, ym = x[m], y[m]
    idx = int(np.argmax(ym))
    return {
        "min": float(np.min(ym)),
        "mean": float(np.mean(ym)),
        "max": float(np.max(ym)),
        "peak": (float(xm[idx]), float(ym[idx])),
        "end": float(ym[-1])
    }

def set_caption(filename: str, text: str):
    if filename:
        FIGURE_CAPTIONS[filename] = _latin1_safe(text)

def build_caption(title: str, constants: dict, metrics: dict, extras: dict | None = None) -> str:
    """
    Value-dependent explanation generator for ALL graphs.
    Uses (constants) + (metrics from computed arrays) + optional extras.
    """
    extras = extras or {}
    S = constants["Strong Force Multiplier"]
    EM = constants["Electromagnetic Force Multiplier"]
    W = constants["Weak Force Multiplier"]
    G = constants["Gravitational Constant Multiplier"]
    DE = constants["Dark Energy Multiplier"]
    T = constants["Temperature Multiplier"]
    P = constants["Pressure Multiplier"]
    deviation = sum(abs(v - 1.0) for v in constants.values())

    header = _cap_header(constants)
    lines = [header, f"Figure: {title}"]

    # Normalize a common stats printout if present
    if "min" in metrics and "mean" in metrics and "max" in metrics:
        lines.append(
            f"Stats: min={_fmt(metrics['min'])}, mean={_fmt(metrics['mean'])}, max={_fmt(metrics['max'])}."
        )

    # Peak reporting
    if metrics.get("peak") is not None:
        pk = metrics["peak"]
        if len(pk) == 3:
            lines.append(f"Peak location: x={_fmt(pk[0])}, y={_fmt(pk[1])}, value={_fmt(pk[2])}.")
        elif len(pk) == 2:
            lines.append(f"Peak location: x={_fmt(pk[0])}, value={_fmt(pk[1])}.")

    # Tailored, value-dependent interpretations by title keywords
    t_low = title.lower()

    if "periodic table stability" in t_low or "stability vs atomic number" in t_low:
        lines.append(
            "Meaning: This surface is a proxy for how binding vs Coulomb repulsion shifts across atomic number."
        )
        lines.append(
            f"Value impact: Higher S={S:.2f} boosts cohesion; higher EM={EM:.2f} increases repulsion especially at high Z; "
            f"W={W:.2f} shapes beta-stability. A higher peak/mean implies a broader viable periodic table."
        )

    elif "island of instability" in t_low or "instability" in t_low:
        lines.append("Meaning: This map highlights synthetic instability bands where shell-like structure fails to stabilize nuclei.")
        lines.append(
            f"Value impact: With EM={EM:.2f}, heavy nuclei destabilize faster; shifting S={S:.2f} moves where instability spikes occur. "
            "Higher maxima indicate a larger region of short-lived heavy isotopes."
        )

    elif "star formation" in t_low:
        rad_pressure = extras.get("rad_pressure")
        ignition = extras.get("ignition")
        if rad_pressure is not None:
            lines.append(f"Radiation-pressure factor (from EM): {float(rad_pressure):.3f}.")
        if ignition is not None:
            lines.append(f"Ignition factor (from S/W): {float(ignition):.3f}.")
        lines.append("Meaning: Gravity-driven collapse competes with dark-energy expansion and radiative feedback.")
        lines.append(
            f"Value impact: Higher G={G:.2f} increases collapse; higher DE={DE:.2f} suppresses structure; "
            f"EM={EM:.2f} feeds radiative resistance. If mean potential is low, fewer stars form → fewer heavy elements → simpler chemistry."
        )

    elif "life probability" in t_low and "over time" not in t_low:
        metals_factor = extras.get("metals_factor")
        thermo_factor = extras.get("thermo_factor")
        if metals_factor is not None:
            lines.append(f"Metallicity proxy factor: {float(metals_factor):.3f}.")
        if thermo_factor is not None:
            lines.append(f"Thermo window factor (from T/P): {float(thermo_factor):.3f}.")
        lines.append("Meaning: A proxy ‘joint window’ where chemistry + environment + element availability align.")
        lines.append(
            f"Value impact: With T={T:.2f}, P={P:.2f}, the thermo gate tightens/loosens; "
            f"G={G:.2f} vs DE={DE:.2f} influences metals. Higher peaks imply parameter regions where complex molecules are more likely to persist."
        )

    elif "quantum bonding" in t_low or "bonding" in t_low and "molecular bonding viability" not in t_low:
        temp_kill = extras.get("temp_kill")
        press_help = extras.get("pressure_help")
        if temp_kill is not None:
            lines.append(f"Temperature suppression factor: {float(temp_kill):.3f}.")
        if press_help is not None:
            lines.append(f"Pressure overlap factor: {float(press_help):.3f}.")
        lines.append("Meaning: Proxy for how stable electron sharing/overlap is across force settings.")
        lines.append(
            f"Value impact: EM near 1 typically maximizes binding; high |EM-1| reduces it. "
            f"Temperature T={T:.2f} suppresses bonding; pressure P={P:.2f} can partially restore overlap. "
            "Higher mean implies more robust molecular diversity."
        )

    elif "viability" in t_low or "emergence" in t_low:
        viability = extras.get("viability")
        chaos = extras.get("chaos")
        if viability is not None and chaos is not None:
            lines.append(f"Computed: viability={float(viability):.3f}, chaos={float(chaos):.3f}.")
        lines.append("Meaning: A compact global ‘distance-from-baseline’ proxy (not a true cosmological probability).")
        lines.append(
            f"Value impact: As deviation={deviation:.2f} increases, this score decays exponentially. "
            "Low viability implies multiple pillars degrade simultaneously (nuclear stability, chemistry window, stellar processing)."
        )

    elif "element abundance vs z" in t_low or ("element abundance" in t_low and "evolution" not in t_low):
        peakZ = extras.get("peakZ")
        peakVal = extras.get("peakVal")
        if peakZ is not None and peakVal is not None:
            lines.append(f"Peak abundance at Z={int(peakZ)} with value={float(peakVal):.3f}.")
        lines.append("Meaning: Proxy relative abundance of elements produced and retained by nuclear + stellar filters.")
        lines.append(
            f"Value impact: If abundance collapses toward low Z, the universe trends to H/He dominance → limited molecule types. "
            f"Higher mid-Z abundance supports richer chemistry (C/N/O/Si/Fe-like complexity)."
        )

    elif "abundance evolution" in t_low:
        stellar_factor = extras.get("stellar_factor")
        if stellar_factor is not None:
            lines.append(f"Enrichment driver (stellar_factor): {float(stellar_factor):.3f}.")
        lines.append("Meaning: How quickly different Z-bands appear across a normalized cosmic timeline.")
        lines.append(
            "Value impact: Earlier/higher enrichment boosts availability of heavy elements sooner, widening the molecular and planetary chemistry phase space."
        )

    elif "em radiation risk" in t_low:
        risk_at_current = extras.get("risk_at_current")
        if risk_at_current is not None:
            lines.append(f"Risk at current EM={EM:.2f}: {float(risk_at_current):.3f}.")
        lines.append("Meaning: Proxy for how strongly EM coupling pushes radiation intensity/interaction.")
        lines.append(
            "Value impact: Higher risk can sterilize surfaces and increase bond-breaking; low risk favors molecular persistence and stable climates."
        )

    elif "star lifespan" in t_low:
        tau_at_G = extras.get("tau_at_G")
        if tau_at_G is not None:
            lines.append(f"Relative lifetime at current G={G:.2f}: {float(tau_at_G):.3f}.")
        lines.append("Meaning: Proxy for how fast stars burn fuel given gravity-driven mass scaling.")
        lines.append(
            "Value impact: Short lifetimes reduce time for biological/chemical complexity to accumulate; longer lifetimes broaden the habitability window."
        )

    elif "cosmic web" in t_low or "dark matter" in t_low:
        clustering = extras.get("clustering")
        count_thr = extras.get("count_thr")
        if clustering is not None:
            lines.append(f"Clustering metric (std of density slice): {float(clustering):.3f}.")
        if count_thr is not None:
            lines.append(f"3D points above threshold: {int(count_thr)}.")
        lines.append("Meaning: Proxy structure formation (filaments/voids) driven by G vs DE.")
        lines.append(
            f"Value impact: Higher G={G:.2f} compresses structure; higher DE={DE:.2f} stretches voids. "
            "Weaker structure generally reduces star formation and metal production."
        )

    elif "isotope stability" in t_low or "atomic / isotope stability" in t_low:
        viable_count = extras.get("viable_count")
        if viable_count is not None:
            lines.append(f"Count of viable isotopes (thresholded): {int(viable_count)}.")
        lines.append("Meaning: A valley-of-stability proxy over (Z,N) space.")
        lines.append(
            f"Value impact: Higher mean stability implies more long-lived nuclides → more stable elements and isotopes for chemistry and geology. "
            f"EM={EM:.2f} penalizes high Z; S={S:.2f} provides binding; W={W:.2f} shapes beta-decay routes."
        )

    elif "life probability over time" in t_low or "life probability over cosmic time" in t_low:
        peak_time = extras.get("peak_time")
        final = metrics.get("end")
        if peak_time is not None:
            lines.append(f"Peak time: t={float(peak_time):.3f} (normalized).")
        if final is not None and np.isfinite(final):
            lines.append(f"End-of-time value: {float(final):.3f}.")
        lines.append("Meaning: Proxy timeline: metals rise, then chemistry + stellar windows gate life potential.")
        lines.append(
            "Value impact: If the curve stays near zero, either metals never accumulate (weak star formation), "
            "or chemistry/thermo windows fail. A broad peak implies a longer ‘complexity window’ for molecules and life."
        )

    elif "molecular bonding viability" in t_low:
        best_name = extras.get("best_name")
        best_val = extras.get("best_val")
        if best_name is not None and best_val is not None:
            lines.append(f"Best-performing family: {best_name} at {float(best_val):.3f}.")
        lines.append("Meaning: A coarse family-level proxy for