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
st.title("Multiverse Simulator")

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

# Stores per-image PDF titles + value-dependent explanations
PLOT_PDF_META = {}  # filename -> {"title": str, "text": str}

def _latin1(s: str) -> str:
    return (s or "").encode("latin-1", "replace").decode("latin-1")

def register_pdf_plot(filename: str, title: str, text: str):
    PLOT_PDF_META[filename] = {"title": title.strip(), "text": (text or "").strip()}

st.sidebar.header("Controls")

disable_3d = st.sidebar.checkbox(
    "Disable 3D for iPhone compatibility",
    value=False
)

auto_save_plots = st.sidebar.checkbox(
    "Auto-save plots for PDF",
    value=True
)

res_3d = st.sidebar.slider(
    "3D resolution",
    min_value=25, max_value=120, value=60, step=5
)
st.sidebar.caption("Lower resolution renders faster on iPhone.")

def save_plot(fig, filename, is_plotly=True):
    """Never crash the app if export fails."""
    if not auto_save_plots:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    try:
        if is_plotly:
            # Plotly → PNG via kaleido
            pio.write_image(fig, path, format="png")
        else:
            # Matplotlib → PNG
            plt.savefig(path, bbox_inches="tight", dpi=300)
            plt.close()
    except Exception as e:
        st.warning(f"Plot save skipped for '{filename}': {e}")
        try:
            if not is_plotly:
                plt.close()
        except Exception:
            pass


def slider_with_input(label, min_val, max_val, default_val, step):
    col1, col2 = st.sidebar.columns([3, 1])
    slider_val = col1.slider(label, min_val, max_val, default_val, step=step)
    user_input = col2.text_input(f"{label}", value=str(slider_val), label_visibility="collapsed")
    try:
        input_val = float(user_input)
        if min_val <= input_val <= max_val:
            slider_val = input_val
    except Exception:
        pass

    percent_change = ((slider_val - 1.0) / 1.0) * 100
    st.sidebar.caption(f"{label}: {percent_change:+.2f}% vs baseline")
    return float(slider_val)


def safe_norm(arr):
    m = np.nanmax(arr)
    if m <= 0 or not np.isfinite(m):
        return np.zeros_like(arr)
    return np.clip(arr / m, 0, 1)


def generate_pdf(constants, summary_text, output_dir=OUTPUT_DIR):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    font = "Helvetica"

    # Cover
    pdf.add_page()
    pdf.set_font(font, "B", 22)
    pdf.cell(0, 12, "Multiverse Simulation Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font(font, "", 12)
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 8, f"Date: {date_str}", ln=True, align="C")
    pdf.cell(0, 8, "Generated via Multiverse Simulator", ln=True, align="C")

    # Params
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Parameters", ln=True)
    pdf.set_font(font, "", 12)
    for k, v in constants.items():
        line = f"{k}: {v:.4f}"
        pdf.cell(0, 7, _latin1(line), ln=True)

    # AI Summary
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Universe Summary", ln=True)
    pdf.set_font(font, "", 12)
    for line in (summary_text or "").split("\n"):
        pdf.multi_cell(0, 7, _latin1(line))

    # Visuals
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Visuals", ln=True)

    image_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".png")])
    for image_file in image_files:
        path = os.path.join(output_dir, image_file)
        meta = PLOT_PDF_META.get(image_file, None)

        pdf.add_page()
        pdf.set_font(font, "B", 12)

        if meta:
            title = meta["title"]
        else:
            title = image_file.replace(".png", "").replace("_", " ")

        pdf.cell(0, 8, _latin1(title), ln=True)

        try:
            pdf.image(path, w=180)
        except Exception:
            pdf.set_font(font, "", 10)
            pdf.multi_cell(0, 6, _latin1(f"Could not embed image: {image_file}"))
            continue

        # Explanation below the graph
        expl = meta["text"] if meta else ""
        if expl:
            pdf.ln(3)
            pdf.set_font(font, "", 11)
            pdf.multi_cell(0, 6, _latin1(expl))

    outname = "Universe_Simulation_Report.pdf"
    pdf.output(outname)
    return outname


# =========================
# Plot Helpers
# =========================
def show_surface_or_heatmap(z, x, y, title, xlab, ylab, zlab, fname_base, colorscale="Viridis",
                            pdf_title=None, pdf_text=None):
    """
    z must be shape (len(y), len(x)) if you pass x as x-axis and y as y-axis.
    Uses stable camera + margins to reduce iPhone blank WebGL risk.
    """
    z = np.asarray(z, dtype=float)

    if z.shape != (len(y), len(x)):
        st.error(f"Shape mismatch in {title}: z{z.shape} must be ({len(y)}, {len(x)})")
        return

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=zlab)
        )])
        fig3d.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlab,
                yaxis_title=ylab,
                zaxis_title=zlab,
                aspectmode="auto",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)
        fname = f"{fname_base}_3D.png"
        save_plot(fig3d, fname, is_plotly=True)
    else:
        fig2d = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=zlab)
        ))
        fig2d.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
        st.plotly_chart(fig2d, use_container_width=True)
        fname = f"{fname_base}_2D.png"
        save_plot(fig2d, fname, is_plotly=True)

    # Register PDF metadata
    if pdf_title or pdf_text:
        register_pdf_plot(fname, pdf_title or title, pdf_text or "")


def show_ribbon_or_line(x, y, title, xlab, ylab, fname_base, ribbon_width=1.2,
                        pdf_title=None, pdf_text=None):
    """
    3D-first: shows a thin surface ribbon for reliability/consistency.
    Falls back to 2D line if 3D disabled.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not disable_3d:
        yy = np.array([0.0, ribbon_width])
        X2, Y2 = np.meshgrid(x, yy, indexing="xy")
        Z2 = np.vstack([y, y])

        fig3d = go.Figure(data=[go.Surface(
            x=X2,
            y=Y2,
            z=Z2,
            colorscale="Viridis",
            showscale=False
        )])
        fig3d.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlab,
                yaxis_title="",
                zaxis_title=ylab,
                aspectmode="auto",
                camera=dict(eye=dict(x=1.7, y=1.4, z=1.0))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)
        fname = f"{fname_base}_3D.png"
        save_plot(fig3d, fname, is_plotly=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
        fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
        st.plotly_chart(fig, use_container_width=True)
        fname = f"{fname_base}_2D.png"
        save_plot(fig, fname, is_plotly=True)

    if pdf_title or pdf_text:
        register_pdf_plot(fname, pdf_title or title, pdf_text or "")


# =========================
# Constants
# =========================
constants = {
    "Strong Force Multiplier": slider_with_input("Strong Force", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_input("Electromagnetic EM", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": slider_with_input("Weak Force", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": slider_with_input("Gravity", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": slider_with_input("Dark Energy", 0.1, 10.0, 1.0, 0.01),
    "Temperature Multiplier": slider_with_input("Temperature", 0.1, 10.0, 1.0, 0.01),
    "Pressure Multiplier": slider_with_input("Pressure", 0.1, 10.0, 1.0, 0.01),
}

S = constants["Strong Force Multiplier"]
EM = constants["Electromagnetic Force Multiplier"]
W = constants["Weak Force Multiplier"]
G = constants["Gravitational Constant Multiplier"]
DE = constants["Dark Energy Multiplier"]
T = constants["Temperature Multiplier"]
P = constants["Pressure Multiplier"]

deviation = sum(abs(v - 1.0) for v in constants.values())

st.header("Universe Stability Summary")
st.write(f"Deviation from baseline constants: **{deviation:.3f}**")
if deviation < 1.5:
    st.success("Low deviation: broad stability expected.")
elif deviation < 4.0:
    st.warning("Moderate deviation: some instabilities likely.")
else:
    st.error("High deviation: instability likely across physics and chemistry.")

st.divider()


# =========================
# Shared nuclear grids
# =========================
Z = np.arange(1, 121)                # proton number
N = np.arange(1, 181)                # neutron number
Zg, Ng = np.meshgrid(Z, N, indexing="ij")
Ag = Zg + Ng

Nz_target = 1.0 + (Zg / 80.0)
ratio = Ng / np.maximum(Zg, 1)

ratio_penalty = np.exp(-((ratio - Nz_target) ** 2) * 8.0)
strong_bonus = np.exp(-np.abs(Zg - 82) / (28.0 * max(S, 1e-3)))
em_penalty = np.exp(-np.maximum(Zg - 20, 0) / (30.0 / max(EM, 1e-3)))
weak_opt = np.exp(-((W - 1.0) ** 2) * 2.5)

nuclear_stability = np.clip(ratio_penalty * strong_bonus * em_penalty * weak_opt, 0, 1)
isotope_viable_per_Z = (nuclear_stability > 0.20).sum(axis=1)

# Some global summary metrics used in value-dependent text
mean_nuclear_stability = float(np.nanmean(nuclear_stability))
viability_score = float(np.exp(-0.35 * deviation))
instability_score = float(1.0 - viability_score)


# =========================
# AI Summary
# =========================
st.subheader("AI Global Universe Analysis")
ai_ok = ("OPENAI_API_KEY" in st.secrets) and (openai is not None)

if not ai_ok:
    st.info("OpenAI not configured. Add OPENAI_API_KEY to .streamlit/secrets.toml to enable.")
else:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if st.button("Generate AI Universe Summary"):
        with st.spinner("Generating summary..."):
            user_context = "\n".join([f"{k}: {v:.4f}" for k, v in constants.items()])
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a physics and cosmology expert. Provide a rigorous but accessible analysis of a universe defined by these dimensionless multipliers. Discuss nuclear stability, chemistry, star formation, habitability, and long-term evolution. Avoid fluff."},
                        {"role": "user", "content": f"Constants:\n{user_context}"}
                    ],
                    max_tokens=700,
                    temperature=0.5
                )
                summary = resp.choices[0].message.content
                st.session_state["summary"] = summary
                st.success("Summary generated")
                st.markdown(summary)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

st.divider()


# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Periodic Table Stability",
    "Island of Instability",
    "Star Formation",
    "Life Probability",
    "Quantum Bonding",
    "Universe Viability",
    "Element Abundance",
    "Radiation Risk",
    "Star Lifespan",
    "Cosmic Web",
    "Atomic Stability",
    "Life Over Time",
    "Molecular Bonding",
    "Molecular Abundance",
    "Isotope Half-Life",
    "Periodic Table Expansion",
    "Proton–Neutron Map",
    "Binding Energy",
    "Decoherence Map",
    "Branch Count",
    "Quantum Gravity Horizon",
])
# -------------------------
# Tab 0: Periodic Table Stability
# -------------------------
with tabs[0]:
    st.subheader("Periodic Table Stability")

    Z_vals = np.arange(1, 121)

    # Shared base shell structure (purely proxy)
    Z2_base = Z_vals[:, None]
    base_shell = np.exp(-np.abs(Z2_base - 30) / 22.0)

    # Helper to build the stability map for a chosen sweep axis
    def stability_map(Z_vals, sweep_vals, sweep_name):
        Z2, X2 = np.meshgrid(Z_vals, sweep_vals, indexing="ij")

        # Use the swept variable as the one on the x-axis, others fixed at current sliders
        if sweep_name == "EM":
            EM_use = np.maximum(X2, 1e-3)
            S_use = max(S, 1e-3)
            W_use = W
        elif sweep_name == "S":
            EM_use = max(EM, 1e-3)
            S_use = np.maximum(X2, 1e-3)
            W_use = W
        elif sweep_name == "W":
            EM_use = max(EM, 1e-3)
            S_use = max(S, 1e-3)
            W_use = X2
        else:
            raise ValueError("Unknown sweep")

        # Proxy terms
        strong_term = np.exp(-np.abs(Z2 - 82) / (28.0 * S_use))
        em_term = np.exp(-np.maximum(Z2 - 20, 0) / (40.0 / EM_use))
        weak_term = np.exp(-((W_use - 1.0) ** 2) * 2.5)

        stab = np.clip((0.55 + 0.45*np.exp(-np.abs(Z2 - 30) / 22.0)) * strong_term * em_term * weak_term, 0, 1)
        return stab

    # 1) Z vs EM
    em_sweep = np.linspace(0.1, 10.0, res_3d)
    stab_em = stability_map(Z_vals, em_sweep, "EM")
    show_surface_or_heatmap(
        z=stab_em,
        x=em_sweep,
        y=Z_vals,
        title="Stability vs EM",
        xlab="EM Multiplier",
        ylab="Atomic Number Z",
        zlab="Stability",
        fname_base="PT_Stability_EM",
        colorscale="Viridis"
    )
    st.markdown(
        f"- Current universe: Strong={S:.2f}, Weak={W:.2f}\n"
        f"- If EM increases, heavy nuclei lose stability faster from stronger Coulomb repulsion."
    )

    st.divider()

    # 2) Z vs Strong
    s_sweep = np.linspace(0.1, 10.0, res_3d)
    stab_s = stability_map(Z_vals, s_sweep, "S")
    show_surface_or_heatmap(
        z=stab_s,
        x=s_sweep,
        y=Z_vals,
        title="Stability vs Strong",
        xlab="Strong Multiplier",
        ylab="Atomic Number Z",
        zlab="Stability",
        fname_base="PT_Stability_Strong",
        colorscale="Viridis"
    )
    st.markdown(
        f"- Current universe: EM={EM:.2f}, Weak={W:.2f}\n"
        f"- Higher Strong generally boosts binding and pushes the stable region toward higher Z."
    )

    st.divider()

    # 3) Z vs Weak
    w_sweep = np.linspace(0.1, 10.0, res_3d)
    stab_w = stability_map(Z_vals, w_sweep, "W")
    show_surface_or_heatmap(
        z=stab_w,
        x=w_sweep,
        y=Z_vals,
        title="Stability vs Weak",
        xlab="Weak Multiplier",
        ylab="Atomic Number Z",
        zlab="Stability",
        fname_base="PT_Stability_Weak",
        colorscale="Viridis"
    )
    st.markdown(
        f"- Current universe: Strong={S:.2f}, EM={EM:.2f}\n"
        f"- Weak sets beta-decay pacing in this proxy: far from 1.0 reduces long-lived isotopes."
    )
# -------------------------
# Tab 1: Island of Instability
# -------------------------
with tabs[1]:
    st.subheader("Island of Instability")

    Z_vals = np.linspace(40, 140, res_3d)
    S_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    Z2, S2 = np.meshgrid(Z_vals, S_vals, indexing="ij")

    shell_waves = (0.55 + 0.45*np.sin(Z2 / 7.0)**2) * (0.55 + 0.45*np.sin(Z2 / 11.0)**2)
    strong_opt = np.exp(-np.abs(S2 - S) / 1.6)
    coulomb = np.exp(-np.maximum(Z2 - 60, 0) / (25.0 / max(EM, 1e-3)))

    instability = np.clip((1 - strong_opt) * shell_waves * (1 / np.maximum(coulomb, 1e-6)), 0, 2.0)

    inst_mean = float(np.nanmean(instability))
    inst_peak = float(np.nanmax(instability))

    show_surface_or_heatmap(
        z=instability,
        x=S_vals,
        y=Z_vals,
        title="Instability vs Z and Strong",
        xlab="Strong Multiplier",
        ylab="Atomic Number Z",
        zlab="Instability",
        fname_base="Island_Instability",
        colorscale="Inferno",
        pdf_title="Island of Instability",
        pdf_text=(
            f"This surface highlights where heavy nuclei become fragile. "
            f"At Strong={S:.2f} and EM={EM:.2f}, mean instability is {inst_mean:.2f} "
            f"with a peak near {inst_peak:.2f}. Higher EM steepens the high-Z penalty; "
            f"higher Strong can partially offset it, shifting the unstable ridge upward in Z."
        )
    )

    st.markdown(
        "- High Z is punished by charge repulsion.\n"
        "- Stronger nuclear cohesion reduces the unstable ridge and widens the viable region."
    )


# -------------------------
# Tab 2: Star Formation
# -------------------------
with tabs[2]:
    st.subheader("Star Formation")

    g_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    de_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    G2, DE2 = np.meshgrid(g_vals, de_vals, indexing="ij")

    collapse = (G2**1.1) / (DE2**1.2 + 0.15)
    rad_pressure = 1 / (1 + (EM**1.3))
    ignition = np.exp(-np.abs(S - 1.0) * 0.8) * np.exp(-((W - 1.0) ** 2) * 1.2)

    sfr = safe_norm(collapse * rad_pressure * ignition)

    sfr_mean = float(np.nanmean(sfr))
    sfr_here = float(np.tanh(((G**1.1) / (DE**1.2 + 0.15)) * (1 / (1 + EM**1.3)) / 2.0))

    show_surface_or_heatmap(
        z=sfr,
        x=de_vals,
        y=g_vals,
        title="Star Formation Potential",
        xlab="Dark Energy Multiplier",
        ylab="Gravity Multiplier",
        zlab="Potential",
        fname_base="Star_Formation",
        colorscale="Viridis",
        pdf_title="Star Formation Potential",
        pdf_text=(
            f"Star formation rises with gravitational collapse and falls as dark energy suppresses structure. "
            f"With Gravity={G:.2f} and Dark Energy={DE:.2f}, the local collapse index is {sfr_here:.2f}. "
            f"The surface mean is {sfr_mean:.2f}. EM={EM:.2f} increases radiation pressure and can reduce collapse efficiency."
        )
    )

    st.markdown(
        "- Higher gravity promotes collapse.\n"
        "- Higher dark energy suppresses large-scale structure and slows star formation."
    )


# -------------------------
# Tab 3: Life Probability
# -------------------------
with tabs[3]:
    st.subheader("Life Probability")

    s_vals = np.linspace(0.1, 10.0, res_3d)
    em_vals = np.linspace(0.1, 10.0, res_3d)
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    force_window = (
        np.exp(-((S2 - 1.0) ** 2) / 1.6)
        * np.exp(-((EM2 - 1.0) ** 2) / 1.6)
        * np.exp(-((W - 1.0) ** 2) / 2.0)
    )
    thermo = np.exp(-((T - 1.0) ** 2) * 1.2) * np.exp(-((P - 1.0) ** 2) * 1.0)

    metals = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0) * 0.6), 0, 5)
    metals_factor = np.tanh(metals / 1.5)

    life = np.clip(force_window * thermo * metals_factor, 0, 1)

    life_mean = float(np.nanmean(life))
    life_peak = float(np.nanmax(life))

    show_surface_or_heatmap(
        z=life,
        x=em_vals,
        y=s_vals,
        title="Life Probability Map",
        xlab="EM Multiplier",
        ylab="Strong Multiplier",
        zlab="Life Score",
        fname_base="Life_Probability",
        colorscale="Plasma",
        pdf_title="Life Probability Map",
        pdf_text=(
            f"This score combines chemistry, environment, and element availability. "
            f"With Temperature={T:.2f} and Pressure={P:.2f}, the thermal gate is {thermo:.2f}. "
            f"Element availability is driven by Gravity={G:.2f} and Dark Energy={DE:.2f}, giving a metallicity factor {metals_factor:.2f}. "
            f"The map mean is {life_mean:.2f} and the peak is {life_peak:.2f}. "
            f"Large shifts in Strong={S:.2f} or EM={EM:.2f} narrow the viable chemistry window."
        )
    )

    st.markdown(
        "- This is a comparative life score, not a literal probability.\n"
        "- It rises when chemistry, temperature, pressure, and element supply align."
    )


# -------------------------
# Tab 4: Quantum Bonding
# -------------------------
with tabs[4]:
    st.subheader("Quantum Bonding")

    s_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    em_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    em_binding = np.exp(-((EM2 - 1.0) ** 2) / 1.4)
    strong_chem = np.exp(-abs(S2 - 1.0) / 2.0)
    temp_kill = np.exp(-((T - 1.0) ** 2) * 1.8)
    pressure_help = np.tanh(P / 1.5)

    bonding = np.clip(em_binding * strong_chem * temp_kill * pressure_help, 0, 1)

    bond_mean = float(np.nanmean(bonding))
    bond_peak = float(np.nanmax(bonding))

    show_surface_or_heatmap(
        z=bonding,
        x=em_vals,
        y=s_vals,
        title="Bonding Strength",
        xlab="EM Multiplier",
        ylab="Strong Multiplier",
        zlab="Bonding",
        fname_base="Quantum_Bonding",
        colorscale="Viridis",
        pdf_title="Bonding Strength",
        pdf_text=(
            f"Bonding is strongest near EM≈1 and Strong≈1, then suppressed by heat and helped by pressure. "
            f"With EM={EM:.2f}, Strong={S:.2f}, Temperature={T:.2f}, Pressure={P:.2f}, the bonding field mean is {bond_mean:.2f} "
            f"and the peak is {bond_peak:.2f}. Higher temperature reduces bonding sharply when T drifts far from 1."
        )
    )

    st.markdown(
        "- Bonding is strongest near baseline forces.\n"
        "- Higher temperature weakens stable molecules; pressure partially restores overlap."
    )


# -------------------------
# Tab 5: Universe Viability
# -------------------------
with tabs[5]:
    st.subheader("Universe Viability")

    viability = viability_score
    chaos = instability_score

    if not disable_3d:
        x = np.array([0, 1])
        y = np.array([0, 0.6])
        Zs = np.array([
            [viability, chaos],
            [viability, chaos]
        ])

        fig = go.Figure(data=[go.Surface(
            x=x, y=y, z=Zs,
            colorscale="Viridis",
            colorbar=dict(title="Score")
        )])
        fig.update_layout(
            title="Global Viability",
            scene=dict(
                xaxis=dict(title="", tickmode="array", tickvals=[0, 1], ticktext=["Viability", "Instability"]),
                yaxis_title="",
                zaxis_title="Score",
                aspectmode="auto",
                camera=dict(eye=dict(x=1.8, y=1.5, z=1.0))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        fname = "Universe_Viability_3D.png"
        save_plot(fig, fname, is_plotly=True)
    else:
        fig2d = go.Figure(data=[go.Bar(x=["Viability", "Instability"], y=[viability, chaos])])
        fig2d.update_layout(title="Global Viability", yaxis_title="Score")
        st.plotly_chart(fig2d, use_container_width=True)
        fname = "Universe_Viability_2D.png"
        save_plot(fig2d, fname, is_plotly=True)

    register_pdf_plot(
        fname,
        "Global Viability",
        (
            f"Viability is a compact stability score based on total deviation from baseline constants. "
            f"Deviation={deviation:.3f} gives Viability={viability:.2f} and Instability={chaos:.2f}. "
            f"Lower deviation generally supports stable nuclei, chemistry, and long-lived stars."
        )
    )

    st.markdown(
        "- Viability is based on distance from baseline constants.\n"
        "- It is a compact comparative score, not a literal emergence probability."
    )


# -------------------------
# Tab 6: Element Abundance
# -------------------------
with tabs[6]:
    st.subheader("Element Abundance")

    surv = safe_norm(isotope_viable_per_Z.astype(float))
    stellar = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0) * 0.4) * np.exp(-abs(S - 1.0) * 0.3), 0, 3)
    stellar_factor = np.tanh(stellar / 1.3)

    abundance = np.clip(surv * stellar_factor, 0, 1)
    abund_mean = float(np.nanmean(abundance))
    abund_peak = float(np.nanmax(abundance))

    show_ribbon_or_line(
        x=Z,
        y=abundance,
        title="Element Abundance vs Z",
        xlab="Atomic Number Z",
        ylab="Relative Abundance",
        fname_base="Element_Abundance_Line",
        ribbon_width=1.5,
        pdf_title="Element Abundance vs Z",
        pdf_text=(
            f"Abundance combines isotope survivability with star processing. "
            f"Mean abundance is {abund_mean:.2f} with a peak of {abund_peak:.2f}. "
            f"Gravity={G:.2f} and Dark Energy={DE:.2f} set enrichment pace via stellar factor {stellar_factor:.2f}. "
            f"Large EM shifts can reduce heavy-element survival and production."
        )
    )

    # Evolution surface
    t = np.linspace(0, 1, max(30, res_3d // 2))
    Z_ds = Z[::2]
    abundance_ds = abundance[::2]

    Z2, T2 = np.meshgrid(Z_ds, t, indexing="ij")
    enrich = 1 - np.exp(-T2 * (2.0 + 4.0 * stellar_factor))
    abund_time = np.clip((abundance_ds[:, None]) * enrich, 0, 1)

    show_surface_or_heatmap(
        z=abund_time,
        x=t,
        y=Z_ds,
        title="Abundance Over Time",
        xlab="Cosmic Time",
        ylab="Atomic Number Z",
        zlab="Abundance",
        fname_base="Element_Abundance_Evolution",
        colorscale="Cividis",
        pdf_title="Abundance Over Time",
        pdf_text=(
            f"This surface shows enrichment from early to late cosmic time. "
            f"With stellar factor {stellar_factor:.2f}, enrichment rises quickly when gravity dominates dark energy. "
            f"If DE={DE:.2f} is high, enrichment saturates later and the heavy-element tail grows more slowly."
        )
    )

    st.markdown(
        "- Abundance rises when nuclei can survive and stars can form efficiently.\n"
        "- Faster enrichment supports complex chemistry by supplying heavier elements."
    )


# -------------------------
# Tab 7: Radiation Risk
# -------------------------
with tabs[7]:
    st.subheader("Radiation Risk")

    x = np.linspace(0.1, 10.0, 600)
    y = (x ** 2) * (0.4 + 0.6 * np.tanh(T / 2.0)) / 20.0
    y = np.clip(y, 0, 1)

    risk_here = float(np.clip((EM ** 2) * (0.4 + 0.6 * np.tanh(T / 2.0)) / 20.0, 0, 1))

    show_ribbon_or_line(
        x=x, y=y,
        title="Radiation Risk vs EM",
        xlab="EM Multiplier",
        ylab="Risk",
        fname_base="EM_Radiation_Risk",
        ribbon_width=1.2,
        pdf_title="Radiation Risk vs EM",
        pdf_text=(
            f"Radiation risk increases with EM and is amplified by temperature. "
            f"With EM={EM:.2f} and Temperature={T:.2f}, the local risk index is {risk_here:.2f}. "
            f"Higher EM can intensify radiative effects and reduce habitability margins."
        )
    )


# -------------------------
# Tab 8: Star Lifespan
# -------------------------
with tabs[8]:
    st.subheader("Star Lifespan")

    g_vals = np.linspace(0.1, 10.0, 400)
    M = g_vals
    L = (M ** 3.5) * (1.0 + 0.15 * (EM - 1.0))
    tau = (M / np.maximum(L, 1e-9))
    tau = safe_norm(tau)

    tau_here = float(np.clip((G / max((G ** 3.5) * (1.0 + 0.15 * (EM - 1.0)), 1e-9)), 0, 1))

    show_ribbon_or_line(
        x=g_vals, y=tau,
        title="Star Lifespan vs Gravity",
        xlab="Gravity Multiplier",
        ylab="Relative Lifespan",
        fname_base="Star_Lifespan",
        ribbon_width=1.2,
        pdf_title="Star Lifespan vs Gravity",
        pdf_text=(
            f"Higher gravity trends toward higher stellar luminosity and shorter lifetimes. "
            f"With Gravity={G:.2f} and EM={EM:.2f}, the local lifespan index is {tau_here:.2f}. "
            f"Short lifetimes reduce the time available for complex chemistry and biology to develop."
        )
    )


# -------------------------
# Tab 9: Cosmic Web
# -------------------------
with tabs[9]:
    st.subheader("Cosmic Web")

    size = 32
    scale = 8.0

    cluster_spread = (3.5 / max(G, 1e-3))
    stretch = (1.0 / max(DE, 1e-3))

    x = np.linspace(-scale * stretch, scale * stretch, size)
    y = np.linspace(-scale * stretch, scale * stretch, size)
    z = np.linspace(-scale * stretch, scale * stretch, size)
    X, Y, Z3 = np.meshgrid(x, y, z, indexing="ij")

    density = np.zeros_like(X)
    rng = np.random.default_rng(42)
    centers = rng.uniform(-scale, scale, size=(5, 3))

    for cx, cy, cz in centers:
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z3 - cz) ** 2
        density += np.exp(-r2 / (cluster_spread ** 2))

    density = safe_norm(density)

    mid = size // 2
    slice2d = density[:, :, mid]

    fig2d = go.Figure(data=go.Heatmap(
        z=slice2d.T, x=x, y=y,
        colorscale="Inferno",
        colorbar=dict(title="Density")
    ))
    fig2d.update_layout(title="Cosmic Web Slice", xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Dark_Matter_2D_Slice.png", is_plotly=True)

    register_pdf_plot(
        "Dark_Matter_2D_Slice.png",
        "Cosmic Web Slice",
        (
            f"This slice shows how structure clusters under gravity and stretches under dark energy. "
            f"Gravity={G:.2f} tightens clusters, Dark Energy={DE:.2f} expands the scale and thins filaments."
        )
    )

    if not disable_3d:
        thr = 0.25
        pts = np.where(density > thr)
        Xp, Yp, Zp = X[pts], Y[pts], Z3[pts]
        Cp = density[pts]

        fig3d = go.Figure(data=[go.Scatter3d(
            x=Xp, y=Yp, z=Zp,
            mode="markers",
            marker=dict(size=2, color=Cp, colorscale="Inferno", opacity=0.7)
        )])
        fig3d.update_layout(
            title="Cosmic Web",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Dark_Matter_3D.png", is_plotly=True)

        register_pdf_plot(
            "Dark_Matter_3D.png",
            "Cosmic Web",
            (
                f"3D structure appears where density exceeds a threshold. "
                f"With Gravity={G:.2f} and Dark Energy={DE:.2f}, clustering is "
                f"{'strong' if G > 1.2 and DE < 1.2 else 'weaker'} and filaments are "
                f"{'more compact' if DE < 1.0 else 'more stretched'}."
            )
        )

    st.markdown(
        "- Higher gravity compresses structure.\n"
        "- Higher dark energy expands voids and reduces clustering."
    )


# -------------------------
# Tab 10: Atomic Stability
# -------------------------
with tabs[10]:
    st.subheader("Atomic Stability")

    Zs = Z[::3]
    Ns = N[::3]
    stab_ds = nuclear_stability[::3, ::3]

    show_surface_or_heatmap(
        z=stab_ds,
        x=Ns,
        y=Zs,
        title="Isotope Stability Map",
        xlab="Neutrons N",
        ylab="Protons Z",
        zlab="Stability",
        fname_base="Atomic_Stability",
        colorscale="Plasma",
        pdf_title="Isotope Stability Map",
        pdf_text=(
            f"This map visualizes the stability band across neutron and proton counts. "
            f"Mean nuclear stability is {mean_nuclear_stability:.2f}. "
            f"Strong={S:.2f} increases binding, EM={EM:.2f} penalizes high Z, and Weak={W:.2f} shapes beta decay routes."
        )
    )

    st.markdown(
        "- Stability follows a valley where neutron and proton counts balance binding and decay.\n"
        "- Strong boosts binding, EM penalizes high charge, weak controls beta decay pressure."
    )


# -------------------------
# Tab 11: Life Over Time
# -------------------------
with tabs[11]:
    st.subheader("Life Over Time")

    time = np.linspace(0, 1, 200)

    sfr_now = np.clip((G**1.1) / (DE**1.2 + 0.15) * (1 / (1 + EM**1.3)), 0, 10)
    sfr_now = float(np.tanh(sfr_now / 2.0))

    metals = 1 - np.exp(-(2.0 + 5.0 * sfr_now) * time)
    metals = np.clip(metals, 0, 1)

    star_window = float(np.exp(-abs(G - 1.0) * 0.6))
    chem_window = float(np.exp(-abs(S - 1.0) * 0.4) * np.exp(-abs(EM - 1.0) * 0.4) * np.exp(-((W - 1.0) ** 2) * 0.4))
    thermo_window = float(np.exp(-((T - 1.0) ** 2) * 0.6) * np.exp(-((P - 1.0) ** 2) * 0.4))

    life_t = np.clip(metals * star_window * chem_window * thermo_window, 0, 1)

    show_ribbon_or_line(
        x=time,
        y=life_t,
        title="Life Score Over Time",
        xlab="Cosmic Time",
        ylab="Life Score",
        fname_base="Life_Over_Time",
        ribbon_width=1.6,
        pdf_title="Life Score Over Time",
        pdf_text=(
            f"Life score rises as metals accumulate, then saturates. "
            f"With current parameters, star window={star_window:.2f}, chemistry window={chem_window:.2f}, thermal window={thermo_window:.2f}. "
            f"Faster star formation increases early enrichment; high dark energy delays it."
        )
    )

    show_ribbon_or_line(
        x=time,
        y=metals,
        title="Metallicity Over Time",
        xlab="Cosmic Time",
        ylab="Metallicity",
        fname_base="Metallicity_Over_Time",
        ribbon_width=1.2,
        pdf_title="Metallicity Over Time",
        pdf_text=(
            f"Metallicity tracks heavy-element buildup from stellar processing. "
            f"With Gravity={G:.2f}, Dark Energy={DE:.2f}, and EM={EM:.2f}, the enrichment rate is {sfr_now:.2f}. "
            f"Higher enrichment supports more complex chemistry and larger molecular diversity."
        )
    )


# -------------------------
# Tab 12: Molecular Bonding
# -------------------------
with tabs[12]:
    st.subheader("Molecular Bonding")

    isotope_factor = float(np.mean(isotope_viable_per_Z) / np.max(isotope_viable_per_Z))
    isotope_factor = np.clip(isotope_factor, 0, 1)

    families = {
        "Simple Covalent": 0.95,
        "Polar": 0.90,
        "Carbon Backbone": 0.88,
        "Metallic": 0.75,
        "Heavy Chemistry": 0.60,
    }

    em_mod = np.exp(-abs(EM - 1.0) * 0.7)
    strong_mod = np.exp(-abs(S - 1.0) * 0.6)
    weak_mod = np.exp(-((W - 1.0) ** 2) * 0.4)
    temp_mod = np.exp(-((T - 1.0) ** 2) * 0.9)
    press_mod = np.tanh(P / 1.6)

    global_mod = float(np.clip(em_mod * strong_mod * weak_mod * temp_mod * press_mod * (0.6 + 0.4 * isotope_factor), 0, 1))

    names = list(families.keys())
    vals = [float(np.clip(families[k] * global_mod, 0, 1)) for k in names]

    fig = go.Figure(data=[go.Bar(x=names, y=vals, text=[f"{v:.2f}" for v in vals], textposition="outside")])
    fig.update_layout(
        title="Molecular Bonding Viability",
        yaxis_title="Viability",
        yaxis_range=[0, 1.15]
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Molecular_Bonding.png", is_plotly=True)

    best_idx = int(np.argmax(vals))
    register_pdf_plot(
        "Molecular_Bonding.png",
        "Molecular Bonding Viability",
        (
            f"Bonding viability scales with EM, Strong, temperature, pressure, and isotope availability. "
            f"Global bonding modifier is {global_mod:.2f} with isotope factor {isotope_factor:.2f}. "
            f"Highest family score is {names[best_idx]} at {vals[best_idx]:.2f}. "
            f"High temperature or large EM and Strong shifts reduce stable molecular diversity."
        )
    )


# -------------------------
# Tab 13: Molecular Abundance
# -------------------------
with tabs[13]:
    st.subheader("Molecular Abundance")

    tvals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    pvals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    T2, P2 = np.meshgrid(tvals, pvals, indexing="ij")

    force_gate = float(np.exp(-abs(S - 1.0) * 0.4) * np.exp(-abs(EM - 1.0) * 0.4))
    thermo_gate = np.exp(-((T2 - 1.0) ** 2) * 0.8) * np.exp(-((P2 - 1.0) ** 2) * 0.6)
    abundance = np.clip(force_gate * thermo_gate, 0, 1)

    abundance_mean = float(np.nanmean(abundance))
    abundance_peak = float(np.nanmax(abundance))

    show_surface_or_heatmap(
        z=abundance,
        x=pvals,
        y=tvals,
        title="Molecular Abundance Map",
        xlab="Pressure Multiplier",
        ylab="Temperature Multiplier",
        zlab="Abundance",
        fname_base="Molecular_Abundance",
        colorscale="Viridis",
        pdf_title="Molecular Abundance Map",
        pdf_text=(
            f"Molecular abundance rises when forces support bonding and conditions sit near stable temperature and pressure. "
            f"Force gate is {force_gate:.2f} from Strong={S:.2f} and EM={EM:.2f}. "
            f"Surface mean is {abundance_mean:.2f} and peak is {abundance_peak:.2f}. "
            f"Shifting temperature or pressure away from 1 compresses the viable region."
        )
    )


# -------------------------
# Tab 14: Isotope Half-Life
# -------------------------
with tabs[14]:
    st.subheader("Isotope Half-Life")

    half_life = np.clip(nuclear_stability * np.exp(-abs(W - 1.0) * 0.6), 0, 1)

    Zs = Z[::3]
    Ns = N[::3]
    hl_ds = half_life[::3, ::3]

    hl_mean = float(np.nanmean(half_life))
    long_lived = (half_life > 0.35).sum(axis=1).astype(float)
    long_lived_total = float(np.sum(long_lived))

    show_surface_or_heatmap(
        z=hl_ds,
        x=Ns,
        y=Zs,
        title="Half-Life Map",
        xlab="Neutrons N",
        ylab="Protons Z",
        zlab="Half-Life",
        fname_base="Half_Life",
        colorscale="Cividis",
        pdf_title="Half-Life Map",
        pdf_text=(
            f"Half-life score follows nuclear stability and weak-force tuning. "
            f"With Weak={W:.2f}, mean half-life score is {hl_mean:.2f}. "
            f"Long-lived isotopes support steady heat sources and long-term chemistry."
        )
    )

    show_ribbon_or_line(
        x=Z,
        y=long_lived,
        title="Long-Lived Isotopes per Z",
        xlab="Atomic Number Z",
        ylab="Count",
        fname_base="Half_Life_LongLived_Count",
        ribbon_width=1.2,
        pdf_title="Long-Lived Isotopes per Z",
        pdf_text=(
            f"This curve counts isotopes above a longevity threshold. "
            f"Total long-lived count across Z is {long_lived_total:.0f}. "
            f"Higher Strong and near-baseline Weak increase the pool of long-lived nuclei."
        )
    )


# -------------------------
# Tab 15: Periodic Table Expansion
# -------------------------
with tabs[15]:
    st.subheader("Periodic Table Expansion")

    Z_ext = np.arange(1, 201)
    cohesion = np.exp(-np.abs(Z_ext - 82) / (30.0 * max(S, 1e-3)))
    coulomb = 1 / (1 + np.exp(-(Z_ext - 110) / (12.0 / max(EM, 1e-3))))
    decay = np.exp(-((W - 1.0) ** 2) * 1.2)

    stability_curve = np.clip(cohesion * (1 - coulomb) * decay, 0, 1)
    maxZ = int(Z_ext[stability_curve > 0.12][-1]) if np.any(stability_curve > 0.12) else 0

    show_ribbon_or_line(
        x=Z_ext,
        y=stability_curve,
        title="Expansion Limit Curve",
        xlab="Atomic Number Z",
        ylab="Stability Potential",
        fname_base="Periodic_Table_Expansion_Curve",
        ribbon_width=1.2,
        pdf_title="Expansion Limit Curve",
        pdf_text=(
            f"This curve estimates how far the periodic table can extend before instability dominates. "
            f"With Strong={S:.2f}, EM={EM:.2f}, Weak={W:.2f}, the estimated stable limit is Z≈{maxZ}. "
            f"Stronger cohesion pushes the limit higher; higher EM pulls it lower."
        )
    )

    st.markdown(f"**Estimated stable limit Z: {maxZ}**")

    em_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    Z2, EM2 = np.meshgrid(Z_ext, em_vals, indexing="ij")
    cohesion2 = np.exp(-np.abs(Z2 - 82) / (30.0 * max(S, 1e-3)))
    coulomb2 = 1 / (1 + np.exp(-(Z2 - 110) / (12.0 / np.maximum(EM2, 1e-3))))
    stab2 = np.clip(cohesion2 * (1 - coulomb2) * decay, 0, 1)

    show_surface_or_heatmap(
        z=stab2,
        x=em_vals,
        y=Z_ext,
        title="Expansion vs Z and EM",
        xlab="EM Multiplier",
        ylab="Atomic Number Z",
        zlab="Stability",
        fname_base="Periodic_Table_Expansion_Map",
        colorscale="Viridis",
        pdf_title="Expansion vs Z and EM",
        pdf_text=(
            f"This surface shows how the expansion limit shifts as EM changes. "
            f"At EM={EM:.2f}, high Z stability drops quickly once Coulomb repulsion dominates. "
            f"If EM is reduced, the viable high-Z region expands and superheavy chemistry becomes more plausible."
        )
    )


# -------------------------
# Tab 16: Proton–Neutron Map
# -------------------------
with tabs[16]:
    st.subheader("Proton–Neutron Map")

    Zs = Z[::3]
    Ns = N[::3]
    vi_ds = nuclear_stability[::3, ::3]

    show_surface_or_heatmap(
        z=vi_ds,
        x=Ns,
        y=Zs,
        title="Viability in Z and N",
        xlab="Neutrons N",
        ylab="Protons Z",
        zlab="Viability",
        fname_base="PN_Viability",
        colorscale="Magma",
        pdf_title="Viability in Z and N",
        pdf_text=(
            f"This surface highlights the stable band in proton–neutron space. "
            f"Mean viability is {mean_nuclear_stability:.2f}. "
            f"Strong={S:.2f} widens the viable band, EM={EM:.2f} narrows it at high Z, and Weak={W:.2f} shifts decay balance."
        )
    )

    target_line = (1.0 + (Z / 80.0)) * Z
    show_ribbon_or_line(
        x=Z,
        y=target_line,
        title="Valley Target N",
        xlab="Protons Z",
        ylab="Target Neutrons N",
        fname_base="PN_Target_Curve",
        ribbon_width=1.2,
        pdf_title="Valley Target N",
        pdf_text=(
            "This curve is a simple valley target where stable isotopes tend to cluster. "
            "As Z increases, extra neutrons are required to dilute charge repulsion and sustain binding."
        )
    )


# -------------------------
# Tab 17: Binding Energy
# -------------------------
with tabs[17]:
    st.subheader("Binding Energy")

    a_v = 15.8 * S
    a_s = 18.3
    a_c = 0.714 * EM
    a_sym = 23.2 * (1 / max(W, 1e-3))
    a_pair = 12.0

    A = Ag
    pairing = np.where(((Zg % 2 == 0) & (Ng % 2 == 0)), +1, -1)

    BE = (
        a_v * A
        - a_s * (A ** (2 / 3))
        - a_c * (Zg * (Zg - 1)) / np.maximum(A ** (1 / 3), 1e-9)
        - a_sym * ((A - 2 * Zg) ** 2) / np.maximum(A, 1e-9)
        + pairing * a_pair / np.maximum(A ** 0.5, 1e-9)
    )

    BE_per_A = np.clip(BE / np.maximum(A, 1), 0, None)
    BE_per_A = np.clip(BE_per_A, 0, np.nanpercentile(BE_per_A, 99))

    Zs = Z[::3]
    Ns = N[::3]
    be_ds = BE_per_A[::3, ::3]

    be_mean = float(np.nanmean(BE_per_A))

    show_surface_or_heatmap(
        z=be_ds,
        x=Ns,
        y=Zs,
        title="Binding Energy per Nucleon",
        xlab="Neutrons N",
        ylab="Protons Z",
        zlab="BE per A",
        fname_base="Binding_Energy",
        colorscale="Viridis",
        pdf_title="Binding Energy per Nucleon",
        pdf_text=(
            f"Binding energy per nucleon indicates how tightly nuclei hold together. "
            f"Mean BE/A is {be_mean:.2f} in this grid. "
            f"Strong={S:.2f} raises the volume term, EM={EM:.2f} increases Coulomb penalty, and Weak={W:.2f} shifts symmetry pressure."
        )
    )

    target_line = (1.0 + (Z / 80.0)) * Z
    N_pick = np.clip(target_line.astype(int), 1, N[-1])
    idxN = N_pick - 1
    be_line = BE_per_A[np.arange(len(Z)), idxN]

    peakZ = int(Z[np.nanargmax(be_line)])
    peakBE = float(np.nanmax(be_line))

    show_ribbon_or_line(
        x=Z,
        y=be_line,
        title="Binding Energy Along Valley",
        xlab="Protons Z",
        ylab="BE per A",
        fname_base="Binding_Energy_Line",
        ribbon_width=1.2,
        pdf_title="Binding Energy Along Valley",
        pdf_text=(
            f"Along the valley target, BE/A peaks at Z≈{peakZ} with peak value {peakBE:.2f}. "
            f"This peak marks the most tightly bound region, which tends to dominate stable nucleosynthesis pathways."
        )
    )

    st.markdown(f"**Peak along valley near Z ≈ {peakZ}**")


# -------------------------
# Tab 18: Decoherence Map
# -------------------------
with tabs[18]:
    st.subheader("Decoherence Map")

    s_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    em_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    t_vals = np.linspace(0, 1, max(30, res_3d // 2))

    S2, EM2, TT = np.meshgrid(s_vals, em_vals, t_vals, indexing="ij")

    dist2 = (
        (S2 - S) ** 2 +
        (EM2 - EM) ** 2 +
        (W - 1.0) ** 2 +
        (G - 1.0) ** 2 +
        (DE - 1.0) ** 2
    )

    coherence = np.exp(-dist2 / 4.0) * np.exp(-TT * (0.8 + 0.6 * deviation / 4.0))
    mid = coherence[:, :, len(t_vals) // 2]

    coh_mean = float(np.nanmean(mid))
    coh_peak = float(np.nanmax(mid))

    show_surface_or_heatmap(
        z=mid,
        x=em_vals,
        y=s_vals,
        title="Coherence Slice",
        xlab="EM Multiplier",
        ylab="Strong Multiplier",
        zlab="Coherence",
        fname_base="Decoherence_MidTime",
        colorscale="Magma",
        pdf_title="Coherence Slice",
        pdf_text=(
            f"This slice shows how quickly coherence drops as constants drift from baseline. "
            f"Deviation={deviation:.3f} yields mean coherence {coh_mean:.2f} and peak {coh_peak:.2f}. "
            f"Larger deviation causes faster decay and a narrower region of high coherence."
        )
    )

    st.markdown(
        "- Coherence decreases with parameter distance and time.\n"
        "- Higher deviation compresses the high-coherence region."
    )


# -------------------------
# Tab 19: Branch Count
# -------------------------
with tabs[19]:
    st.subheader("Branch Count")

    steps = 200
    t = np.linspace(0, 1, steps)

    rate = (S * EM * G) / (DE * (W + 0.2))
    rate = float(np.clip(rate, 0.05, 10.0))

    branches = np.exp((0.8 * rate) * t * (1.0 + 0.25 * deviation))
    branches = np.clip(branches, 1, 1e18)

    branches_log = np.log10(branches + 1e-9)
    end_log = float(branches_log[-1])

    show_ribbon_or_line(
        x=t,
        y=branches_log,
        title="Branch Growth Over Time",
        xlab="Time",
        ylab="log10 Branch Count",
        fname_base="Branch_Count_Log",
        ribbon_width=1.2,
        pdf_title="Branch Growth Over Time",
        pdf_text=(
            f"Branch growth rate scales with Strong, EM, and Gravity and is suppressed by Dark Energy and Weak. "
            f"Rate={rate:.2f} with deviation={deviation:.3f} yields final log10 branch count {end_log:.2f} at late time."
        )
    )

    d_vals = np.linspace(0, 6, max(30, res_3d // 2))
    TT, DD = np.meshgrid(t[::3], d_vals, indexing="ij")
    branches2 = np.exp((0.8 * rate) * TT * (1.0 + 0.25 * DD))
    branches2 = np.clip(branches2, 1, 1e18)

    show_surface_or_heatmap(
        z=np.log10(branches2 + 1e-9),
        x=d_vals,
        y=t[::3],
        title="Branch Growth vs Deviation",
        xlab="Deviation",
        ylab="Time",
        zlab="log10 Branches",
        fname_base="Branch_Count_Surface",
        colorscale="Viridis",
        pdf_title="Branch Growth vs Deviation",
        pdf_text=(
            f"This surface shows how branch count accelerates as deviation grows. "
            f"At your current deviation={deviation:.3f}, growth follows the lower edge of the surface. "
            f"Universes far from baseline split more aggressively in this toy model."
        )
    )


# -------------------------
# Tab 20: Quantum Gravity Horizon
# -------------------------
with tabs[20]:
    st.subheader("Quantum Gravity Horizon")

    r_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    g_vals = np.linspace(0.1, 10.0, max(30, res_3d // 2))
    R, GG = np.meshgrid(r_vals, g_vals, indexing="ij")

    curvature = (GG * G) / (R + 1e-6)
    curvature *= (1.0 / (1.0 + 0.25 * DE))
    curvature = np.clip(curvature, 0, 2.0)
    curvature_gr = curvature.T

    curv_mean = float(np.nanmean(curvature_gr))
    curv_peak = float(np.nanmax(curvature_gr))

    show_surface_or_heatmap(
        z=curvature_gr,
        x=r_vals,
        y=g_vals,
        title="Curvature Map",
        xlab="Radius r",
        ylab="Gravity Field",
        zlab="Curvature",
        fname_base="Quantum_Gravity",
        colorscale="Cividis",
        pdf_title="Curvature Map",
        pdf_text=(
            f"Curvature increases with gravity strength and decreases with radius. "
            f"With Gravity={G:.2f} and Dark Energy={DE:.2f}, mean curvature is {curv_mean:.2f} and peak is {curv_peak:.2f}. "
            f"Regions near the cap represent horizon-like, high-curvature zones in this proxy."
        )
    )

    st.markdown(
        "- Higher gravity raises curvature and pushes toward horizon-like behavior.\n"
        "- Dark energy reduces effective curvature in this simplified mapping."
    )


# =========================
# PDF Export
# =========================
st.divider()
st.subheader("Export Simulation Report")

if st.button("Generate Scientific PDF Report"):
    with st.spinner("Compiling PDF..."):
        try:
            summary_text = st.session_state.get("summary", "No AI summary generated.")
            outpdf = generate_pdf(constants, summary_text)
            st.success("PDF generated successfully!")
            with open(outpdf, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=outpdf,
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")