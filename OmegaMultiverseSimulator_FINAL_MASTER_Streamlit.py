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

st.sidebar.header("Controls")

disable_3d = st.sidebar.checkbox(
    "Disable 3D (recommended on iPhone if plots appear blank)",
    value=False
)

auto_save_plots = st.sidebar.checkbox(
    "Auto-save plots for PDF (can slow app)",
    value=True
)

def clamp(x, eps=1e-6):
    """Array-safe clamp (works for scalars and numpy arrays)."""
    return np.maximum(x, eps)

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
    pdf.cell(0, 12, "Omega Multiverse Simulation Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font(font, "", 12)
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 8, f"Date: {date_str}", ln=True, align="C")
    pdf.cell(0, 8, "Generated via Omega Multiverse Simulator", ln=True, align="C")

    # Params
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Parameters", ln=True)
    pdf.set_font(font, "", 12)
    for k, v in constants.items():
        line = f"{k}: {v:.4f}"
        pdf.cell(0, 7, line.encode("latin-1", "replace").decode("latin-1"), ln=True)

    # AI Summary
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "AI Universe Summary", ln=True)
    pdf.set_font(font, "", 12)
    for line in (summary_text or "").split("\n"):
        safe_line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe_line)

    # Visuals
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Visuals", ln=True)

    image_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".png")])
    for image_file in image_files:
        path = os.path.join(output_dir, image_file)
        pdf.add_page()
        pdf.set_font(font, "B", 12)
        title = image_file.replace(".png", "").replace("_", " ")
        pdf.cell(0, 8, title.encode("latin-1", "replace").decode("latin-1"), ln=True)
        try:
            pdf.image(path, w=180)
        except Exception:
            pdf.set_font(font, "", 10)
            pdf.multi_cell(0, 6, f"(Could not embed image: {image_file})")

    outname = "Omega_Universe_Simulation_Report.pdf"
    pdf.output(outname)
    return outname


# =========================
# Constants
# =========================
constants = {
    "Strong Force Multiplier": slider_with_input("Strong Force", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_input("Electromagnetic (EM)", 0.1, 10.0, 1.0, 0.01),
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
    st.success("Low deviation → broad stability expected.")
elif deviation < 4.0:
    st.warning("Moderate deviation → some instabilities possible.")
else:
    st.error("High deviation → instability likely across physics & chemistry.")

st.divider()


# =========================
# Shared nuclear grids
# =========================
Z = np.arange(1, 121)                # proton number
N = np.arange(1, 181)                # neutron number
Zg, Ng = np.meshgrid(Z, N, indexing="ij")
Ag = Zg + Ng

# A soft “valley-of-stability” target:
Nz_target = 1.0 + (Zg / 80.0)
ratio = Ng / clamp(Zg, 1)

# nuclear stability proxy:
ratio_penalty = np.exp(-((ratio - Nz_target) ** 2) * 8.0)
strong_bonus = np.exp(-np.abs(Zg - 82) / (28.0 * clamp(S, 1e-3)))
em_penalty = np.exp(-np.maximum(Zg - 20, 0) / (30.0 / clamp(EM, 1e-3)))
weak_opt = np.exp(-((W - 1.0) ** 2) * 2.5)

nuclear_stability = np.clip(ratio_penalty * strong_bonus * em_penalty * weak_opt, 0, 1)

# isotopes count approximation
isotope_viable_per_Z = (nuclear_stability > 0.20).sum(axis=1)


# =========================
# AI Summary (Optional)
# =========================
st.subheader("AI Global Universe Analysis (Optional)")
ai_ok = ("OPENAI_API_KEY" in st.secrets) and (openai is not None)

if not ai_ok:
    st.info("OpenAI not configured. Add OPENAI_API_KEY to .streamlit/secrets.toml to enable.")

if ai_ok:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if st.button("Generate AI Universe Summary"):
        with st.spinner("Generating summary..."):
            user_context = "\n".join([f"{k}: {v:.4f}" for k, v in constants.items()])
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a physics/cosmology expert. Provide a rigorous but accessible analysis of a universe defined by these dimensionless multipliers. Discuss nuclear stability, chemistry, star formation, habitability, and long-term evolution. Avoid fluff."},
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
    "Star Formation Potential",
    "Life Probability (Heatmap)",
    "Quantum Bonding",
    "Universe Emergence Probability",
    "Element Abundance Probability",
    "EM Radiation Risk",
    "Star Lifespan Model",
    "Dark Matter Simulation",
    "Atomic Stability",
    "Universe Life Probability Over Time",
    "Molecular Bonding Model (Element Specific)",
    "Molecular Abundance Map",
    "Isotope Decay & Half-Life Model",
    "Periodic Table Expansion Potential",
    "Proton–Neutron Ratio Heatmap",
    "Nuclear Binding Energy Map",
    "Multiverse Decoherence Map",
    "Quantum Branch Count Estimator",
    "Quantum Gravity Horizon Map",
])


# -------------------------
# Tab 0: Periodic Table Stability
# -------------------------
with tabs[0]:
    st.subheader("Periodic Table Stability (2D + 3D)")

    em_vals = np.linspace(0.1, 10.0, 70)
    Z_vals = np.arange(1, 121)
    Z2, EM2 = np.meshgrid(Z_vals, em_vals, indexing="ij")

    base_shell = np.exp(-np.abs(Z2 - 30) / 22.0)
    strong_term = np.exp(-np.abs(Z2 - 82) / (28.0 * clamp(S, 1e-3)))

    # ✅ FIX: array-safe clamp (no Python max on arrays)
    em_term = (
        np.exp(-np.abs(EM2 - EM) / 1.2)
        * np.exp(-np.maximum(Z2 - 20, 0) / (40.0 / clamp(EM2, 1e-3)))
    )

    weak_term = np.exp(-((W - 1.0) ** 2) * 2.5)

    stability = np.clip(base_shell * strong_term * em_term * weak_term, 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=stability.T,
        x=Z_vals,
        y=em_vals,
        colorscale="Viridis",
        colorbar=dict(title="Stability")
    ))
    fig2d.update_layout(
        title="Stability vs Atomic Number and EM Multiplier",
        xaxis_title="Atomic Number (Z)",
        yaxis_title="EM Multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Periodic_Table_Stability_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=stability,
            x=em_vals,
            y=Z_vals,
            colorscale="Viridis",
            colorbar=dict(title="Stability")
        )])
        fig3d.update_layout(
            title="3D Stability Surface",
            scene=dict(
                xaxis_title="EM Multiplier",
                yaxis_title="Atomic Number (Z)",
                zaxis_title="Stability"
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Periodic_Table_Stability_3D.png", is_plotly=True)

    st.markdown(
        "- Nuclear stability rises with **strong force** (cohesion) and falls with **EM** (Coulomb repulsion).\n"
        "- **Weak force** sets beta-decay efficiency; far from ~1.0 tends to reduce long-lived isotopes."
    )


# -------------------------
# Tab 1: Island of Instability
# -------------------------
with tabs[1]:
    st.subheader("Island of Instability (shell-like periodicity + force tuning)")

    Z_vals = np.linspace(40, 140, 100)
    S_vals = np.linspace(0.1, 10.0, 90)
    Z2, S2 = np.meshgrid(Z_vals, S_vals, indexing="ij")

    shell_waves = (0.55 + 0.45*np.sin(Z2 / 7.0)**2) * (0.55 + 0.45*np.sin(Z2 / 11.0)**2)
    strong_opt = np.exp(-np.abs(S2 - S) / 1.6)
    coulomb = np.exp(-np.maximum(Z2 - 60, 0) / (25.0 / clamp(EM, 1e-3)))

    instability = np.clip((1 - strong_opt) * shell_waves * (1 / np.maximum(coulomb, 1e-6)), 0, 2.0)

    fig2d = go.Figure(data=go.Contour(
        z=instability.T,
        x=Z_vals,
        y=S_vals,
        contours_coloring="heatmap",
        colorscale="Inferno",
        colorbar=dict(title="Instability")
    ))
    fig2d.update_layout(
        title="Instability Contours (Z vs Strong Force)",
        xaxis_title="Atomic Number (Z)",
        yaxis_title="Strong Force Multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Island_Instability_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=instability,
            x=S_vals,
            y=Z_vals,
            colorscale="Inferno",
            colorbar=dict(title="Instability")
        )])
        fig3d.update_layout(
            title="3D Instability Surface",
            scene=dict(
                xaxis_title="Strong Force",
                yaxis_title="Atomic Number (Z)",
                zaxis_title="Instability"
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Island_Instability_3D.png", is_plotly=True)

    st.markdown(
        "- This tab visualizes a synthetic “island of instability” combining shell-like periodic structure with force tuning.\n"
        "- Higher **EM** pushes heavy nuclei toward instability; higher **Strong** can counteract it."
    )


# -------------------------
# Tab 2: Star Formation Potential
# -------------------------
with tabs[2]:
    st.subheader("Star Formation Potential (Gravity vs Dark Energy + radiative feedback)")

    g_vals = np.linspace(0.1, 10.0, 80)
    de_vals = np.linspace(0.1, 10.0, 80)
    G2, DE2 = np.meshgrid(g_vals, de_vals, indexing="ij")

    collapse = (G2**1.1) / (DE2**1.2 + 0.15)
    rad_pressure = 1 / (1 + (EM**1.3))
    ignition = np.exp(-np.abs(S - 1.0) * 0.8) * np.exp(-((W - 1.0) ** 2) * 1.2)

    sfr = safe_norm(collapse * rad_pressure * ignition)

    fig2d = go.Figure(data=go.Heatmap(
        z=sfr.T,
        x=g_vals,
        y=de_vals,
        colorscale="Viridis",
        colorbar=dict(title="Star Formation")
    ))
    fig2d.update_layout(
        title="Star Formation Potential (2D)",
        xaxis_title="Gravity Multiplier",
        yaxis_title="Dark Energy Multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Star_Formation_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=sfr,
            x=g_vals,
            y=de_vals,
            colorscale="Viridis",
            colorbar=dict(title="Star Formation")
        )])
        fig3d.update_layout(
            title="Star Formation Potential (3D)",
            scene=dict(
                xaxis_title="Gravity",
                yaxis_title="Dark Energy",
                zaxis_title="Potential"
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Star_Formation_3D.png", is_plotly=True)

    st.markdown(
        "- Higher **Gravity** promotes collapse; higher **Dark Energy** suppresses large-scale structure formation.\n"
        "- **EM** adds radiative pressure feedback; **Strong/Weak** influence ignition efficiency (toy proxy)."
    )


# -------------------------
# Tab 3: Life Probability (Heatmap)
# -------------------------
with tabs[3]:
    st.subheader("Life Probability Map (forces + thermal window + metallicity proxy)")

    s_vals = np.linspace(0.1, 10.0, 90)
    em_vals = np.linspace(0.1, 10.0, 90)
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    force_window = np.exp(-((S2 - 1.0) ** 2) / 1.6) * np.exp(-((EM2 - 1.0) ** 2) / 1.6) * np.exp(-((W - 1.0) ** 2) / 2.0)
    thermo = np.exp(-((T - 1.0) ** 2) * 1.2) * np.exp(-((P - 1.0) ** 2) * 1.0)
    metals = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0)*0.6), 0, 5)
    metals_factor = np.tanh(metals / 1.5)

    life = np.clip(force_window * thermo * metals_factor, 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=life.T,
        x=s_vals,
        y=em_vals,
        colorscale="Plasma",
        colorbar=dict(title="Life Probability")
    ))
    fig2d.update_layout(
        title="Life Probability (2D)",
        xaxis_title="Strong Force Multiplier",
        yaxis_title="EM Force Multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Life_Probability_2D.png", is_plotly=True)

    figc = go.Figure(data=go.Contour(
        z=life.T,
        x=s_vals,
        y=em_vals,
        contours_coloring="heatmap",
        colorscale="Plasma",
        colorbar=dict(title="Life Probability")
    ))
    figc.update_layout(
        title="Life Probability Zones (Contour)",
        xaxis_title="Strong",
        yaxis_title="EM"
    )
    st.plotly_chart(figc, use_container_width=True)
    save_plot(figc, "Life_Probability_Contour.png", is_plotly=True)

    st.markdown(
        "- This combines a **force-compatibility window** (chemistry) + **thermodynamic window** (T/P) + a **metallicity proxy** (needs stars & heavy elements).\n"
        "- Treat as a scientific *proxy* model: consistent, not a claim of real probabilities."
    )


# -------------------------
# Tab 4: Quantum Bonding
# -------------------------
with tabs[4]:
    st.subheader("Quantum Bonding (2D contour + optional 3D)")

    s_vals = np.linspace(0.1, 10.0, 80)
    em_vals = np.linspace(0.1, 10.0, 80)
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    em_binding = np.exp(-((EM2 - 1.0) ** 2) / 1.4)
    strong_chem = np.exp(-abs(S2 - 1.0) / 2.0)
    temp_kill = np.exp(-((T - 1.0) ** 2) * 1.8)
    pressure_help = np.tanh(P / 1.5)

    bonding = np.clip(em_binding * strong_chem * temp_kill * pressure_help, 0, 1)

    fig2d = go.Figure(data=go.Contour(
        z=bonding.T,
        x=s_vals,
        y=em_vals,
        contours_coloring="heatmap",
        colorscale="Viridis",
        colorbar=dict(title="Bonding")
    ))
    fig2d.update_layout(
        title="Quantum Bonding Zone Map (2D)",
        xaxis_title="Strong",
        yaxis_title="EM"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Quantum_Bonding_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=bonding,
            x=s_vals,
            y=em_vals,
            colorscale="Viridis",
            colorbar=dict(title="Bonding")
        )])
        fig3d.update_layout(
            title="Quantum Bonding Surface (3D)",
            scene=dict(
                xaxis_title="Strong",
                yaxis_title="EM",
                zaxis_title="Bonding"
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Quantum_Bonding_3D.png", is_plotly=True)

    st.markdown(
        "- Bonding rises near EM≈1 and Strong≈1, but falls at high temperature.\n"
        "- Pressure increases overlap (proxy), raising bonding in this simplified model."
    )


# -------------------------
# Tab 5: Universe Emergence Probability
# -------------------------
with tabs[5]:
    st.subheader("Universe Emergence / Viability Score")

    viability = float(np.exp(-0.35 * deviation))
    chaos = float(1.0 - viability)

    fig = go.Figure(data=[
        go.Bar(name="Viability", x=["Score"], y=[viability]),
        go.Bar(name="Chaos/Instability", x=["Score"], y=[chaos]),
    ])
    fig.update_layout(
        barmode="stack",
        title="Global Viability Proxy",
        yaxis_title="Normalized Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Universe_Viability.png", is_plotly=True)

    st.markdown(
        "- This is a **global proxy** based on how far constants deviate from baseline.\n"
        "- It’s not a physical probability; it’s a compact “distance-from-our-universe” viability score."
    )


# -------------------------
# Tab 6: Element Abundance Probability
# -------------------------
with tabs[6]:
    st.subheader("Element Abundance Probability (toy nucleosynthesis proxy)")

    surv = safe_norm(isotope_viable_per_Z.astype(float))
    stellar = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0)*0.4) * np.exp(-abs(S - 1.0)*0.3), 0, 3)
    stellar_factor = np.tanh(stellar / 1.3)

    abundance = np.clip(surv * stellar_factor, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Z, y=abundance, mode="lines", name="Abundance Proxy"))
    fig.update_layout(
        title="Element Abundance vs Z",
        xaxis_title="Atomic Number (Z)",
        yaxis_title="Relative Abundance (proxy)"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Element_Abundance_Line.png", is_plotly=True)

    t = np.linspace(0, 1, 60)
    Z2, T2 = np.meshgrid(Z, t, indexing="ij")
    enrich = 1 - np.exp(-T2 * (2.0 + 4.0*stellar_factor))
    abund_time = np.clip((abundance[:, None]) * enrich, 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=abund_time.T,
        x=Z,
        y=t,
        colorscale="Cividis",
        colorbar=dict(title="Abundance")
    ))
    fig2d.update_layout(
        title="Abundance Evolution (2D)",
        xaxis_title="Z",
        yaxis_title="Normalized Cosmic Time"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Element_Abundance_2D.png", is_plotly=True)

    st.markdown(
        "- Combines isotope survivability with a crude star-processing/enrichment proxy.\n"
        "- Useful for comparative “what if constants change?” exploration."
    )


# -------------------------
# Tab 7: EM Radiation Risk
# -------------------------
with tabs[7]:
    st.subheader("EM Radiation Risk")

    x = np.linspace(0.1, 10.0, 600)
    y = (x ** 2) * (0.4 + 0.6*np.tanh(T / 2.0)) / 20.0
    y = np.clip(y, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Radiation Risk"))
    fig.add_vline(x=EM, line_dash="dash", annotation_text="Current EM", annotation_position="top")
    fig.update_layout(
        title="Radiation Risk vs EM Multiplier",
        xaxis_title="EM Multiplier",
        yaxis_title="Normalized Risk"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "EM_Radiation_Risk.png", is_plotly=True)

    st.markdown("- Higher EM generally increases radiative coupling; this is a proxy curve for comparative risk.")


# -------------------------
# Tab 8: Star Lifespan Model
# -------------------------
with tabs[8]:
    st.subheader("Star Lifespan Model (mass-luminosity proxy)")

    g_vals = np.linspace(0.1, 10.0, 400)
    M = g_vals
    L = (M ** 3.5) * (1.0 + 0.15*(EM-1.0))
    tau = (M / np.maximum(L, 1e-9))
    tau = safe_norm(tau)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g_vals, y=tau, mode="lines", name="Relative Lifetime"))
    fig.add_vline(x=G, line_dash="dash", annotation_text="Current G", annotation_position="top")
    fig.update_layout(
        title="Relative Stellar Lifetime vs Gravity",
        xaxis_title="Gravity Multiplier",
        yaxis_title="Relative Lifetime"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Star_Lifespan.png", is_plotly=True)

    st.markdown("- Stronger gravity → more massive characteristic stars → shorter lifetimes (proxy).")


# -------------------------
# Tab 9: Dark Matter Simulation
# -------------------------
with tabs[9]:
    st.subheader("Dark Matter / Cosmic Web Proxy (3D + 2D slice)")

    size = 32
    scale = 8.0

    cluster_spread = (3.5 / clamp(G, 1e-3))
    stretch = (1.0 / clamp(DE, 1e-3))

    x = np.linspace(-scale*stretch, scale*stretch, size)
    y = np.linspace(-scale*stretch, scale*stretch, size)
    z = np.linspace(-scale*stretch, scale*stretch, size)
    X, Y, Z3 = np.meshgrid(x, y, z, indexing="ij")

    density = np.zeros_like(X)
    rng = np.random.default_rng(42)
    centers = rng.uniform(-scale, scale, size=(5, 3))

    for cx, cy, cz in centers:
        r2 = (X-cx)**2 + (Y-cy)**2 + (Z3-cz)**2
        density += np.exp(-r2 / (cluster_spread**2))

    density = safe_norm(density)

    mid = size // 2
    slice2d = density[:, :, mid]

    fig2d = go.Figure(data=go.Heatmap(
        z=slice2d.T,
        x=x, y=y,
        colorscale="Inferno",
        colorbar=dict(title="Density")
    ))
    fig2d.update_layout(
        title="Cosmic Web Density Slice (2D)",
        xaxis_title="X", yaxis_title="Y"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Dark_Matter_2D_Slice.png", is_plotly=True)

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
            title="Cosmic Web (3D points above threshold)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Dark_Matter_3D.png", is_plotly=True)

    st.markdown("- Higher Gravity compresses structure; higher Dark Energy expands voids (proxy).")


# -------------------------
# Tab 10: Atomic Stability
# -------------------------
with tabs[10]:
    st.subheader("Atomic / Isotope Stability (2D + optional 3D)")

    fig2d = go.Figure(data=go.Heatmap(
        z=nuclear_stability.T,
        x=Z, y=N,
        colorscale="Plasma",
        colorbar=dict(title="Stability")
    ))
    fig2d.update_layout(
        title="Isotope Stability Map (Z vs N)",
        xaxis_title="Z (protons)",
        yaxis_title="N (neutrons)"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Atomic_Stability_2D.png", is_plotly=True)

    if not disable_3d:
        Zs = Z[::3]
        Ns = N[::3]
        stab_ds = nuclear_stability[::3, ::3]
        fig3d = go.Figure(data=[go.Surface(
            z=stab_ds,
            x=Zs,
            y=Ns,
            colorscale="Plasma",
            colorbar=dict(title="Stability")
        )])
        fig3d.update_layout(
            title="Stability Surface (Downsampled 3D)",
            scene=dict(xaxis_title="Z", yaxis_title="N", zaxis_title="Stability")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Atomic_Stability_3D.png", is_plotly=True)

    st.markdown("- Built from a valley-of-stability proxy: strong force helps binding; EM penalizes high-Z; weak sets beta-decay efficiency.")


# -------------------------
# Tab 11: Universe Life Probability Over Time
# -------------------------
with tabs[11]:
    st.subheader("Life Probability Over Cosmic Time (metals + stable stars + chemistry window)")

    time = np.linspace(0, 1, 200)

    sfr_now = np.clip((G**1.1) / (DE**1.2 + 0.15) * (1 / (1 + EM**1.3)), 0, 10)
    sfr_now = float(np.tanh(sfr_now / 2.0))

    metals = 1 - np.exp(-(2.0 + 5.0*sfr_now) * time)
    metals = np.clip(metals, 0, 1)

    star_window = float(np.exp(-abs(G - 1.0) * 0.6))
    chem_window = float(np.exp(-abs(S - 1.0)*0.4) * np.exp(-abs(EM - 1.0)*0.4) * np.exp(-((W - 1.0)**2)*0.4))
    thermo_window = float(np.exp(-((T - 1.0)**2)*0.6) * np.exp(-((P - 1.0)**2)*0.4))

    life_t = np.clip(metals * star_window * chem_window * thermo_window, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=metals, mode="lines", name="Metallicity"))
    fig.add_trace(go.Scatter(x=time, y=life_t, mode="lines", name="Life Probability (proxy)"))
    fig.update_layout(
        title="Life Probability Over Time (Proxy)",
        xaxis_title="Normalized Cosmic Time",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Life_Over_Time.png", is_plotly=True)


# -------------------------
# Tab 12: Molecular Bonding Model
# -------------------------
with tabs[12]:
    st.subheader("Molecular Bonding Viability (element families)")

    denom = np.max(isotope_viable_per_Z)
    isotope_factor = float(np.mean(isotope_viable_per_Z) / denom) if denom > 0 else 0.0
    isotope_factor = np.clip(isotope_factor, 0, 1)

    families = {
        "Simple Covalent (H₂/O₂)": 0.95,
        "Polar (H₂O/NH₃)": 0.90,
        "Carbon Backbone (CO₂/CH₄)": 0.88,
        "Metallic (Fe/Ni)": 0.75,
        "Heavy Chemistry (U/Th)": 0.60,
    }

    em_mod = np.exp(-abs(EM - 1.0) * 0.7)
    strong_mod = np.exp(-abs(S - 1.0) * 0.6)
    weak_mod = np.exp(-((W - 1.0)**2) * 0.4)
    temp_mod = np.exp(-((T - 1.0)**2) * 0.9)
    press_mod = np.tanh(P / 1.6)

    global_mod = float(np.clip(em_mod * strong_mod * weak_mod * temp_mod * press_mod * (0.6 + 0.4*isotope_factor), 0, 1))

    names = list(families.keys())
    vals = [np.clip(families[k] * global_mod, 0, 1) for k in names]

    fig = go.Figure(data=[go.Bar(x=names, y=vals, text=[f"{v:.2f}" for v in vals], textposition="outside")])
    fig.update_layout(
        title="Molecular Bonding Viability (Proxy)",
        yaxis_title="Viability",
        yaxis_range=[0, 1.15]
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Molecular_Bonding.png", is_plotly=True)


# -------------------------
# Tab 13: Molecular Abundance Map
# -------------------------
with tabs[13]:
    st.subheader("Molecular Abundance (2D map over Temperature & Pressure)")

    tvals = np.linspace(0.1, 10.0, 80)
    pvals = np.linspace(0.1, 10.0, 80)
    T2, P2 = np.meshgrid(tvals, pvals, indexing="ij")

    force_gate = float(np.exp(-abs(S - 1.0)*0.4) * np.exp(-abs(EM - 1.0)*0.4))
    thermo_gate = np.exp(-((T2 - 1.0)**2) * 0.8) * np.exp(-((P2 - 1.0)**2) * 0.6)
    abundance = np.clip(force_gate * thermo_gate, 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=abundance.T,
        x=tvals, y=pvals,
        colorscale="Viridis",
        colorbar=dict(title="Abundance")
    ))
    fig2d.update_layout(
        title="Molecular Abundance (T vs P)",
        xaxis_title="Temperature Multiplier",
        yaxis_title="Pressure Multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Molecular_Abundance_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=abundance,
            x=tvals, y=pvals,
            colorscale="Viridis",
            colorbar=dict(title="Abundance")
        )])
        fig3d.update_layout(
            title="Molecular Abundance Surface (3D)",
            scene=dict(xaxis_title="T", yaxis_title="P", zaxis_title="Abundance")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Molecular_Abundance_3D.png", is_plotly=True)


# -------------------------
# Tab 14: Isotope Decay & Half-Life Model
# -------------------------
with tabs[14]:
    st.subheader("Isotope Half-Life Proxy (Z vs N) + long-lived count")

    half_life = np.clip(nuclear_stability * np.exp(-abs(W - 1.0) * 0.6), 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=half_life.T,
        x=Z, y=N,
        colorscale="Cividis",
        colorbar=dict(title="Half-life (proxy)")
    ))
    fig2d.update_layout(
        title="Half-Life Proxy Map (Z vs N)",
        xaxis_title="Z", yaxis_title="N"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Half_Life_2D.png", is_plotly=True)

    long_lived = (half_life > 0.35).sum(axis=1)
    fig = go.Figure(data=[go.Bar(x=Z, y=long_lived)])
    fig.update_layout(
        title="Count of Long-Lived Isotopes per Z (proxy threshold)",
        xaxis_title="Z",
        yaxis_title="# long-lived isotopes"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Half_Life_LongLived_Count.png", is_plotly=True)


# -------------------------
# Tab 15: Periodic Table Expansion Potential
# -------------------------
with tabs[15]:
    st.subheader("Periodic Table Expansion Potential")

    Z_ext = np.arange(1, 201)
    cohesion = np.exp(-np.abs(Z_ext - 82) / (30.0 * clamp(S, 1e-3)))
    coulomb = 1 / (1 + np.exp(-(Z_ext - 110) / (12.0 / clamp(EM, 1e-3))))
    decay = np.exp(-((W - 1.0) ** 2) * 1.2)

    stability_curve = np.clip(cohesion * (1 - coulomb) * decay, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Z_ext, y=stability_curve, mode="lines", name="Stability Potential"))
    fig.update_layout(
        title="Expansion Limit Curve",
        xaxis_title="Z",
        yaxis_title="Stability Potential"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Periodic_Table_Expansion.png", is_plotly=True)

    maxZ = int(Z_ext[stability_curve > 0.12][-1]) if np.any(stability_curve > 0.12) else 0
    st.markdown(f"**Estimated max potentially-stable Z (proxy threshold): {maxZ}**")

    em_vals = np.linspace(0.1, 10.0, 80)
    Z2, EM2 = np.meshgrid(Z_ext, em_vals, indexing="ij")
    cohesion2 = np.exp(-np.abs(Z2 - 82) / (30.0 * clamp(S, 1e-3)))
    coulomb2 = 1 / (1 + np.exp(-(Z2 - 110) / (12.0 / clamp(EM2, 1e-3))))
    stab2 = np.clip(cohesion2 * (1 - coulomb2) * decay, 0, 1)

    fig2d = go.Figure(data=go.Heatmap(
        z=stab2.T, x=Z_ext, y=em_vals, colorscale="Viridis",
        colorbar=dict(title="Stability")
    ))
    fig2d.update_layout(
        title="Expansion Potential Map (Z vs EM)",
        xaxis_title="Z",
        yaxis_title="EM"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Periodic_Table_Expansion_2D.png", is_plotly=True)


# -------------------------
# Tab 16: Proton–Neutron Ratio Heatmap
# -------------------------
with tabs[16]:
    st.subheader("Proton–Neutron Ratio / Valley of Stability")

    fig2d = go.Figure(data=go.Heatmap(
        z=nuclear_stability.T,
        x=Z, y=N,
        colorscale="Magma",
        colorbar=dict(title="Viability")
    ))
    fig2d.update_layout(
        title="Viability in (Z, N)",
        xaxis_title="Z",
        yaxis_title="N"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "PN_Heatmap.png", is_plotly=True)

    target_line = (1.0 + (Z / 80.0)) * Z
    figline = go.Figure()
    figline.add_trace(go.Scatter(x=Z, y=target_line, mode="lines", name="Target N(Z)"))
    figline.update_layout(
        title="Valley-of-Stability Target Curve (proxy)",
        xaxis_title="Z",
        yaxis_title="Target N"
    )
    st.plotly_chart(figline, use_container_width=True)
    save_plot(figline, "PN_Target_Curve.png", is_plotly=True)


# -------------------------
# Tab 17: Nuclear Binding Energy Map
# -------------------------
with tabs[17]:
    st.subheader("Nuclear Binding Energy (SEMF-style proxy)")

    a_v = 15.8 * S
    a_s = 18.3
    a_c = 0.714 * EM
    a_sym = 23.2 * (1 / clamp(W, 1e-3))
    a_pair = 12.0

    A = Ag
    pairing = np.where(((Zg % 2 == 0) & (Ng % 2 == 0)), +1, -1)

    BE = (
        a_v * A
        - a_s * (A ** (2/3))
        - a_c * (Zg * (Zg - 1)) / clamp(A ** (1/3), 1e-9)
        - a_sym * ((A - 2*Zg) ** 2) / clamp(A, 1e-9)
        + pairing * a_pair / clamp(A ** 0.5, 1e-9)
    )

    BE_per_A = np.clip(BE / clamp(A, 1), 0, None)
    BE_per_A = np.clip(BE_per_A, 0, np.nanpercentile(BE_per_A, 99))

    fig2d = go.Figure(data=go.Heatmap(
        z=BE_per_A.T,
        x=Z, y=N,
        colorscale="Viridis",
        colorbar=dict(title="BE/A (proxy)")
    ))
    fig2d.update_layout(
        title="Binding Energy per Nucleon Map (Z vs N)",
        xaxis_title="Z",
        yaxis_title="N"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Binding_Energy_2D.png", is_plotly=True)

    target_line = (1.0 + (Z / 80.0)) * Z
    N_pick = np.clip(target_line.astype(int), 1, N[-1])
    idxN = N_pick - 1
    be_line = BE_per_A[np.arange(len(Z)), idxN]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Z, y=be_line, mode="lines+markers", name="BE/A along valley"))
    fig.update_layout(
        title="BE/A along Valley-of-Stability (proxy)",
        xaxis_title="Z",
        yaxis_title="BE/A"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Binding_Energy_Line.png", is_plotly=True)

    peakZ = int(Z[np.nanargmax(be_line)])
    st.markdown(f"**Peak along valley at Z ≈ {peakZ} (proxy)**")


# -------------------------
# Tab 18: Multiverse Decoherence Map
# -------------------------
with tabs[18]:
    st.subheader("Multiverse Decoherence Map (parameter-space distance + time decay)")

    s_vals = np.linspace(0.1, 10.0, 70)
    em_vals = np.linspace(0.1, 10.0, 70)
    t_vals = np.linspace(0, 1, 60)

    S2, EM2, TT = np.meshgrid(s_vals, em_vals, t_vals, indexing="ij")

    dist2 = (
        (S2 - S)**2 +
        (EM2 - EM)**2 +
        (W - 1.0)**2 +
        (G - 1.0)**2 +
        (DE - 1.0)**2
    )

    coherence = np.exp(-dist2 / 4.0) * np.exp(-TT * (0.8 + 0.6*deviation/4.0))

    mid = coherence[:, :, len(t_vals)//2]

    fig2d = go.Figure(data=go.Heatmap(
        z=mid.T, x=s_vals, y=em_vals,
        colorscale="Magma",
        colorbar=dict(title="Coherence")
    ))
    fig2d.update_layout(
        title="Mid-Time Coherence Slice (2D)",
        xaxis_title="Strong",
        yaxis_title="EM"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Decoherence_MidTime_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=mid,
            x=s_vals,
            y=em_vals,
            colorscale="Magma",
            colorbar=dict(title="Coherence")
        )])
        fig3d.update_layout(
            title="Coherence Surface (3D, mid-time)",
            scene=dict(xaxis_title="Strong", yaxis_title="EM", zaxis_title="Coherence")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Decoherence_MidTime_3D.png", is_plotly=True)

    st.markdown("- A toy Everett-style visualization: coherence decays with parameter-distance and time.")


# -------------------------
# Tab 19: Quantum Branch Count Estimator
# -------------------------
with tabs[19]:
    st.subheader("Quantum Branch Count Estimator (decoherence-driven growth)")

    steps = 200
    t = np.linspace(0, 1, steps)

    rate = (S * EM * G) / (DE * (W + 0.2))
    rate = float(np.clip(rate, 0.05, 10.0))

    branches = np.exp((0.8 * rate) * t * (1.0 + 0.25*deviation))
    branches = np.clip(branches, 1, 1e18)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=branches, mode="lines", name="Estimated Branches"))
    fig.update_layout(
        title="Branch Count vs Time (log scale)",
        xaxis_title="Normalized Time",
        yaxis_title="Branch Count",
        yaxis_type="log"
    )
    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Branch_Count.png", is_plotly=True)

    d_vals = np.linspace(0, 6, 80)
    TT, DD = np.meshgrid(t, d_vals, indexing="ij")
    branches2 = np.exp((0.8 * rate) * TT * (1.0 + 0.25*DD))
    branches2 = np.clip(branches2, 1, 1e18)

    fig2d = go.Figure(data=go.Heatmap(
        z=np.log10(branches2.T + 1e-9),
        x=t, y=d_vals,
        colorscale="Viridis",
        colorbar=dict(title="log10(branches)")
    ))
    fig2d.update_layout(
        title="Branch Growth vs Time and Deviation (2D)",
        xaxis_title="Time",
        yaxis_title="Deviation"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Branch_Count_2D.png", is_plotly=True)


# -------------------------
# Tab 20: Quantum Gravity Horizon Map
# -------------------------
with tabs[20]:
    st.subheader("Quantum Gravity Horizon Map (curvature proxy)")

    r_vals = np.linspace(0.1, 10.0, 120)
    g_vals = np.linspace(0.1, 10.0, 120)
    R, GG = np.meshgrid(r_vals, g_vals, indexing="ij")

    curvature = (GG * G) / (R + 1e-6)
    curvature *= (1.0 / (1.0 + 0.25*DE))
    curvature = np.clip(curvature, 0, 2.0)

    fig2d = go.Figure(data=go.Heatmap(
        z=curvature.T,
        x=r_vals, y=g_vals,
        colorscale="Cividis",
        colorbar=dict(title="Curvature")
    ))
    fig2d.update_layout(
        title="Curvature Map (2D)",
        xaxis_title="Radial coordinate r (proxy)",
        yaxis_title="Gravitational field multiplier"
    )
    st.plotly_chart(fig2d, use_container_width=True)
    save_plot(fig2d, "Quantum_Gravity_2D.png", is_plotly=True)

    if not disable_3d:
        fig3d = go.Figure(data=[go.Surface(
            z=curvature,
            x=r_vals, y=g_vals,
            colorscale="Cividis",
            colorbar=dict(title="Curvature")
        )])
        fig3d.update_layout(
            title="Curvature Surface (3D)",
            scene=dict(xaxis_title="r", yaxis_title="G-field", zaxis_title="Curvature")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Quantum_Gravity_3D.png", is_plotly=True)

    st.markdown("- Regions approaching the cap represent horizon-like Planck-curvature zones (proxy).")


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