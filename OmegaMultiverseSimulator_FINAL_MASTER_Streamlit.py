Ok here’s my code. Implement it so it changes based on values for graphs: 

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

# NEW: 3D resolution (lower = faster/more reliable on iPhone)
res_3d = st.sidebar.slider(
    "3D resolution (lower = faster / more reliable on iPhone)",
    min_value=25, max_value=120, value=60, step=5
)

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
# Plot Helpers (3D-first)
# =========================
def show_surface_or_heatmap(z, x, y, title, xlab, ylab, zlab, fname_base, colorscale="Viridis"):
    """
    z must be shape (len(y), len(x)) if you pass x as x-axis and y as y-axis.
    We enforce stable camera + margins to prevent iPhone blank WebGL.
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
            title=title + " (3D)",
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
        save_plot(fig3d, f"{fname_base}_3D.png", is_plotly=True)
    else:
        fig2d = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=zlab)
        ))
        fig2d.update_layout(title=title + " (2D fallback)", xaxis_title=xlab, yaxis_title=ylab)
        st.plotly_chart(fig2d, use_container_width=True)
        save_plot(fig2d, f"{fname_base}_2D.png", is_plotly=True)


def show_ribbon_or_line(x, y, title, xlab, ylab, fname_base, ribbon_width=1.2):
    """
    3D-first: show as a thin surface ribbon for reliability/consistency.
    Falls back to 2D line if 3D disabled.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not disable_3d:
        yy = np.array([0.0, ribbon_width])  # ribbon thickness direction
        X2, Y2 = np.meshgrid(x, yy, indexing="xy")
        Z2 = np.vstack([y, y])  # same curve duplicated across ribbon width

        fig3d = go.Figure(data=[go.Surface(
            x=X2,
            y=Y2,
            z=Z2,
            colorscale="Viridis",
            showscale=False
        )])
        fig3d.update_layout(
            title=title + " (3D ribbon)",
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
        save_plot(fig3d, f"{fname_base}_3D.png", is_plotly=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
        fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
        st.plotly_chart(fig, use_container_width=True)
        save_plot(fig, f"{fname_base}_2D.png", is_plotly=True)


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

Nz_target = 1.0 + (Zg / 80.0)
ratio = Ng / np.maximum(Zg, 1)

ratio_penalty = np.exp(-((ratio - Nz_target) ** 2) * 8.0)
strong_bonus = np.exp(-np.abs(Zg - 82) / (28.0 * max(S, 1e-3)))
em_penalty = np.exp(-np.maximum(Zg - 20, 0) / (30.0 / max(EM, 1e-3)))
weak_opt = np.exp(-((W - 1.0) ** 2) * 2.5)

nuclear_stability = np.clip(ratio_penalty * strong_bonus * em_penalty * weak_opt, 0, 1)
isotope_viable_per_Z = (nuclear_stability > 0.20).sum(axis=1)


# =========================
# AI Summary (Optional)
# =========================
st.subheader("AI Global Universe Analysis (Optional)")
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
    st.subheader("Periodic Table Stability (3D-first)")

    em_vals = np.linspace(0.1, 10.0, res_3d)
    Z_vals = np.arange(1, 121)
    Z2, EM2 = np.meshgrid(Z_vals, em_vals, indexing="ij")

    base_shell = np.exp(-np.abs(Z2 - 30) / 22.0)
    strong_term = np.exp(-np.abs(Z2 - 82) / (28.0 * max(S, 1e-3)))
    em_term = np.exp(-np.abs(EM2 - EM) / 1.2) * np.exp(-np.maximum(Z2 - 20, 0) / (40.0 / np.maximum(EM2, 1e-3)))
    weak_term = np.exp(-((W - 1.0) ** 2) * 2.5)

    stability = np.clip(base_shell * strong_term * em_term * weak_term, 0, 1)

    # NOTE: show_surface_or_heatmap expects z shape (len(y), len(x)).
    # Here x=em_vals, y=Z_vals, so z must be (len(Z_vals), len(em_vals)) -> stability is (Z, EM) already.
    show_surface_or_heatmap(
        z=stability,
        x=em_vals,
        y=Z_vals,
        title="Stability vs Atomic Number and EM Multiplier",
        xlab="EM Multiplier",
        ylab="Atomic Number (Z)",
        zlab="Stability",
        fname_base="Periodic_Table_Stability",
        colorscale="Viridis"
    )

    st.markdown(
        "- Nuclear stability rises with **strong force** (cohesion) and falls with **EM** (Coulomb repulsion).\n"
        "- **Weak force** sets beta-decay efficiency; far from ~1.0 tends to reduce long-lived isotopes."
    )


# -------------------------
# Tab 1: Island of Instability
# -------------------------
with tabs[1]:
    st.subheader("Island of Instability (3D-first)")

    Z_vals = np.linspace(40, 140, res_3d)
    S_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    Z2, S2 = np.meshgrid(Z_vals, S_vals, indexing="ij")

    shell_waves = (0.55 + 0.45*np.sin(Z2 / 7.0)**2) * (0.55 + 0.45*np.sin(Z2 / 11.0)**2)
    strong_opt = np.exp(-np.abs(S2 - S) / 1.6)
    coulomb = np.exp(-np.maximum(Z2 - 60, 0) / (25.0 / max(EM, 1e-3)))

    instability = np.clip((1 - strong_opt) * shell_waves * (1 / np.maximum(coulomb, 1e-6)), 0, 2.0)

    # x=S_vals, y=Z_vals, so z must be (len(Z_vals), len(S_vals)) -> instability is (Z, S) already
    show_surface_or_heatmap(
        z=instability,
        x=S_vals,
        y=Z_vals,
        title="Instability Surface (Z vs Strong Force)",
        xlab="Strong Force Multiplier",
        ylab="Atomic Number (Z)",
        zlab="Instability",
        fname_base="Island_Instability",
        colorscale="Inferno"
    )

    st.markdown(
        "- Synthetic “island of instability”: shell-like periodic structure + force tuning.\n"
        "- Higher **EM** pushes heavy nuclei toward instability; higher **Strong** can counteract it."
    )


# -------------------------
# Tab 2: Star Formation Potential
# -------------------------
with tabs[2]:
    st.subheader("Star Formation Potential (3D-first)")

    g_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    de_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    G2, DE2 = np.meshgrid(g_vals, de_vals, indexing="ij")

    collapse = (G2**1.1) / (DE2**1.2 + 0.15)
    rad_pressure = 1 / (1 + (EM**1.3))
    ignition = np.exp(-np.abs(S - 1.0) * 0.8) * np.exp(-((W - 1.0) ** 2) * 1.2)

    sfr = safe_norm(collapse * rad_pressure * ignition)

    # x=de_vals? or x=g_vals? We'll use x=g_vals, y=de_vals by building z accordingly.
    # sfr currently shape (len(g_vals), len(de_vals)) if indexing="ij" (g, de). Good.
    show_surface_or_heatmap(
        z=sfr,
        x=de_vals,
        y=g_vals,
        title="Star Formation Potential",
        xlab="Dark Energy Multiplier",
        ylab="Gravity Multiplier",
        zlab="Potential",
        fname_base="Star_Formation",
        colorscale="Viridis"
    )

    st.markdown(
        "- Higher **Gravity** promotes collapse; higher **Dark Energy** suppresses structure.\n"
        "- **EM** adds radiative pressure feedback; **Strong/Weak** influence ignition efficiency (proxy)."
    )


# -------------------------
# Tab 3: Life Probability (Heatmap)
# -------------------------
with tabs[3]:
    st.subheader("Life Probability Map (3D-first, iPhone-safe)")

    s_vals = np.linspace(0.1, 10.0, res_3d)
    em_vals = np.linspace(0.1, 10.0, res_3d)
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    force_window = np.exp(-((S2 - 1.0) ** 2) / 1.6) * np.exp(-((EM2 - 1.0) ** 2) / 1.6) * np.exp(-((W - 1.0) ** 2) / 2.0)
    thermo = np.exp(-((T - 1.0) ** 2) * 1.2) * np.exp(-((P - 1.0) ** 2) * 1.0)

    metals = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0)*0.6), 0, 5)
    metals_factor = np.tanh(metals / 1.5)

    life = np.clip(force_window * thermo * metals_factor, 0, 1)

    # x=em_vals, y=s_vals, z must be (len(s_vals), len(em_vals)) -> life is (S, EM) good
    show_surface_or_heatmap(
        z=life,
        x=em_vals,
        y=s_vals,
        title="Life Probability (Proxy)",
        xlab="EM Force Multiplier",
        ylab="Strong Force Multiplier",
        zlab="Life Probability",
        fname_base="Life_Probability",
        colorscale="Plasma"
    )

    st.markdown(
        "- Combines a **force window** (chemistry) + **thermo window** (T/P) + **metallicity proxy** (needs stars).\n"
        "- Proxy model for consistent comparisons (not real-life probability)."
    )


# -------------------------
# Tab 4: Quantum Bonding
# -------------------------
with tabs[4]:
    st.subheader("Quantum Bonding (3D-first)")

    s_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    em_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    S2, EM2 = np.meshgrid(s_vals, em_vals, indexing="ij")

    em_binding = np.exp(-((EM2 - 1.0) ** 2) / 1.4)
    strong_chem = np.exp(-abs(S2 - 1.0) / 2.0)
    temp_kill = np.exp(-((T - 1.0) ** 2) * 1.8)
    pressure_help = np.tanh(P / 1.5)

    bonding = np.clip(em_binding * strong_chem * temp_kill * pressure_help, 0, 1)

    show_surface_or_heatmap(
        z=bonding,
        x=em_vals,
        y=s_vals,
        title="Quantum Bonding (Proxy)",
        xlab="EM",
        ylab="Strong",
        zlab="Bonding",
        fname_base="Quantum_Bonding",
        colorscale="Viridis"
    )

    st.markdown(
        "- Bonding rises near EM≈1 and Strong≈1, but falls at high temperature.\n"
        "- Pressure increases overlap (proxy), raising bonding in this simplified model."
    )


# -------------------------
# Tab 5: Universe Emergence Probability (FIXED to always show)
# -------------------------
with tabs[5]:
    st.subheader("Universe Emergence / Viability Score (3D pillars, always visible)")

    viability = float(np.exp(-0.35 * deviation))
    chaos = float(1.0 - viability)

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
            title="Global Viability Proxy (3D pillars)",
            scene=dict(
                xaxis=dict(title="", tickmode="array", tickvals=[0, 1], ticktext=["Viability", "Chaos"]),
                yaxis_title="",
                zaxis_title="Score",
                aspectmode="auto",
                camera=dict(eye=dict(x=1.8, y=1.5, z=1.0))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        save_plot(fig, "Universe_Viability_3D.png", is_plotly=True)
    else:
        fig2d = go.Figure(data=[go.Bar(x=["Viability", "Chaos/Instability"], y=[viability, chaos])])
        fig2d.update_layout(title="Global Viability Proxy", yaxis_title="Score")
        st.plotly_chart(fig2d, use_container_width=True)
        save_plot(fig2d, "Universe_Viability_2D.png", is_plotly=True)

    st.markdown(
        "- **Viability proxy** based on distance from baseline constants.\n"
        "- Not a physical emergence probability; it’s a compact comparative score."
    )


# -------------------------
# Tab 6: Element Abundance Probability (FIXED)
# -------------------------
with tabs[6]:
    st.subheader("Element Abundance Probability (3D-first + evolution surface)")

    surv = safe_norm(isotope_viable_per_Z.astype(float))
    stellar = np.clip((G / (DE + 0.2)) * np.exp(-abs(EM - 1.0)*0.4) * np.exp(-abs(S - 1.0)*0.3), 0, 3)
    stellar_factor = np.tanh(stellar / 1.3)

    abundance = np.clip(surv * stellar_factor, 0, 1)

    # 3D ribbon for abundance curve vs Z (always visible)
    show_ribbon_or_line(
        x=Z,
        y=abundance,
        title="Element Abundance vs Z (Proxy)",
        xlab="Atomic Number (Z)",
        ylab="Relative Abundance",
        fname_base="Element_Abundance_Line",
        ribbon_width=1.5
    )

    # Evolution surface (downsample for iPhone reliability)
    t = np.linspace(0, 1, max(30, res_3d//2))

    Z_ds = Z[::2]
    abundance_ds = abundance[::2]

    Z2, T2 = np.meshgrid(Z_ds, t, indexing="ij")
    enrich = 1 - np.exp(-T2 * (2.0 + 4.0*stellar_factor))
    abund_time = np.clip((abundance_ds[:, None]) * enrich, 0, 1)

    # x=t, y=Z_ds, z must be (len(Z_ds), len(t)) -> abund_time is (Z, t)
    show_surface_or_heatmap(
        z=abund_time,
        x=t,
        y=Z_ds,
        title="Abundance Evolution (Proxy)",
        xlab="Normalized Cosmic Time",
        ylab="Atomic Number (Z)",
        zlab="Abundance",
        fname_base="Element_Abundance_Evolution",
        colorscale="Cividis"
    )

    st.markdown(
        "- Combines isotope survivability with a crude star-processing/enrichment proxy.\n"
        "- Designed for stable 3D rendering on mobile."
    )


# -------------------------
# Tab 7: EM Radiation Risk
# -------------------------
with tabs[7]:
    st.subheader("EM Radiation Risk (3D ribbon)")

    x = np.linspace(0.1, 10.0, 600)
    y = (x ** 2) * (0.4 + 0.6*np.tanh(T / 2.0)) / 20.0
    y = np.clip(y, 0, 1)

    show_ribbon_or_line(
        x=x, y=y,
        title="Radiation Risk vs EM Multiplier (Proxy)",
        xlab="EM Multiplier",
        ylab="Normalized Risk",
        fname_base="EM_Radiation_Risk",
        ribbon_width=1.2
    )


# -------------------------
# Tab 8: Star Lifespan Model
# -------------------------
with tabs[8]:
    st.subheader("Star Lifespan Model (3D ribbon)")

    g_vals = np.linspace(0.1, 10.0, 400)
    M = g_vals
    L = (M ** 3.5) * (1.0 + 0.15*(EM-1.0))
    tau = (M / np.maximum(L, 1e-9))
    tau = safe_norm(tau)

    show_ribbon_or_line(
        x=g_vals, y=tau,
        title="Relative Stellar Lifetime vs Gravity (Proxy)",
        xlab="Gravity Multiplier",
        ylab="Relative Lifetime",
        fname_base="Star_Lifespan",
        ribbon_width=1.2
    )


# -------------------------
# Tab 9: Dark Matter Simulation
# -------------------------
with tabs[9]:
    st.subheader("Dark Matter / Cosmic Web Proxy (3D + slice)")

    size = 32
    scale = 8.0

    cluster_spread = (3.5 / max(G, 1e-3))
    stretch = (1.0 / max(DE, 1e-3))

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

    # Keep 2D slice as a diagnostic (useful) even in 3D-first mode
    fig2d = go.Figure(data=go.Heatmap(
        z=slice2d.T, x=x, y=y,
        colorscale="Inferno",
        colorbar=dict(title="Density")
    ))
    fig2d.update_layout(title="Cosmic Web Density Slice (2D)", xaxis_title="X", yaxis_title="Y")
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
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)
        save_plot(fig3d, "Dark_Matter_3D.png", is_plotly=True)

    st.markdown("- Higher Gravity compresses structure; higher Dark Energy expands voids (proxy).")


# -------------------------
# Tab 10: Atomic Stability
# -------------------------
with tabs[10]:
    st.subheader("Atomic / Isotope Stability (3D-first)")

    # Downsample for more reliable WebGL
    Zs = Z[::3]
    Ns = N[::3]
    stab_ds = nuclear_stability[::3, ::3]

    show_surface_or_heatmap(
        z=stab_ds,
        x=Ns,
        y=Zs,
        title="Isotope Stability Map (Downsampled)",
        xlab="N (neutrons)",
        ylab="Z (protons)",
        zlab="Stability",
        fname_base="Atomic_Stability",
        colorscale="Plasma"
    )

    st.markdown("- Valley-of-stability proxy: strong helps binding; EM penalizes high-Z; weak shapes beta-decay.")


# -------------------------
# Tab 11: Universe Life Probability Over Time (FIXED & 3D ribbon)
# -------------------------
with tabs[11]:
    st.subheader("Life Probability Over Cosmic Time (3D ribbon)")

    time = np.linspace(0, 1, 200)

    sfr_now = np.clip((G**1.1) / (DE**1.2 + 0.15) * (1 / (1 + EM**1.3)), 0, 10)
    sfr_now = float(np.tanh(sfr_now / 2.0))

    metals = 1 - np.exp(-(2.0 + 5.0*sfr_now) * time)
    metals = np.clip(metals, 0, 1)

    star_window = float(np.exp(-abs(G - 1.0) * 0.6))
    chem_window = float(np.exp(-abs(S - 1.0)*0.4) * np.exp(-abs(EM - 1.0)*0.4) * np.exp(-((W - 1.0)**2)*0.4))
    thermo_window = float(np.exp(-((T - 1.0)**2)*0.6) * np.exp(-((P - 1.0)**2)*0.4))

    life_t = np.clip(metals * star_window * chem_window * thermo_window, 0, 1)

    show_ribbon_or_line(
        x=time,
        y=life_t,
        title="Life Probability Over Time (Proxy)",
        xlab="Normalized Cosmic Time",
        ylab="Life Probability",
        fname_base="Life_Over_Time",
        ribbon_width=1.6
    )

    # Also show metallicity as a second ribbon (helps interpret life curve)
    show_ribbon_or_line(
        x=time,
        y=metals,
        title="Metallicity Proxy Over Time",
        xlab="Normalized Cosmic Time",
        ylab="Metallicity",
        fname_base="Metallicity_Over_Time",
        ribbon_width=1.2
    )


# -------------------------
# Tab 12: Molecular Bonding Model (kept as bars; 3D bars are messy on mobile)
# -------------------------
with tabs[12]:
    st.subheader("Molecular Bonding Viability (element families)")

    isotope_factor = float(np.mean(isotope_viable_per_Z) / np.max(isotope_viable_per_Z))
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
    st.subheader("Molecular Abundance (3D-first)")

    tvals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    pvals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    T2, P2 = np.meshgrid(tvals, pvals, indexing="ij")

    force_gate = float(np.exp(-abs(S - 1.0)*0.4) * np.exp(-abs(EM - 1.0)*0.4))
    thermo_gate = np.exp(-((T2 - 1.0)**2) * 0.8) * np.exp(-((P2 - 1.0)**2) * 0.6)
    abundance = np.clip(force_gate * thermo_gate, 0, 1)

    show_surface_or_heatmap(
        z=abundance,
        x=pvals,
        y=tvals,
        title="Molecular Abundance (Proxy)",
        xlab="Pressure Multiplier",
        ylab="Temperature Multiplier",
        zlab="Abundance",
        fname_base="Molecular_Abundance",
        colorscale="Viridis"
    )


# -------------------------
# Tab 14: Isotope Decay & Half-Life Model
# -------------------------
with tabs[14]:
    st.subheader("Isotope Half-Life Proxy (3D-first + long-lived count ribbon)")

    half_life = np.clip(nuclear_stability * np.exp(-abs(W - 1.0) * 0.6), 0, 1)

    # downsample map for 3D stability
    Zs = Z[::3]
    Ns = N[::3]
    hl_ds = half_life[::3, ::3]

    show_surface_or_heatmap(
        z=hl_ds,
        x=Ns,
        y=Zs,
        title="Half-Life Proxy Map (Downsampled)",
        xlab="N",
        ylab="Z",
        zlab="Half-life (proxy)",
        fname_base="Half_Life",
        colorscale="Cividis"
    )

    long_lived = (half_life > 0.35).sum(axis=1)
    show_ribbon_or_line(
        x=Z,
        y=long_lived.astype(float),
        title="Count of Long-Lived Isotopes per Z (Proxy Threshold)",
        xlab="Z",
        ylab="# long-lived isotopes",
        fname_base="Half_Life_LongLived_Count",
        ribbon_width=1.2
    )


# -------------------------
# Tab 15: Periodic Table Expansion Potential
# -------------------------
with tabs[15]:
    st.subheader("Periodic Table Expansion Potential (3D-first map + curve ribbon)")

    Z_ext = np.arange(1, 201)
    cohesion = np.exp(-np.abs(Z_ext - 82) / (30.0 * max(S, 1e-3)))
    coulomb = 1 / (1 + np.exp(-(Z_ext - 110) / (12.0 / max(EM, 1e-3))))
    decay = np.exp(-((W - 1.0) ** 2) * 1.2)

    stability_curve = np.clip(cohesion * (1 - coulomb) * decay, 0, 1)

    show_ribbon_or_line(
        x=Z_ext,
        y=stability_curve,
        title="Expansion Limit Curve (Proxy)",
        xlab="Z",
        ylab="Stability Potential",
        fname_base="Periodic_Table_Expansion_Curve",
        ribbon_width=1.2
    )

    maxZ = int(Z_ext[stability_curve > 0.12][-1]) if np.any(stability_curve > 0.12) else 0
    st.markdown(f"**Estimated max potentially-stable Z (proxy threshold): {maxZ}**")

    em_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    Z2, EM2 = np.meshgrid(Z_ext, em_vals, indexing="ij")
    cohesion2 = np.exp(-np.abs(Z2 - 82) / (30.0 * max(S, 1e-3)))
    coulomb2 = 1 / (1 + np.exp(-(Z2 - 110) / (12.0 / np.maximum(EM2, 1e-3))))
    stab2 = np.clip(cohesion2 * (1 - coulomb2) * decay, 0, 1)

    show_surface_or_heatmap(
        z=stab2,
        x=em_vals,
        y=Z_ext,
        title="Expansion Potential Map (Z vs EM)",
        xlab="EM",
        ylab="Z",
        zlab="Stability",
        fname_base="Periodic_Table_Expansion_Map",
        colorscale="Viridis"
    )


# -------------------------
# Tab 16: Proton–Neutron Ratio Heatmap
# -------------------------
with tabs[16]:
    st.subheader("Proton–Neutron Ratio / Valley of Stability (3D-first)")

    # downsample for 3D
    Zs = Z[::3]
    Ns = N[::3]
    vi_ds = nuclear_stability[::3, ::3]

    show_surface_or_heatmap(
        z=vi_ds,
        x=Ns,
        y=Zs,
        title="Viability in (Z, N) (Downsampled)",
        xlab="N",
        ylab="Z",
        zlab="Viability",
        fname_base="PN_Viability",
        colorscale="Magma"
    )

    target_line = (1.0 + (Z / 80.0)) * Z
    show_ribbon_or_line(
        x=Z,
        y=target_line,
        title="Valley-of-Stability Target Curve (Proxy)",
        xlab="Z",
        ylab="Target N",
        fname_base="PN_Target_Curve",
        ribbon_width=1.2
    )


# -------------------------
# Tab 17: Nuclear Binding Energy Map
# -------------------------
with tabs[17]:
    st.subheader("Nuclear Binding Energy (SEMF-style proxy, 3D-first)")

    a_v = 15.8 * S
    a_s = 18.3
    a_c = 0.714 * EM
    a_sym = 23.2 * (1 / max(W, 1e-3))
    a_pair = 12.0

    A = Ag
    pairing = np.where(((Zg % 2 == 0) & (Ng % 2 == 0)), +1, -1)

    BE = (
        a_v * A
        - a_s * (A ** (2/3))
        - a_c * (Zg * (Zg - 1)) / np.maximum(A ** (1/3), 1e-9)
        - a_sym * ((A - 2*Zg) ** 2) / np.maximum(A, 1e-9)
        + pairing * a_pair / np.maximum(A ** 0.5, 1e-9)
    )

    BE_per_A = np.clip(BE / np.maximum(A, 1), 0, None)
    BE_per_A = np.clip(BE_per_A, 0, np.nanpercentile(BE_per_A, 99))

    # downsample for 3D stability
    Zs = Z[::3]
    Ns = N[::3]
    be_ds = BE_per_A[::3, ::3]

    show_surface_or_heatmap(
        z=be_ds,
        x=Ns,
        y=Zs,
        title="Binding Energy per Nucleon (Downsampled)",
        xlab="N",
        ylab="Z",
        zlab="BE/A (proxy)",
        fname_base="Binding_Energy",
        colorscale="Viridis"
    )

    target_line = (1.0 + (Z / 80.0)) * Z
    N_pick = np.clip(target_line.astype(int), 1, N[-1])
    idxN = N_pick - 1
    be_line = BE_per_A[np.arange(len(Z)), idxN]

    show_ribbon_or_line(
        x=Z,
        y=be_line,
        title="BE/A along Valley-of-Stability (Proxy)",
        xlab="Z",
        ylab="BE/A",
        fname_base="Binding_Energy_Line",
        ribbon_width=1.2
    )

    peakZ = int(Z[np.nanargmax(be_line)])
    st.markdown(f"**Peak along valley at Z ≈ {peakZ} (proxy)**")


# -------------------------
# Tab 18: Multiverse Decoherence Map
# -------------------------
with tabs[18]:
    st.subheader("Multiverse Decoherence Map (3D-first, mid-time slice)")

    s_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    em_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    t_vals = np.linspace(0, 1, max(30, res_3d//2))

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

    show_surface_or_heatmap(
        z=mid,
        x=em_vals,
        y=s_vals,
        title="Mid-Time Coherence Slice",
        xlab="EM",
        ylab="Strong",
        zlab="Coherence",
        fname_base="Decoherence_MidTime",
        colorscale="Magma"
    )

    st.markdown("- Toy Everett-style visualization: coherence decays with parameter distance and time.")


# -------------------------
# Tab 19: Quantum Branch Count Estimator
# -------------------------
with tabs[19]:
    st.subheader("Quantum Branch Count Estimator (3D ribbon + 3D surface)")

    steps = 200
    t = np.linspace(0, 1, steps)

    rate = (S * EM * G) / (DE * (W + 0.2))
    rate = float(np.clip(rate, 0.05, 10.0))

    branches = np.exp((0.8 * rate) * t * (1.0 + 0.25*deviation))
    branches = np.clip(branches, 1, 1e18)

    show_ribbon_or_line(
        x=t,
        y=np.log10(branches + 1e-9),
        title="Branch Count vs Time (log10)",
        xlab="Normalized Time",
        ylab="log10(Branch Count)",
        fname_base="Branch_Count_Log",
        ribbon_width=1.2
    )

    d_vals = np.linspace(0, 6, max(30, res_3d//2))
    TT, DD = np.meshgrid(t[::3], d_vals, indexing="ij")
    branches2 = np.exp((0.8 * rate) * TT * (1.0 + 0.25*DD))
    branches2 = np.clip(branches2, 1, 1e18)

    show_surface_or_heatmap(
        z=np.log10(branches2 + 1e-9),
        x=d_vals,
        y=t[::3],
        title="Branch Growth vs Deviation and Time (log10)",
        xlab="Deviation",
        ylab="Time",
        zlab="log10(branches)",
        fname_base="Branch_Count_Surface",
        colorscale="Viridis"
    )


# -------------------------
# Tab 20: Quantum Gravity Horizon Map
# -------------------------
with tabs[20]:
    st.subheader("Quantum Gravity Horizon Map (3D-first)")

    r_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    g_vals = np.linspace(0.1, 10.0, max(30, res_3d//2))
    R, GG = np.meshgrid(r_vals, g_vals, indexing="ij")

    curvature = (GG * G) / (R + 1e-6)
    curvature *= (1.0 / (1.0 + 0.25*DE))
    curvature = np.clip(curvature, 0, 2.0)

    # x=r, y=g, z must be (len(g), len(r)) if x=r and y=g; currently curvature is (r,g) due to ij
    # Let's transpose by rebuilding into (g,r):
    curvature_gr = curvature.T  # (g, r)

    show_surface_or_heatmap(
        z=curvature_gr,
        x=r_vals,
        y=g_vals,
        title="Curvature Map (Proxy)",
        xlab="Radial coordinate r (proxy)",
        ylab="Gravitational field multiplier",
        zlab="Curvature",
        fname_base="Quantum_Gravity",
        colorscale="Cividis"
    )

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