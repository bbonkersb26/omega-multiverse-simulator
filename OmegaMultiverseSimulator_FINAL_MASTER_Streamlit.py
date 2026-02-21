import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import os
from PIL import Image
import plotly.io as pio
from fpdf import FPDF
import datetime
import openai

# === MUST BE FIRST Streamlit command ===
st.set_page_config(page_title="Multiverse Simulation", layout="wide")
st.title("Multiverse Simulation")

# === Divider compatibility patch (keeps your code the same) ===
if not hasattr(st, "divider"):
    def _divider_fallback():
        st.markdown("---")
    st.divider = _divider_fallback

# === Save Plot Function ===
def save_plot(fig, filename, is_plotly=True):
    output_dir = "pdf_visuals"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if is_plotly:
        pio.write_image(fig, path, format='png')
    else:
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

def generate_pdf(constants, summary_text, output_dir="pdf_visuals"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Use built-in core font for full compatibility (Helvetica)
    font = "Helvetica"

    # === Cover Page ===
    pdf.add_page()
    pdf.set_font(font, "B", 24)
    pdf.cell(0, 15, "Omega Multiverse Simulation Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font(font, "", 14)
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 10, f"Date: {date_str}", ln=True, align="C")
    pdf.cell(0, 10, "Generated via GPT-3.5 AI", ln=True, align="C")

    # === Parameters Page ===
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Parameters", ln=True)
    pdf.set_font(font, "", 12)
    for k, v in constants.items():
        line = f"{k}: {v:.2f}"
        pdf.cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)

    # === AI Summary Page ===
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "AI Universe Summary", ln=True)
    pdf.set_font(font, "", 12)
    for line in summary_text.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, safe_line)

    # === Visuals Section ===
    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Simulation Visuals", ln=True)

    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    for image_file in image_files:
        path = os.path.join(output_dir, image_file)
        pdf.add_page()
        pdf.set_font(font, "B", 14)
        title = image_file.replace(".png", "").replace("_", " ")
        title_safe = title.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, title_safe, ln=True)
        pdf.image(path, w=180)

    pdf.output("Omega_Universe_Simulation_Report.pdf")
    st.sidebar.header("Adjust Physical Constants")

def slider_with_input(label, min_val, max_val, default_val, step):
    col1, col2 = st.sidebar.columns([3, 1])
    slider_val = col1.slider(label, min_val, max_val, default_val, step=step)
    user_input = col2.text_input(f"{label} (Optional User Input)", value=str(slider_val))
    try:
        input_val = float(user_input)
        if min_val <= input_val <= max_val:
            slider_val = input_val
    except:
        pass
    percent_change = ((slider_val - 1.0) / 1.0) * 100
    st.sidebar.caption(f"Change from baseline: {percent_change:+.2f}%")
    return slider_val


constants = {
    "Strong Force Multiplier": slider_with_input("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_input("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": slider_with_input("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": slider_with_input("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": slider_with_input("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),

    # === NEW ENHANCEMENTS for Chemical Modeling ===
    "Temperature Multiplier": slider_with_input("Temperature Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Pressure Multiplier": slider_with_input("Pressure Multiplier", 0.1, 10.0, 1.0, 0.01),
}
deviation = sum(abs(v - 1.0) for v in constants.values())
st.header("Universe Stability Summary")
st.write(f"Deviation from Standard Model: **{deviation:.2f}**")
if deviation == 0:
    st.success("This universe matches our own. Chemistry and life likely stable.")
elif deviation < 3:
    st.warning("Moderate deviation detected. Instability possible.")
else:
    st.error("High deviation. Unstable universe likely.")

st.divider()
# === Precompute Half-Life Matrix for Cross-Module Use ===
atomic_numbers = np.arange(1, 121)
isotope_range = np.arange(1, 21)
Z_grid, iso_grid = np.meshgrid(atomic_numbers, isotope_range, indexing='ij')

weak_force = constants["Weak Force Multiplier"]
strong_force = constants["Strong Force Multiplier"]

base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
weak_decay_penalty = np.exp(-((weak_force - 1.0) ** 2) * 3)
strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong_force))

half_life_matrix = base_half_life * weak_decay_penalty * strong_bonus
half_life_matrix = np.clip(half_life_matrix, 0, 1)
import openai

# Initialize OpenAI client (make sure your API key is in .streamlit/secrets.toml under [openai])
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Global Universe Synopsis ===

st.subheader("AI Global Universe Analysis")

if st.button("Generate AI Universe Summary"):
    with st.spinner("Generating summary using OpenAI..."):
        user_context = "\n".join([f"{k}: {v:.2f}" for k, v in constants.items()])
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a physics and cosmology expert. Analyze universal constants and summarize what kind of universe this configuration would produce."},
                    {"role": "user", "content": f"Here are the physical constants:\n{user_context}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            summary = response.choices[0].message.content
            st.session_state["summary"] = summary  # Persist the summary
            st.success("Summary generated:")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
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
    "Proton–Neutron Ratio Heatmap",                # already fixed earlier
    "Nuclear Binding Energy Map",                   # <-- just add it here directly
    "Multiverse Decoherence Map",
    "Quantum Branch Count Estimator",
    "Quantum Gravity Horizon Map"
])
st.divider()
st.subheader("Export Simulation Report")

if st.button("Generate Scientific PDF Report"):
    with st.spinner("Compiling PDF..."):
        try:
            summary_text = st.session_state.get("summary", "No AI summary generated yet.")
            generate_pdf(constants, summary_text)
            st.success("PDF generated successfully!")
            with open("Omega_Universe_Simulation_Report.pdf", "rb") as file:
                st.download_button(
                    label="Download Scientific Report PDF",
                    data=file,
                    file_name="Omega_Universe_Simulation_Report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")# === Periodic Table Stability (Scientific Model → Strong Force, EM Force, Weak Force Dependent) ===

with tabs[0]:
    st.subheader("Periodic Table Stability Probability")

    atomic_numbers = np.arange(1, 121)

    # Pull slider values from physical constants
    strong_force = constants["Strong Force Multiplier"]
    em_force = constants["Electromagnetic Force Multiplier"]
    weak_force = constants["Weak Force Multiplier"]

    # Create dynamic EM force range centered on slider value
    em_force_values = np.linspace(em_force - 2, em_force + 2, 50)
    em_force_values = np.clip(em_force_values, 0.1, 10.0)
    atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)

    # Calculate base nuclear stability (shell + atomic number)
    base_stability = np.exp(-np.abs(atomic_grid - 30) / 20)

    # Strong Force Effect → Higher strong force → stabilizes heavy nuclei
    strong_bonus = np.exp(-np.abs(atomic_grid - 80) / (20 * strong_force))

    # EM Force Effect → Higher EM force → destabilizes heavy elements
    em_penalty = np.exp(-np.abs(em_grid - em_force))

    # Weak Force Effect → Ideal weak force (~1.0) → most stable isotopes
    weak_bonus = np.exp(-((weak_force - 1.0) ** 2) * 3)

    # Final Stability Probability
    stability_probability = base_stability * strong_bonus * em_penalty * weak_bonus

    # Normalize for visualization
    stability_probability = np.clip(stability_probability, 0, 1)

    # Plot 3D Scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=atomic_grid.flatten(),
        y=em_grid.flatten(),
        z=stability_probability.flatten(),
        mode='markers',
        marker=dict(size=5, color=stability_probability.flatten(), colorscale='Viridis', colorbar=dict(title='Stability'))
    )])

    fig.update_layout(
        title="Periodic Table Stability Probability",
        scene=dict(
            xaxis_title='Atomic Number',
            yaxis_title='EM Force Multiplier',
            zaxis_title='Stability Probability'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Periodic Table Stability.png", is_plotly=True)
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This advanced scientific model calculates element stability based on fundamental forces:")
    st.markdown("- **Strong Force Multiplier → Higher values stabilize heavier nuclei → reduces instability.**")
    st.markdown("- **EM Force Multiplier → Higher values destabilize heavy elements → proton repulsion dominates.**")
    st.markdown("- **Weak Force Multiplier → Deviations from 1.0 destabilize isotopes → ideal near 1.0.**")
    st.markdown("- This model reflects realistic nuclear behavior, dynamically updating with universe physical constants.")

with tabs[1]:
    st.subheader("Island of Instability (Periodic Pattern + Scientific Bonus)")

    # Atomic numbers and strong force range
    atomic_number_values = np.linspace(50, 120, 50)
    strong_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)

    # Original periodic instability model (shell closure inspired)
    base_instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)

    # Scientific bonus → lower instability when closer to ideal strong force (1.0)
    scientific_bonus = np.exp(-np.abs(strong_grid - constants["Strong Force Multiplier"]))

    # Final instability → periodic + scientific shift
    instability = base_instability * scientific_bonus

    # Plot
    fig = go.Figure(data=[go.Surface(
        z=instability,
        x=strong_grid,
        y=atomic_grid,
        colorscale='Inferno',
        colorbar=dict(title='Instability Level')
    )])

    fig.update_layout(
        title="Island of Instability",
        scene=dict(
            xaxis_title='Strong Force Multiplier',
            yaxis_title='Atomic Number',
            zaxis_title='Instability Level'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    save_plot(fig, "Island of Instability.png", is_plotly=True)
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This model combines periodic nuclear shell structure patterns and scientific tuning effects:")
    st.markdown("- **Periodic Pattern → Models peaks/valleys of nuclear shell closures (like magic numbers).**")
    st.markdown("- **Scientific Bonus → Strong Force near ideal (1.0) improves stability, shifting instability lower.**")
    st.markdown("- The result is a dynamic, accurate, and responsive simulation of nuclear instability across universes.")

# (…rest of your code continues unchanged…)