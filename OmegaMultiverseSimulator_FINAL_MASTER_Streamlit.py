
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation (Verbose Scientific Dev Version)")

# Sidebar - Universe Constants
st.sidebar.header("Adjust Physical Constants")
def render_slider(label, min_val, max_val, default, step):
    val = st.sidebar.slider(label, min_val, max_val, default, step=step)
    change_percent = (val - 1.0) * 100
    st.sidebar.markdown(f"<small>Change from baseline: {change_percent:+.2f}%</small>", unsafe_allow_html=True)
    return val

constants = {
    "Strong Force Multiplier": render_slider("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": render_slider("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": render_slider("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": render_slider("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": render_slider("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),
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

tabs = st.tabs(["Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)",
                "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
                "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"])

with tabs[0]:
    st.subheader("Periodic Table Stability (3D)")

atomic_numbers = np.arange(1, 121)
em_force_values = np.linspace(0.1, 10.0, 50)
atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)
stability_probability = np.exp(-np.abs(atomic_grid - 30) / 20) * np.exp(-np.abs(em_grid - constants["Electromagnetic Force Multiplier"]))
fig = go.Figure(data=[go.Scatter3d(x=atomic_grid.flatten(), y=em_grid.flatten(), z=stability_probability.flatten(),
                                   mode='markers', marker=dict(size=5, color=stability_probability.flatten(),
                                   colorscale='Viridis', colorbar=dict(title='Stability')))])
fig.update_layout(scene=dict(xaxis_title='Atomic Number', yaxis_title='EM Force Multiplier', zaxis_title='Stability Probability'))
st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This simulation visualizes how element stability varies across the periodic table as the electromagnetic force multiplier changes. Higher EM force favors tighter atomic binding, stabilizing lighter elements, while reducing stability of heavier atoms.")

with tabs[1]:
    st.subheader("Island of Instability (3D)")

strong_force_values = np.linspace(0.1, 10.0, 50)
atomic_number_values = np.linspace(50, 120, 50)
strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)
instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)
fig = go.Figure(data=[go.Surface(z=instability, x=strong_grid, y=atomic_grid, colorscale='Inferno', colorbar=dict(title='Instability'))])
fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='Atomic Number', zaxis_title='Instability Level'))
st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This graph highlights the sensitivity of heavy element nuclei to the strong nuclear force multiplier. Small changes can destabilize or stabilize heavy isotopes, impacting superheavy element formation.")

with tabs[2]:
    st.subheader("Star Formation Potential (3D)")

gravity_values = np.linspace(0.1, 10.0, 50)
dark_energy_values = np.linspace(0.1, 10.0, 50)
gravity_grid, dark_grid = np.meshgrid(gravity_values, dark_energy_values)
star_potential = np.exp(-((gravity_grid - constants["Gravitational Constant Multiplier"])**2 + (dark_grid - constants["Dark Energy Multiplier"])**2) / 4)
fig = go.Figure(data=[go.Surface(z=star_potential, x=gravity_grid, y=dark_grid, colorscale='Viridis', colorbar=dict(title='Potential'))])
fig.update_layout(scene=dict(xaxis_title='Gravity Multiplier', yaxis_title='Dark Energy Multiplier', zaxis_title='Star Formation Potential'))
st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Star formation relies on gravitational collapse balanced by dark energy. This graph models how these factors affect the ability of galaxies to form stars.")
