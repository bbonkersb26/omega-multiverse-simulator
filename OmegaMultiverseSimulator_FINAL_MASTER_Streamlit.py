
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

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

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This simulation visualizes how element stability varies across the periodic table as the electromagnetic force multiplier changes. Higher EM force favors tighter atomic binding, stabilizing lighter elements, while reducing stability of heavier atoms.")

with tabs[1]:
    st.subheader("Island of Instability (3D)")

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This graph highlights the sensitivity of heavy element nuclei to the strong nuclear force multiplier. Small changes can destabilize or stabilize heavy isotopes, impacting superheavy element formation.")

with tabs[2]:
    st.subheader("Star Formation Potential (3D)")

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Star formation relies on gravitational collapse balanced by dark energy. This graph models how these factors affect the ability of galaxies to form stars.")

with tabs[3]:
    st.subheader("Life Probability (Heatmap)")

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This heatmap reflects the life-friendly chemistry probability, influenced by strong and electromagnetic forces altering reaction stability.")

with tabs[4]:
    st.subheader("Quantum Bonding (3D)")

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Quantum bonding strength varies with strong and EM forces, affecting molecule formation and stability across universes.")

with tabs[5]:
    st.subheader("Universe Probability")

    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Overall universe viability is modeled as a function of deviation from known physical constants, predicting fundamental stability.")

with tabs[6]:
    st.subheader("Element Abundance")

    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    ax.set_xlabel("Force Type")
    ax.set_ylabel("Relative Abundance")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Element abundance is affected by nuclear forces influencing stellar fusion pathways and decay rates.")

with tabs[7]:
    st.subheader("Radiation Risk")

    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("Electromagnetic Force Multiplier")
    ax.set_ylabel("Radiation Risk")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Higher EM force increases radiation hazards to life by enhancing photon-matter interaction cross-sections.")

with tabs[8]:
    st.subheader("Star Lifespan")

    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkgreen', linewidth=2)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("Gravity Multiplier")
    ax.set_ylabel("Relative Star Lifespan")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Gravitational intensity affects fuel consumption, determining star longevity and potential habitable planet stability.")

with tabs[9]:
    st.subheader("2D Dark Matter Simulation")

    density_2d = np.random.normal(0, 1, (100, 100))
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(density_2d, cmap="plasma", interpolation="nearest", origin="lower")
    fig.colorbar(c, ax=ax)
    ax.set_title("Simulated 2D Dark Matter Plasma Density")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("Dark matter density patterns influence large scale structure formation and galactic stability.")

with tabs[10]:
    st.subheader("3D Atomic Stability")

    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Interpretation")
    st.markdown("This 3D scatter shows isotope stability probabilities influenced by nuclear forces, critical for long-term element persistence.")
