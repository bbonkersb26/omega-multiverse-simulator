
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator PRO — Scientific Master Edition")

# ----- Sidebar Sliders for Physical Constants -----
st.sidebar.header("Adjust Physical Constants")

constants = {
    "Strong Force Multiplier": st.sidebar.slider("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Electromagnetic Force Multiplier": st.sidebar.slider("EM Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Weak Force Multiplier": st.sidebar.slider("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Gravitational Constant Multiplier": st.sidebar.slider("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Dark Energy Multiplier": st.sidebar.slider("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.1),
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "Stability Curve",
    "Periodic Table Probability",
    "Island of Instability",
    "Universe Probability",
    "Star Formation",
    "Life Probability",
    "Element Abundance",
    "Radiation Risk",
    "Quantum Bonding",
    "Star Lifespan",
    "3D Dark Matter Expansion",
    "3D Atomic Stability (New!)"
])

with tab1:
    st.subheader("Element Stability vs Strong Force Multiplier")
    x = np.linspace(0.5, 2.0, 500)
    y = np.exp(-((x - constants["Strong Force Multiplier"])**2)/0.02)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', linewidth=2)
    st.pyplot(fig)

with tab2:
    st.subheader("Periodic Table Stability Probability")
    element_numbers = np.arange(1, 31)
    probabilities = np.exp(-abs(deviation) * (element_numbers / 20))
    fig, ax = plt.subplots()
    ax.bar(element_numbers, probabilities, color='cyan', edgecolor='black')
    st.pyplot(fig)

with tab3:
    st.subheader("Island of Instability")
    x = np.linspace(0.5, 2.0, 500)
    y = np.abs(np.sin((x - constants["Strong Force Multiplier"]) * 5))
    fig, ax = plt.subplots()
    ax.plot(x, y, color='red', linewidth=2)
    st.pyplot(fig)

with tab4:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    st.pyplot(fig)

with tab5:
    st.subheader("Star Formation Potential")
    x = np.linspace(0.1, 10.0, 500)
    y = np.exp(-((x - constants["Gravitational Constant Multiplier"])**2)/1.0)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='orange', linewidth=2)
    st.pyplot(fig)

with tab6:
    st.subheader("Life Probability")
    life_prob = np.exp(-deviation/2)
    fig, ax = plt.subplots()
    ax.bar(["Life Probability"], [life_prob], color='green')
    st.pyplot(fig)

with tab7:
    st.subheader("Element Abundance")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    st.pyplot(fig)



with tab8:
    st.subheader("Radiation Risk")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--', label="Current EM Force")
    ax.set_xlabel("Electromagnetic Force Multiplier")
    ax.set_ylabel("Radiation Risk")
    ax.legend()
    st.pyplot(fig)

with tab9:
    st.subheader("Quantum Bonding Probability")
    bonding_prob = np.exp(-abs(constants["Strong Force Multiplier"] - 1))
    fig, ax = plt.subplots()
    ax.bar(["Bonding Probability"], [bonding_prob], color='violet')
    ax.set_ylabel("Probability")
    st.pyplot(fig)

with tab10:
    st.subheader("Star Lifespan vs Gravity Multiplier")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkgreen', linewidth=2)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--', label="Current Gravity Multiplier")
    ax.set_xlabel("Gravitational Constant Multiplier")
    ax.set_ylabel("Relative Star Lifespan")
    ax.legend()
    st.pyplot(fig)


with tab11:
    st.subheader("Dark Matter Plasma Web (Projected)")

    grid_size = 50
    density = np.random.normal(0, 1, (grid_size, grid_size))

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(density, cmap='plasma', interpolation='nearest')
    ax.set_title("Dark Matter Plasma Density Projection")
    plt.colorbar(c, ax=ax, label='Density')

    st.pyplot(fig)


with tab12:
    st.subheader("3D Atomic Stability Probability per Isotope (Dynamic Universe Constants)")

    atomic_numbers = np.arange(1, 121)
    isotopes_per_element = 20

    np.random.seed(42)
    base_stability = np.linspace(0.2, 0.98, len(atomic_numbers))

    modified_stability = base_stability * constants["Strong Force Multiplier"] / constants["Electromagnetic Force Multiplier"]
    modified_stability = np.clip(modified_stability, 0, 1)

    stability_matrix = np.array([modified_stability + np.random.normal(0, 0.05 * constants["Weak Force Multiplier"], len(atomic_numbers))
                                 for _ in range(isotopes_per_element)]).T
    stability_matrix = np.clip(stability_matrix, 0, 1)

    Z_vals = []
    Stability_vals = []
    Isotope_vals = []
    Labels = []

    for Z in atomic_numbers:
        for iso in range(1, isotopes_per_element + 1):
            Z_vals.append(Z)
            Stability_vals.append(stability_matrix[Z - 1, iso - 1])
            Isotope_vals.append(iso)
            Labels.append(f"Z{Z}-Iso{iso}")

    fig3d = go.Figure()

    fig3d.add_trace(go.Scatter3d(
        x=Z_vals,
        y=Isotope_vals,
        z=Stability_vals,
        mode='markers',
        marker=dict(
            size=5,
            color=Stability_vals,
            colorscale='Plasma',
            opacity=0.9,
            colorbar=dict(title="Stability")
        ),
        text=Labels,
        hovertemplate="Element: %{text}<br>Atomic Number: %{x}<br>Isotope: %{y}<br>Stability: %{z:.2f}<extra></extra>"
    ))

    fig3d.update_layout(
        title="3D Atomic Stability Probability per Isotope",
        scene=dict(
            xaxis_title='Atomic Number',
            yaxis_title='Isotope Number',
            zaxis_title='Stability Probability'
        ),
        width=900,
        height=700
    )

    st.plotly_chart(fig3d, use_container_width=True)
