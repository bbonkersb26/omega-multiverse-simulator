# Omega Multiverse Simulator PRO — Scientific Final Version

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
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
    "3D Dark Matter Expansion"
])

with tab1:
    st.subheader("Element Stability vs Strong Force Multiplier")
    x = np.linspace(0.5, 2.0, 500)
    y = np.exp(-((x - constants["Strong Force Multiplier"])**2)/0.02)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

with tab2:
    st.subheader("Periodic Table Stability Probability")
    element_numbers = np.arange(1, 31)
    probabilities = np.exp(-abs(deviation) * (element_numbers / 20))
    fig, ax = plt.subplots()
    ax.bar(element_numbers, probabilities, color='cyan', edgecolor='black')
    ax.set_xlabel("Atomic Number")
    ax.set_ylabel("Stability Probability")
    st.pyplot(fig)

with tab3:
    st.subheader("Island of Instability")
    x = np.linspace(0.5, 2.0, 500)
    y = np.abs(np.sin((x - constants["Strong Force Multiplier"]) * 5))
    fig, ax = plt.subplots()
    ax.plot(x, y, color='r')
    st.pyplot(fig)

with tab4:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob])
    st.pyplot(fig)

with tab5:
    st.subheader("Star Formation Potential")
    x = np.linspace(0.1, 10.0, 500)
    y = np.exp(-((x - constants["Gravitational Constant Multiplier"])**2)/1.0)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

with tab6:
    st.subheader("Life Probability")
    life_prob = np.exp(-deviation/2)
    fig, ax = plt.subplots()
    ax.bar(["Life Probability"], [life_prob])
    st.pyplot(fig)

with tab7:
    st.subheader("Element Abundance")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance)
    st.pyplot(fig)

with tab8:
    st.subheader("Radiation Risk")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--')
    st.pyplot(fig)

with tab9:
    st.subheader("Quantum Bonding Probability")
    bonding = np.exp(-abs(constants["Strong Force Multiplier"] - 1))
    fig, ax = plt.subplots()
    ax.bar(["Bonding Probability"], [bonding])
    st.pyplot(fig)

with tab10:
    st.subheader("Star Lifespan vs Gravity Multiplier")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--')
    st.pyplot(fig)

with tab11:
    st.subheader("3D Dark Matter Plasma Web")
    grid_size = 50
    density = np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    grid = pv.UniformGrid()
    grid.dimensions = np.array(density.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_arrays["density"] = density.flatten(order="F")
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(grid, cmap="plasma", opacity="sigmoid")
    plotter.screenshot("darkmatter.png")
    st.image("darkmatter.png", caption="Dark Matter Plasma Simulation")
