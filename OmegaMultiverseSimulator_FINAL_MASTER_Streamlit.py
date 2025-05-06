import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator PRO — Scientific Master Edition")

# ----- Sidebar Sliders for Physical Constants -----
st.sidebar.header("Adjust Physical Constants")

constants = {
    "Strong Force Multiplier": st.sidebar.slider("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Electromagnetic Force Multiplier": st.sidebar.slider("Electromagnetic Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Weak Force Multiplier": st.sidebar.slider("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.1),
    "Gravitational Constant Multiplier": st.sidebar.slider("Gravitational Constant Multiplier", 0.1, 10.0, 1.0, 0.1),
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

# ----- Graph Tabs -----
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

# ---- Tab 1: Stability Curve ----
with tab1:
    st.subheader("Element Stability vs Strong Force Multiplier")
    x = np.linspace(0.5, 2.0, 500)
    y = np.exp(-((x - constants["Strong Force Multiplier"])**2)/0.02)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Strong Force Multiplier")
    ax.set_ylabel("Relative Stability")
    st.pyplot(fig)

# ---- Tab 2: Periodic Table Stability (Scientific Bar Graph) ----
with tab2:
    st.subheader("Periodic Table Stability Probability (Scientific)")
    
    element_symbols = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Li", "Be", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne", "", ""],
        ["Na", "Mg", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar", "", ""],
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ]
    
    force_factor = deviation
    fig, ax = plt.subplots(figsize=(10, 6))

    for period in element_symbols:
        for symbol in period:
            if symbol != "":
                atomic_number = len(st.session_state) + 1
                probability = np.exp(-abs(force_factor) * (atomic_number / 20))
                ax.bar(symbol, probability, color="green" if probability > 0.5 else "orange")
    
    ax.set_ylabel("Stability Probability")
    ax.set_title("Element Stability Probability Across Periodic Table")
    ax.set_ylim(0, 1.1)
    st.pyplot(fig)

# ---- Tab 3: Island of Instability ----
with tab3:
    st.subheader("Island of Instability")
    x = np.linspace(0.5, 2.0, 500)
    y = np.abs(np.sin((x - constants["Strong Force Multiplier"]) * 5))
    fig, ax = plt.subplots()
    ax.plot(x, y, color='red')
    ax.set_xlabel("Strong Force Multiplier")
    ax.set_ylabel("Instability Intensity")
    st.pyplot(fig)

# ---- Tab 4: Universe Probability ----
with tab4:
    st.subheader("Universe Formation Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ---- Tab 5: Star Formation ----
with tab5:
    st.subheader("Star Formation Potential")
    x = np.linspace(0.1, 10.0, 500)
    y = np.exp(-((x - constants["Gravitational Constant Multiplier"])**2)/1.0)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Gravitational Constant Multiplier")
    ax.set_ylabel("Star Formation Probability")
    st.pyplot(fig)

# ---- Tab 6: Life Probability ----
with tab6:
    st.subheader("Life Probability (based on deviation)")
    life_prob = np.exp(-deviation/2)
    fig, ax = plt.subplots()
    ax.bar(["Life Probability"], [life_prob])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ---- Tab 7: Element Abundance ----
with tab7:
    st.subheader("Element Abundance vs Forces")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance)
    ax.set_ylabel("Relative Abundance")
    st.pyplot(fig)

# ---- Tab 8: Radiation Risk ----
with tab8:
    st.subheader("Radiation Risk vs Electromagnetic Force")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("EM Force Multiplier")
    ax.set_ylabel("Radiation Risk Level")
    st.pyplot(fig)

# ---- Tab 9: Quantum Bonding Probability ----
with tab9:
    st.subheader("Quantum Bonding Probability")
    bonding = np.exp(-abs(constants["Strong Force Multiplier"] - 1))
    fig, ax = plt.subplots()
    ax.bar(["Bonding Probability"], [bonding])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ---- Tab 10: Star Lifespan ----
with tab10:
    st.subheader("Star Lifespan vs Gravity Multiplier")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("Gravitational Constant Multiplier")
    ax.set_ylabel("Relative Star Lifespan")
    st.pyplot(fig)

# ---- Tab 11: Dark Matter 3D Expansion ----
with tab11:
    st.subheader("3D Dark Matter Expansion")

    num_galaxies = 100
    dark_energy = constants["Dark Energy Multiplier"]
    np.random.seed(42)
    positions = np.random.normal(0, 50, (num_galaxies, 3))
    expanded = positions * (1 + dark_energy)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(expanded[:,0], expanded[:,1], expanded[:,2], c='purple', alpha=0.7)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    st.pyplot(fig)