import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator PRO — AI Ultra Graphs Edition")
st.write("Explore scientifically-grounded universes and visualize advanced scientific scenarios.")

# ---- Sidebar GUI ----
st.sidebar.header("Physics Constants")

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
    st.success("This universe matches ours. Chemistry and life likely stable.")
elif deviation < 3:
    st.warning("Moderate deviations detected. Instability possible.")
else:
    st.error("High deviation. Stable chemistry and life unlikely.")

st.divider()

# ---- Create proper tabs (Fixed version) ----
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Stability Curve",
    "Periodic Table Stability",
    "Island of Instability",
    "Universe Probability",
    "Star Formation",
    "Life Probability",
    "Element Abundance",
    "Radiation Risk",
    "Quantum Bonding Probability",
    "Star Lifespan",
    "3D Dark Matter Expansion"
])

# ---- Graph Tabs ----
with tab1:
    st.subheader("Element Stability vs Strong Force")
    x = np.linspace(0.5, 2.0, 500)
    y = np.exp(-((x - constants["Strong Force Multiplier"])**2)/0.02)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

with tab2:
    st.subheader("Periodic Table Stability")
    stable_count = int(max(0, 30 - deviation * 5))
    st.write("Stable elements estimated:", stable_count)
    cols = st.columns(10)
    for i in range(30):
        cols[i % 10].markdown(f"Element {i+1}: {'✅' if i < stable_count else '❌'}")

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
    ax.bar(["Probability"], [prob])
    st.pyplot(fig)

with tab5:
    st.subheader("Star Formation vs Gravity")
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
    st.subheader("Element Abundance vs Forces")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance)
    st.pyplot(fig)

with tab8:
    st.subheader("Radiation Risk vs EM Force")
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
    st.subheader("Star Lifespan vs Gravity")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--')
    st.pyplot(fig)

with tab11:
    st.subheader("3D Dark Matter Expansion Simulation")

    num_galaxies = 100
    dark_energy = constants["Dark Energy Multiplier"]
    np.random.seed(42)
    positions = np.random.normal(0, 50, (num_galaxies, 3))
    expanded = positions * (1 + dark_energy)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(expanded[:,0], expanded[:,1], expanded[:,2], c='purple', alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    st.pyplot(fig)

st.divider()

st.header("AI Scientific Analysis")
if deviation < 1:
    st.success("Universe likely viable for chemistry and stars.")
elif deviation < 3:
    st.warning("Universe marginal → unusual element formation possible.")
else:
    st.error("Universe extremely unstable → likely no stars or chemistry.")

st.write("Simulation complete.")