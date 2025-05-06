import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")

# ---- HEADER ----
st.title("Omega Multiverse Simulator - Classical Periodic Table Edition")
st.write("Explore scientifically-grounded universes with improved classical periodic table layout.")

# ---- PRESET STANDARD MODEL CONSTANTS ----
standard_values = {
    "Strong Force Multiplier": 1.0,
    "Electromagnetic Force Multiplier": 1.0,
    "Weak Force Multiplier": 1.0,
    "Gravitational Constant Multiplier": 1.0,
    "Dark Energy Multiplier": 1.0
}

# ---- SIDEBAR INPUT ----
st.sidebar.header("Physical Constants Sliders")

constants = {}
for const, default in standard_values.items():
    constants[const] = st.sidebar.slider(const, 0.1, 10.0, default, 0.1)

# ---- UNIVERSE ANALYSIS ----
st.header("Universe Simulation Overview")
with st.container():
    deviation = sum(abs(constants[c] - standard_values[c]) for c in constants)
    st.subheader("Standard Model Deviation")
    st.write(f"**Deviation from our universe:** {deviation:.2f}")

    if deviation == 0:
        st.success("Perfect Match → Chemistry & Life Stable")
    elif deviation < 3:
        st.warning("Slightly Different → Some Instabilities Possible")
    else:
        st.error("Highly Unstable → Chemistry and Life unlikely")

st.divider()

# ---- SCIENTIFIC GRAPH ANALYSIS ----
st.header("Scientific Graph Analysis")
tab1, tab2, tab3 = st.tabs(["Stability Curve", "Formation Probability", "Island of Instability"])

with tab1:
    st.subheader("Element Stability vs Strong Force")
    x = np.linspace(0.5, 2.0, 500)
    stability = np.exp(-((x - 1.0) ** 2) / 0.01)

    fig, ax = plt.subplots()
    ax.plot(x, stability)
    ax.set_xlabel("Strong Force Multiplier")
    ax.set_ylabel("Stability (Relative)")
    ax.set_title("Element Stability vs Strong Force")

    st.pyplot(fig)

with tab2:
    st.write("Formation probability model will be added in expansion pack.")

with tab3:
    st.write("Island of instability visualization coming soon.")

st.divider()

# ---- PERIODIC TABLE CLASSICAL LAYOUT ----
st.header("Periodic Table (Classical Layout)")

# Define periodic table rows (using gaps as "")
periodic_table_rows = [
    ["H", "", "", "", "", "", "", "He"],
    ["Li", "Be", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"]
]

for row in periodic_table_rows:
    cols = st.columns(len(row))
    for i, element in enumerate(row):
        if element != "":
            cols[i].button(element, key=f"{element}_{i}")
        else:
            cols[i].empty()  # Add empty space

st.divider()

st.write("Classical periodic table layout applied. Simulator ready for scientific evaluation.")