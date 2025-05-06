import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")

# ---- HEADER ----
st.title("Omega Multiverse Simulator - Scientific Layout Edition")
st.write("Explore universes with scientifically-grounded constants and cleaner visuals.")

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

# ---- PERIODIC TABLE GRID (BETTER LAYOUT) ----
st.header("Periodic Table Grid (Simplified View)")

elements = ["H", "He", "Li", "Be", "Na", "Mg", "K", "Ca", "Cu"]
cols = st.columns(5)

for index, element in enumerate(elements):
    cols[index % 5].button(element)

st.divider()

st.write("Simulator running with updated clean layout. Further AI upgrades paused pending approval.")