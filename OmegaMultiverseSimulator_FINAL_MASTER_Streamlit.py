import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")

# ---- HEADER ----
st.title("Omega Multiverse Simulator FINAL MASTER SCIENTIFIC EDITION + AI MODULE")
st.write("Explore scientifically-grounded universes with AI enhanced explanations and advanced graphing.")

# ---- PRESET STANDARD MODEL CONSTANTS ----
standard_values = {
    "Strong Force Multiplier": 1.0,
    "Electromagnetic Force Multiplier": 1.0,
    "Weak Force Multiplier": 1.0,
    "Gravitational Constant Multiplier": 1.0,
    "Dark Energy Multiplier": 1.0
}

st.sidebar.header("Physical Constants Sliders")

# ---- USER INPUT SLIDERS ----
constants = {}
for const, default in standard_values.items():
    constants[const] = st.sidebar.slider(const, 0.1, 10.0, default, 0.1)

# ---- UNIVERSE ANALYSIS ----
st.header("Universe Simulation Outputs")

# Simple example analysis
st.subheader("Standard Model Deviation")
deviation = sum(abs(constants[c] - standard_values[c]) for c in constants)
st.write(f"Total deviation from our universe: **{deviation:.2f}**")

# ---- AI Analysis (Starter AI Explanation Module) ----
st.subheader("AI Analysis of Universe Stability")
if deviation == 0:
    st.success("This universe matches ours → Chemistry and Life are likely stable.")
elif deviation < 3:
    st.warning("This universe is slightly different → Some chemical and biological systems may destabilize.")
else:
    st.error("This universe is highly unstable → Life and stable chemistry likely impossible.")

# ---- PERIODIC TABLE (Simple AI Output for now) ----
st.subheader("Periodic Table Stability AI Commentary")
if deviation < 2:
    st.write("Periodic table expected to be mostly stable → heavy elements like Uranium still viable.")
else:
    st.write("Heavy elements may not form → island of stability collapses.")

# ---- SAMPLE GRAPH (Force Multiplier Graph) ----
st.subheader("Force Multiplier Effects")

forces = list(constants.keys())
values = list(constants.values())

fig, ax = plt.subplots()
ax.bar(forces, values, color='cyan')
ax.set_ylabel("Multiplier Value")
ax.set_title("Current Universe Constants")
plt.xticks(rotation=45)

st.pyplot(fig)

st.write("---")
st.write("More advanced AI graphing modules will be added in next version.")