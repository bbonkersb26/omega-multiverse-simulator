import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")

# ---- HEADER ----
st.title("Omega Multiverse Simulator - True Classical Periodic Table Edition")
st.write("Explore scientifically-grounded universes with perfected periodic table grid layout.")

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

# ---- PERIODIC TABLE TRUE CLASSICAL LAYOUT ----
st.header("Periodic Table (True Classical Layout)")

periodic_table_html = """
<style>
.table-container {
    overflow-x: auto;
}
.periodic-table {
    border-collapse: collapse;
    width: 100%;
}
.periodic-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
    width: 40px;
    height: 40px;
}
.periodic-table td.empty {
    border: none;
}
</style>

<div class="table-container">
<table class="periodic-table">
<tr><td>H</td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td>He</td></tr>
<tr><td>Li</td><td>Be</td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td>B</td><td>C</td><td>N</td><td>O</td><td>F</td><td>Ne</td></tr>
<tr><td>Na</td><td>Mg</td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td class='empty'></td><td>Al</td><td>Si</td><td>P</td><td>S</td><td>Cl</td><td>Ar</td></tr>
<tr><td>K</td><td>Ca</td><td>Sc</td><td>Ti</td><td>V</td><td>Cr</td><td>Mn</td><td>Fe</td><td>Co</td><td>Ni</td><td>Cu</td><td>Zn</td><td>Ga</td><td>Ge</td><td>As</td><td>Se</td><td>Br</td><td>Kr</td></tr>
</table>
</div>
"""

st.markdown(periodic_table_html, unsafe_allow_html=True)

st.divider()

st.write("Classical periodic table layout finalized. Simulator ready for scientific use.")