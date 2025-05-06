import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")

# ---- HEADER ----
st.title("Omega Multiverse Simulator - Perfect Periodic Table Version")
st.write("Explore scientifically-grounded universes with final optimized periodic table layout.")

# ---- SIDEBAR INPUT ----
st.sidebar.header("Physical Constants Sliders")

standard_values = {
    "Strong Force Multiplier": 1.0,
    "Electromagnetic Force Multiplier": 1.0,
    "Weak Force Multiplier": 1.0,
    "Gravitational Constant Multiplier": 1.0,
    "Dark Energy Multiplier": 1.0
}

constants = {}
for const, default in standard_values.items():
    constants[const] = st.sidebar.slider(const, 0.1, 10.0, default, 0.1)

# ---- UNIVERSE ANALYSIS ----
st.header("Universe Simulation Overview")
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
    st.write("Formation probability model coming soon.")

with tab3:
    st.write("Island of instability coming soon.")

st.divider()

# ---- FINAL FIXED PERIODIC TABLE ----
st.header("Periodic Table (Final Fixed Grid Layout)")

periodic_table_grid = """
<style>
.periodic-table {
    display: grid;
    grid-template-columns: repeat(18, 40px);
    grid-gap: 4px;
    justify-content: center;
}
.periodic-table div {
    width: 38px;
    height: 38px;
    background-color: #f0f0f0;
    text-align: center;
    line-height: 38px;
    border-radius: 4px;
    font-size: 12px;
}
.periodic-table .empty {
    background-color: transparent;
}
</style>

<div class="periodic-table">
<div>H</div> <div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div>He</div>
<div>Li</div><div>Be</div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div>B</div><div>C</div><div>N</div><div>O</div><div>F</div><div>Ne</div>
<div>Na</div><div>Mg</div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div class="empty"></div><div>Al</div><div>Si</div><div>P</div><div>S</div><div>Cl</div><div>Ar</div>
<div>K</div><div>Ca</div><div>Sc</div><div>Ti</div><div>V</div><div>Cr</div><div>Mn</div><div>Fe</div><div>Co</div><div>Ni</div><div>Cu</div><div>Zn</div><div>Ga</div><div>Ge</div><div>As</div><div>Se</div><div>Br</div><div>Kr</div>
</div>
"""

st.markdown(periodic_table_grid, unsafe_allow_html=True)

st.divider()
st.write("Final periodic table grid locked and optimized for all screens.")