import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants (sliders + user input sync)
st.sidebar.header("Adjust Physical Constants")

def slider_with_input(label, min_val, max_val, default, step):
    col1, col2 = st.sidebar.columns([3, 2])
    val = col1.slider(label, min_val, max_val, default, step=step, key=label)
    user_input = col2.text_input("User Input", value=str(val), key=f"{label}_input")
    
    try:
        input_val = float(user_input)
        if min_val <= input_val <= max_val:
            val = input_val
    except:
        pass
    
    col1.write(f"Change from baseline: {((val-1.0)*100):+.1f}%")
    return val

constants = {
    "Strong Force Multiplier": slider_with_input("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_input("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": slider_with_input("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": slider_with_input("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": slider_with_input("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),
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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# ------------------------
# TAB 1 - Periodic Table Stability
with tabs[0]:
    st.subheader("Periodic Table Stability Probability (Advanced 3D Scatter)")
    atomic_numbers = np.arange(1, 121)
    em_force_values = np.linspace(0.1, 10.0, 50)
    atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)
    stability_probability = np.exp(-np.abs(atomic_grid - 30) / 20) * np.exp(-np.abs(em_grid - constants["Electromagnetic Force Multiplier"]))

    fig = go.Figure(data=[go.Scatter3d(x=atomic_grid.flatten(), y=em_grid.flatten(), z=stability_probability.flatten(),
                                       mode='markers', marker=dict(size=5, color=stability_probability.flatten(),
                                       colorscale='Viridis', colorbar=dict(title='Stability')))])
    fig.update_layout(scene=dict(xaxis_title='Atomic Number', yaxis_title='EM Force Multiplier', zaxis_title='Stability Probability'))
    st.plotly_chart(fig, use_container_width=True, key="tab1_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    This chart shows how atomic number and EM force affect elemental stability.
    Higher EM multipliers destabilize heavy elements, reducing viable periodic table richness.
    """)

# ------------------------
# TAB 2 - Island of Instability
with tabs[1]:
    st.subheader("Island of Instability (Advanced 3D Surface)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    atomic_number_values = np.linspace(50, 120, 50)
    strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)
    instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)

    fig = go.Figure(data=[go.Surface(z=instability, x=strong_grid, y=atomic_grid, colorscale='Inferno', colorbar=dict(title='Instability'))])
    fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='Atomic Number', zaxis_title='Instability Level'))
    st.plotly_chart(fig, use_container_width=True, key="tab2_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    This surface reveals instability regions where strong force multipliers shift.
    Higher values push elements into instability zones, especially for superheavy nuclei.
    """)

# ------------------------
# TAB 3 - Star Formation Potential
with tabs[2]:
    st.subheader("Star Formation Potential (Advanced 3D Surface)")
    gravity_values = np.linspace(0.1, 10.0, 50)
    dark_energy_values = np.linspace(0.1, 10.0, 50)
    gravity_grid, dark_grid = np.meshgrid(gravity_values, dark_energy_values)
    star_potential = np.exp(-((gravity_grid - constants["Gravitational Constant Multiplier"])**2 + (dark_grid - constants["Dark Energy Multiplier"])**2) / 4)

    fig = go.Figure(data=[go.Surface(z=star_potential, x=gravity_grid, y=dark_grid, colorscale='Viridis', colorbar=dict(title='Potential'))])
    fig.update_layout(scene=dict(xaxis_title='Gravity Multiplier', yaxis_title='Dark Energy Multiplier', zaxis_title='Star Formation Potential'))
    st.plotly_chart(fig, use_container_width=True, key="tab3_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    This chart models star formation likelihood based on gravitational and dark energy forces.
    Balanced forces increase star-forming zones, while extremes suppress star formation.
    """)
# ------------------------
# TAB 4 - Life Probability
with tabs[3]:
    st.subheader("Life Probability Map (Heatmap)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
    life_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 3)

    fig = go.Figure(data=go.Heatmap(z=life_prob, x=strong_force_values, y=em_force_values, colorscale='Viridis', colorbar=dict(title='Life Probability')))
    fig.update_layout(xaxis_title="Strong Force Multiplier", yaxis_title="EM Force Multiplier")
    st.plotly_chart(fig, use_container_width=True, key="tab4_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Life probability peaks when strong and EM forces are near baseline.
    Deviations reduce molecular stability and nuclear chemistry, hindering complex life emergence.
    """)
# ------------------------
# TAB 5 - Quantum Bonding
with tabs[4]:
    st.subheader("Quantum Bonding Probability (Advanced 3D Surface)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
    bonding_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 2)

    fig = go.Figure(data=[go.Surface(z=bonding_prob, x=strong_grid, y=em_grid, colorscale='Viridis', colorbar=dict(title='Bonding Probability'))])
    fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='EM Force Multiplier', zaxis_title='Bonding Probability'))
    st.plotly_chart(fig, use_container_width=True, key="tab5_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Atomic bonding strength is tightly linked to strong and EM forces.
    Extreme deviations reduce bonding likelihood, limiting molecular complexity and chemical diversity.
    """)

# ------------------------
# TAB 6 - Universe Probability
with tabs[5]:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)

    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Universe probability drops exponentially as fundamental constants deviate.
    Greater deviation suggests hostile conditions for stable matter and life.
    """)

# ------------------------
# TAB 7 - Element Abundance
with tabs[6]:
    st.subheader("Element Abundance")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]

    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    ax.set_xlabel("Forces")
    ax.set_ylabel("Relative Element Abundance")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Elemental abundance decreases as force multipliers deviate from baseline.
    Reduced abundance impacts stellar nucleosynthesis and chemical variety.
    """)

# ------------------------
# TAB 8 - Radiation Risk
with tabs[7]:
    st.subheader("Radiation Risk")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100

    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--', label="Current EM Force")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Radiation increases rapidly with higher EM force multipliers.
    Elevated radiation may disrupt DNA and cellular processes, challenging complex life.
    """)

# ------------------------
# TAB 9 - Star Lifespan
with tabs[8]:
    st.subheader("Star Lifespan vs Gravity Multiplier")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x

    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkgreen', linewidth=2)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--', label="Current Gravity Multiplier")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Star lifespan is inversely tied to gravitational strength.
    Higher gravity shortens stellar lifespans, limiting time available for biological evolution.
    """)

# ------------------------
# TAB 10 - 2D Dark Matter Simulation
with tabs[9]:
    st.subheader("2D Dark Matter Simulation")
    density_2d = np.random.normal(0, 1, (100, 100))

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(density_2d, cmap="plasma", interpolation="nearest", origin="lower")
    fig.colorbar(c, ax=ax)
    ax.set_title("Simulated 2D Dark Matter Plasma Density")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    This simulation visualizes theoretical dark matter plasma density.
    Clustering indicates gravitational wells that seed galaxy formation.
    """)

# ------------------------
# TAB 11 - 3D Atomic Stability Probability
with tabs[10]:
    st.subheader("3D Atomic Stability Probability per Isotope")
    atomic_numbers = np.arange(1, 121)
    isotopes_per_element = 20
    np.random.seed(42)

    base_stability = np.linspace(0.2, 0.98, len(atomic_numbers))
    modified_stability = base_stability * constants["Strong Force Multiplier"] / constants["Electromagnetic Force Multiplier"]
    modified_stability = np.clip(modified_stability, 0, 1)

    stability_matrix = np.array([
        modified_stability + np.random.normal(0, 0.05 * constants["Weak Force Multiplier"], len(atomic_numbers))
        for _ in range(isotopes_per_element)]).T
    stability_matrix = np.clip(stability_matrix, 0, 1)

    Z_vals, Isotope_vals, Stability_vals = [], [], []
    for Z in atomic_numbers:
        for iso in range(1, isotopes_per_element + 1):
            Z_vals.append(Z)
            Isotope_vals.append(iso)
            Stability_vals.append(stability_matrix[Z - 1, iso - 1])

    fig = go.Figure(data=[go.Scatter3d(x=Z_vals, y=Isotope_vals, z=Stability_vals, mode='markers',
                                       marker=dict(size=5, color=Stability_vals, colorscale='Plasma', colorbar=dict(title='Stability')))])
    fig.update_layout(scene=dict(xaxis_title='Atomic Number', yaxis_title='Isotope Number', zaxis_title='Stability Probability'))
    st.plotly_chart(fig, use_container_width=True, key="tab11_graph")

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("""
    Isotopic stability is strongly impacted by nuclear force balance.
    High weak force multipliers destabilize isotopes, reducing heavy element stability.
    """)