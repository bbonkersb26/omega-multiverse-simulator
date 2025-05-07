
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants
st.sidebar.header("Adjust Physical Constants")
def slider_with_percent(label, min_value, max_value, default_value, step):
    col1, col2 = st.sidebar.columns([3, 1])
    value = col1.slider(label, min_value, max_value, default_value, step=step)
    percent_change = ((value - 1.0) * 100)
    col2.markdown(f"<small>Change: {percent_change:.2f}%</small>", unsafe_allow_html=True)
    precise_input = st.sidebar.text_input(f"User Input for {label}", str(value))
    try:
        precise_value = float(precise_input)
        if min_value <= precise_value <= max_value:
            value = precise_value
    except:
        pass
    return value

constants = {
    "Strong Force Multiplier": slider_with_percent("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_percent("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": slider_with_percent("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": slider_with_percent("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": slider_with_percent("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),
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

# Tabs setup
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)",
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Periodic Table Stability (3D)
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
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### AI Analysis → Scientific Summary")
    st.write("This chart shows how atomic number and EM force affect elemental stability. Higher EM multipliers destabilize heavy elements, reducing viable periodic table richness.")

# --- Continue Tabs (starting from tab1) ---

# Periodic Table Stability Probability (3D Scatter)
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
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This 3D scatter visualizes how atomic number and electromagnetic force affect element stability. Higher atomic numbers usually reduce stability, but fine-tuned EM force values can increase bonding possibilities.")

# Island of Instability (3D Surface)
with tabs[1]:
    st.subheader("Island of Instability (Advanced 3D Surface)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    atomic_number_values = np.linspace(50, 120, 50)
    strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)
    instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)
    fig = go.Figure(data=[go.Surface(z=instability, x=strong_grid, y=atomic_grid, colorscale='Inferno', colorbar=dict(title='Instability'))])
    fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='Atomic Number', zaxis_title='Instability Level'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This surface graph explores nuclear instability regions. As the strong force multiplier shifts, peaks and valleys show zones where nuclei are more or less prone to decay or collapse.")

# Star Formation Potential (3D Surface)
with tabs[2]:
    st.subheader("Star Formation Potential (Advanced 3D Surface)")
    gravity_values = np.linspace(0.1, 10.0, 50)
    dark_energy_values = np.linspace(0.1, 10.0, 50)
    gravity_grid, dark_grid = np.meshgrid(gravity_values, dark_energy_values)
    star_potential = np.exp(-((gravity_grid - constants["Gravitational Constant Multiplier"])**2 + (dark_grid - constants["Dark Energy Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=star_potential, x=gravity_grid, y=dark_grid, colorscale='Viridis', colorbar=dict(title='Potential'))])
    fig.update_layout(scene=dict(xaxis_title='Gravity Multiplier', yaxis_title='Dark Energy Multiplier', zaxis_title='Star Formation Potential'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("Star formation is influenced by gravity and dark energy. This simulation shows that optimal values lead to efficient star birth, while too much or too little makes star formation improbable.")

# Life Probability (Heatmap)
with tabs[3]:
    st.subheader("Life Probability Map (Heatmap)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
    life_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 3)
    fig = go.Figure(data=go.Heatmap(z=life_prob, x=strong_force_values, y=em_force_values, colorscale='Viridis', colorbar=dict(title='Life Probability')))
    fig.update_layout(xaxis_title="Strong Force Multiplier", yaxis_title="EM Force Multiplier")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("Life potential depends on finely tuned forces. This heatmap highlights how moderate values enable molecular bonding crucial for biology, while extremes render life chemically unfeasible.")

# Quantum Bonding (3D Surface)
with tabs[4]:
    st.subheader("Quantum Bonding Probability (Advanced 3D Surface)")
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
    bonding_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 2)
    fig = go.Figure(data=[go.Surface(z=bonding_prob, x=strong_grid, y=em_grid, colorscale='Viridis', colorbar=dict(title='Bonding Probability'))])
    fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='EM Force Multiplier', zaxis_title='Bonding Probability'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("Chemical bonding relies on the interplay of nuclear forces. This graph shows optimal force regions where atoms can form molecules efficiently, which is essential for chemistry and life.")


# Universe Probability
with tabs[5]:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    st.pyplot(fig)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This simple probability graph reflects how closely this universe matches our own. High deviation reduces stability, while low deviation means a chemically and physically friendly universe.")

# Element Abundance
with tabs[6]:
    st.subheader("Element Abundance")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    st.pyplot(fig)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This bar graph represents how force changes alter elemental formation. Stable universes produce rich elements, while extreme forces cause scarcity or instability in matter itself.")

# Radiation Risk
with tabs[7]:
    st.subheader("Radiation Risk")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--', label="Current EM Force")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("Higher electromagnetic forces increase radiation hazard. This curve shows how dramatically radiation risks increase as EM values grow, threatening molecular integrity and biology.")

# Star Lifespan
with tabs[8]:
    st.subheader("Star Lifespan vs Gravity Multiplier")
    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkgreen', linewidth=2)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--', label="Current Gravity Multiplier")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("Gravity plays a direct role in star longevity. This inverse graph shows how stronger gravitational multipliers shorten star lifespans, while weaker gravity extends them, affecting planetary ecosystems.")

# 2D Dark Matter Simulation
with tabs[9]:
    st.subheader("2D Dark Matter Simulation")
    density_2d = np.random.normal(0, 1, (100, 100))
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(density_2d, cmap="plasma", interpolation="nearest", origin="lower")
    fig.colorbar(c, ax=ax)
    ax.set_title("Simulated 2D Dark Matter Plasma Density")
    st.pyplot(fig)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This simulated heatmap displays random dark matter density fluctuations. High density areas suggest gravitational wells where galaxies could form, affecting universe structure and evolution.")

# 3D Atomic Stability Probability
with tabs[10]:
    st.subheader("3D Atomic Stability Probability per Isotope")
    atomic_numbers = np.arange(1, 121)
    isotopes_per_element = 20
    np.random.seed(42)
    base_stability = np.linspace(0.2, 0.98, len(atomic_numbers))
    modified_stability = base_stability * constants["Strong Force Multiplier"] / constants["Electromagnetic Force Multiplier"]
    modified_stability = np.clip(modified_stability, 0, 1)
    stability_matrix = np.array([modified_stability + np.random.normal(0, 0.05 * constants["Weak Force Multiplier"], len(atomic_numbers)) for _ in range(isotopes_per_element)]).T
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
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This advanced 3D isotope view reveals how strong and electromagnetic force multipliers shift atomic stability. Stable isotope regions support extended periodic tables in alternate universes.")
