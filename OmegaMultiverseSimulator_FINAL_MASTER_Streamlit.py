
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation — True Fixed Version")

# Sidebar - Universe Constants
st.sidebar.header("Adjust Physical Constants")
def render_slider(label, min_val, max_val, default, step):
    val = st.sidebar.slider(label, min_val, max_val, default, step=step)
    change_percent = (val - 1.0) * 100
    st.sidebar.markdown(f"<small>Change from baseline: {change_percent:+.2f}%</small>", unsafe_allow_html=True)
    return val

constants = {
    "Strong Force Multiplier": render_slider("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": render_slider("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": render_slider("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": render_slider("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": render_slider("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),
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

tabs = st.tabs(["Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)",
                "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
                "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"])

with tabs[0]:
    st.subheader("Periodic Table Stability (3D)")


    atomic_numbers = np.arange(1, 121)
    em_force_values = np.linspace(0.1, 10.0, 50)
    atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)
    stability_probability = np.exp(-np.abs(atomic_grid - 30) / 20) * np.exp(-np.abs(em_grid - constants["Electromagnetic Force Multiplier"]))
    fig = go.Figure(data=[go.Scatter3d(x=atomic_grid.flatten(), y=em_grid.flatten(), z=stability_probability.flatten(),
                                       mode='markers', marker=dict(size=5, color=stability_probability.flatten(),
                                       colorscale='Viridis', colorbar=dict(title='Stability')))])
    fig.update_layout(scene=dict(xaxis_title='Atomic Number', yaxis_title='EM Force Multiplier', zaxis_title='Stability Probability'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("This simulation visualizes how element stability varies across the periodic table as the electromagnetic force multiplier changes.")

with tabs[1]:
    st.subheader("Island of Instability (3D)")


    strong_force_values = np.linspace(0.1, 10.0, 50)
    atomic_number_values = np.linspace(50, 120, 50)
    strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)
    instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)
    fig = go.Figure(data=[go.Surface(z=instability, x=strong_grid, y=atomic_grid, colorscale='Inferno', colorbar=dict(title='Instability'))])
    fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='Atomic Number', zaxis_title='Instability Level'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("This graph highlights the sensitivity of heavy element nuclei to the strong nuclear force multiplier.")

with tabs[2]:
    st.subheader("Star Formation Potential (3D)")


    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Star formation depends critically on the balance between gravitational attraction and dark energy repulsion.")

with tabs[3]:
    st.subheader("Life Probability (Heatmap)")


    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Life-supporting chemistry relies on a careful balance of forces.")

with tabs[4]:
    st.subheader("Quantum Bonding (3D)")


    x = np.linspace(0.1, 10.0, 50)
    y = np.linspace(0.1, 10.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - constants["Strong Force Multiplier"])**2 + (Y - constants["Electromagnetic Force Multiplier"])**2) / 4)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Quantum bonding probabilities are influenced by nuclear and electromagnetic forces.")

with tabs[5]:
    st.subheader("Universe Probability")


    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    ax.set_xlabel("Universe Stability")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Overall universe stability reflects how constants deviate from our own.")

with tabs[6]:
    st.subheader("Element Abundance")


    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    ax.set_xlabel("Forces")
    ax.set_ylabel("Relative Abundance")
    st.pyplot(fig)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Relative abundance of elements depends on force magnitudes influencing nuclear fusion.")

with tabs[7]:
    st.subheader("Radiation Risk")


    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("EM Force Multiplier")
    ax.set_ylabel("Radiation Risk")
    st.pyplot(fig)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Electromagnetic force intensity affects radiation interaction with matter.")

with tabs[8]:
    st.subheader("Star Lifespan")


    x = np.linspace(0.1, 10.0, 500)
    y = 1 / x
    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkgreen', linewidth=2)
    ax.axvline(constants["Gravitational Constant Multiplier"], color='r', linestyle='--')
    ax.set_xlabel("Gravity Multiplier")
    ax.set_ylabel("Relative Star Lifespan")
    st.pyplot(fig)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Gravity affects star pressure and fuel consumption.")

with tabs[9]:
    st.subheader("2D Dark Matter Simulation")


    density_2d = np.random.normal(0, 1, (100, 100))
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(density_2d, cmap="plasma", interpolation="nearest", origin="lower")
    fig.colorbar(c, ax=ax)
    ax.set_title("Simulated 2D Dark Matter Plasma Density")
    st.pyplot(fig)

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("Dark matter clumping affects cosmic web structure.")

with tabs[10]:
    st.subheader("3D Atomic Stability")


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

    st.markdown("### AI Analysis — Scientific Interpretation")
    st.markdown("This 3D scatter shows isotope stability probabilities affected by nuclear forces.")
