
import streamlit as st
import numpy as np
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# 3D Graph Optimizer (prevents cutoff)

def optimize_3d_layout(fig, x_title, y_title, z_title):
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            xaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10)),
            yaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10)),
            zaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10))
        ),
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig


st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator PRO — ULTRA OPTIMIZED VERSION")

# Sidebar - Universe Constants
st.sidebar.header("Adjust Physical Constants")
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
    st.success("This universe matches our own. Chemistry and life likely stable.")
elif deviation < 3:
    st.warning("Moderate deviation detected. Instability possible.")
else:
    st.error("High deviation. Unstable universe likely.")

st.divider()

# Tab setup (lazy loading in each tab)
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])



# ======================================
# Universal 3D Graph Optimizer Function
# ======================================

def optimize_3d_layout(fig, x_title, y_title, z_title):
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            xaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10)),
            yaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10)),
            zaxis=dict(titlefont=dict(size=12), tickfont=dict(size=10))
        ),
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig


# ======================================
# Scientific Graphs (Optimized and Patched)
# ======================================


# Periodic Table Stability Probability (3D Scatter)
with tabs[0]:
    st.subheader("Periodic Table Stability Probability (Advanced 3D Scatter)")
    with st.spinner("Generating Periodic Table Graph..."
fig = optimize_3d_layout(fig, "Atomic Number", "Stability", "Probability"):
            atomic_numbers = np.arange(1, 121)
            em_force_values = np.linspace(0.1, 10.0, 50)
            atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)
            stability_probability = np.exp(-np.abs(atomic_grid - 30) / 20) * np.exp(-np.abs(em_grid - constants["Electromagnetic Force Multiplier"]))
            fig = go.Figure(data=[go.Scatter3d(x=atomic_grid.flatten(), y=em_grid.flatten(), z=stability_probability.flatten(),
                                               mode='markers', marker=dict(size=5, color=stability_probability.flatten(),
                                               colorscale='Viridis', colorbar=dict(title='Stability')))])
            fig.update_layout(scene=dict(xaxis_title='Atomic Number', yaxis_title='EM Force Multiplier', zaxis_title='Stability Probability'))
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)

# Island of Instability (3D Surface)
with tabs[1]:
    st.subheader("Island of Instability (Advanced 3D Surface)")
    with st.spinner("Generating Instability Graph..."):
            strong_force_values = np.linspace(0.1, 10.0, 50)
            atomic_number_values = np.linspace(50, 120, 50)
            strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)
            instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)
            fig = go.Figure(data=[go.Surface(z=instability, x=strong_grid, y=atomic_grid, colorscale='Inferno', colorbar=dict(title='Instability'))])
            fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='Atomic Number', zaxis_title='Instability Level'))
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)

# Star Formation Potential (3D Surface)
with tabs[2]:
    st.subheader("Star Formation Potential (Advanced 3D Surface)")
    with st.spinner("Generating Star Formation Graph..."):
            gravity_values = np.linspace(0.1, 10.0, 50)
            dark_energy_values = np.linspace(0.1, 10.0, 50)
            gravity_grid, dark_grid = np.meshgrid(gravity_values, dark_energy_values)
            star_potential = np.exp(-((gravity_grid - constants["Gravitational Constant Multiplier"])**2 + (dark_grid - constants["Dark Energy Multiplier"])**2) / 4)
            fig = go.Figure(data=[go.Surface(z=star_potential, x=gravity_grid, y=dark_grid, colorscale='Viridis', colorbar=dict(title='Potential'))])
            fig.update_layout(scene=dict(xaxis_title='Gravity Multiplier', yaxis_title='Dark Energy Multiplier', zaxis_title='Star Formation Potential'))
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)

# Life Probability (Heatmap)
with tabs[3]:
    st.subheader("Life Probability Map (Heatmap)")
    with st.spinner("Generating Life Probability Map..."):
            strong_force_values = np.linspace(0.1, 10.0, 50)
            em_force_values = np.linspace(0.1, 10.0, 50)
            strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
            life_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 3)
            fig = go.Figure(data=go.Heatmap(z=life_prob, x=strong_force_values, y=em_force_values, colorscale='Viridis', colorbar=dict(title='Life Probability')))
            fig.update_layout(xaxis_title="Strong Force Multiplier", yaxis_title="EM Force Multiplier")
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)



# Quantum Bonding (3D Surface)
with tabs[4]:
    st.subheader("Quantum Bonding Probability (Advanced 3D Surface)")
    with st.spinner("Generating Quantum Bonding Graph..."
fig = optimize_3d_layout(fig, "Strong Force Multiplier", "EM Force Multiplier", "Bonding Probability"):
            strong_force_values = np.linspace(0.1, 10.0, 50)
            em_force_values = np.linspace(0.1, 10.0, 50)
            strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)
            bonding_prob = np.exp(-((strong_grid - constants["Strong Force Multiplier"])**2 + (em_grid - constants["Electromagnetic Force Multiplier"])**2) / 2)
            fig = go.Figure(data=[go.Surface(z=bonding_prob, x=strong_grid, y=em_grid, colorscale='Viridis', colorbar=dict(title='Bonding Probability'))])
            fig.update_layout(scene=dict(xaxis_title='Strong Force Multiplier', yaxis_title='EM Force Multiplier', zaxis_title='Bonding Probability'))
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)

# Universe Probability
with tabs[5]:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    st.pyplot(fig)

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

# 2D Dark Matter Simulation
with tabs[9]:
    st.subheader("2D Dark Matter Simulation")
    with st.spinner("Generating Dark Matter Map..."):
            density_2d = np.random.normal(0, 1, (100, 100))
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.imshow(density_2d, cmap="plasma", interpolation="nearest", origin="lower")
            fig.colorbar(c, ax=ax)
            ax.set_title("Simulated 2D Dark Matter Plasma Density")
            st.pyplot(fig)

# 3D Atomic Stability Probability
with tabs[10]:
    st.subheader("3D Atomic Stability Probability per Isotope")
    with st.spinner("Generating Atomic Stability 3D Graph..."
fig = optimize_3d_layout(fig, "Atomic Number", "Isotope Number", "Stability Probability"):
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
            fig = optimize_3d_layout(fig, "X Axis", "Y Axis", "Z Axis")
    st.plotly_chart(fig, use_container_width=True)

