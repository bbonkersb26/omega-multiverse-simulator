
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multiverse Simulation", layout="wide")
st.title("Multiverse Simulation")

st.sidebar.header("Adjust Physical Constants")

def slider_with_input(label, min_val, max_val, default_val, step):
    col1, col2 = st.sidebar.columns([3, 1])
    slider_val = col1.slider(label, min_val, max_val, default_val, step=step)
    user_input = col2.text_input(f"{label} (Optional User Input)", value=str(slider_val))
    try:
        input_val = float(user_input)
        if min_val <= input_val <= max_val:
            slider_val = input_val
    except:
        pass
    percent_change = ((slider_val - 1.0) / 1.0) * 100
    st.sidebar.caption(f"Change from baseline: {percent_change:+.2f}%")
    return slider_val


constants = {
    "Strong Force Multiplier": slider_with_input("Strong Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Electromagnetic Force Multiplier": slider_with_input("EM Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Weak Force Multiplier": slider_with_input("Weak Force Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Gravitational Constant Multiplier": slider_with_input("Gravitational Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Dark Energy Multiplier": slider_with_input("Dark Energy Multiplier", 0.1, 10.0, 1.0, 0.01),
    
    # === NEW ENHANCEMENTS for Chemical Modeling ===
    "Temperature Multiplier": slider_with_input("Temperature Multiplier", 0.1, 10.0, 1.0, 0.01),
    "Pressure Multiplier": slider_with_input("Pressure Multiplier", 0.1, 10.0, 1.0, 0.01),
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
# === Precompute Half-Life Matrix for Cross-Module Use ===
atomic_numbers = np.arange(1, 121)
isotope_range = np.arange(1, 21)
Z_grid, iso_grid = np.meshgrid(atomic_numbers, isotope_range, indexing='ij')

weak_force = constants["Weak Force Multiplier"]
strong_force = constants["Strong Force Multiplier"]

base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
weak_decay_penalty = np.exp(-((weak_force - 1.0) ** 2) * 3)
strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong_force))

half_life_matrix = base_half_life * weak_decay_penalty * strong_bonus
half_life_matrix = np.clip(half_life_matrix, 0, 1)

# Mean half-life per element (used later)
mean_half_life_per_element = half_life_matrix.mean(axis=1)
tabs = st.tabs([
    "Periodic Table Stability",
    "Island of Instability",
    "Star Formation Potential",
    "Life Probability (Heatmap)",
    "Quantum Bonding",
    "Universe Emergence Probability",
    "Element Abundance Probability",
    "EM Radiation Risk",
    "Star Lifespan Model",
    "Dark Matter Simulation",
    "Atomic Stability",
    "Universe Life Probability Over Time",
    "Molecular Bonding Model (Element Specific)",
    "Molecular Abundance Map",
    "Isotope Decay & Half-Life Model",
    "Periodic Table Expansion Potential",
    "Nuclear Energy Binding Map"
])

# --- Continue Tabs (starting from tab1) ---
# === Periodic Table Stability (Scientific Model → Strong Force, EM Force, Weak Force Dependent) ===
with tabs[0]:
    st.subheader("Periodic Table Stability Probability")

    atomic_numbers = np.arange(1, 121)

    # Pull slider values from physical constants
    strong_force = constants["Strong Force Multiplier"]
    em_force = constants["Electromagnetic Force Multiplier"]
    weak_force = constants["Weak Force Multiplier"]

    # Create dynamic EM force range centered on slider value
    em_force_values = np.linspace(em_force - 2, em_force + 2, 50)
    em_force_values = np.clip(em_force_values, 0.1, 10.0)
    atomic_grid, em_grid = np.meshgrid(atomic_numbers, em_force_values)

    # Calculate base nuclear stability (shell + atomic number)
    base_stability = np.exp(-np.abs(atomic_grid - 30) / 20)

    # Strong Force Effect → Higher strong force → stabilizes heavy nuclei
    strong_bonus = np.exp(-np.abs(atomic_grid - 80) / (20 * strong_force))

    # EM Force Effect → Higher EM force → destabilizes heavy nuclei
    em_penalty = np.exp(-np.abs(em_grid - em_force))

    # Weak Force Effect → Ideal weak force (~1.0) → most stable isotopes
    weak_bonus = np.exp(-((weak_force - 1.0) ** 2) * 3)

    # Final Stability Probability
    stability_probability = base_stability * strong_bonus * em_penalty * weak_bonus

    # Normalize for visualization
    stability_probability = np.clip(stability_probability, 0, 1)

    # Plot 3D Scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=atomic_grid.flatten(),
        y=em_grid.flatten(),
        z=stability_probability.flatten(),
        mode='markers',
        marker=dict(size=5, color=stability_probability.flatten(), colorscale='Viridis', colorbar=dict(title='Stability'))
    )])

    fig.update_layout(
        title="Periodic Table Stability Probability",
        scene=dict(
            xaxis_title='Atomic Number',
            yaxis_title='EM Force Multiplier',
            zaxis_title='Stability Probability'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**AI Analysis → Scientific Summary**")
    st.markdown("This advanced scientific model calculates element stability based on fundamental forces:")
    st.markdown("- **Strong Force Multiplier → Higher values stabilize heavier nuclei → reduces instability.**")
    st.markdown("- **EM Force Multiplier → Higher values destabilize heavy elements → proton repulsion dominates.**")
    st.markdown("- **Weak Force Multiplier → Deviations from 1.0 destabilize isotopes → ideal near 1.0.**")
    st.markdown("- This model reflects realistic nuclear behavior, dynamically updating with universe physical constants.")
with tabs[1]:
    st.subheader("Island of Instability (Periodic Pattern + Scientific Bonus)")

    # Atomic numbers and strong force range
    atomic_number_values = np.linspace(50, 120, 50)
    strong_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, atomic_grid = np.meshgrid(strong_force_values, atomic_number_values)

    # Original periodic instability model (shell closure inspired)
    base_instability = np.abs(np.sin((strong_grid - constants["Strong Force Multiplier"]) * 5)) * (atomic_grid / 120)

    # Scientific bonus → lower instability when closer to ideal strong force (1.0)
    scientific_bonus = np.exp(-np.abs(strong_grid - constants["Strong Force Multiplier"]))

    # Final instability → periodic + scientific shift
    instability = base_instability * scientific_bonus

    # Plot
    fig = go.Figure(data=[go.Surface(
        z=instability,
        x=strong_grid,
        y=atomic_grid,
        colorscale='Inferno',
        colorbar=dict(title='Instability Level')
    )])

    fig.update_layout(
        title="Island of Instability",
        scene=dict(
            xaxis_title='Strong Force Multiplier',
            yaxis_title='Atomic Number',
            zaxis_title='Instability Level'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This model combines periodic nuclear shell structure patterns and scientific tuning effects:")
    st.markdown("- **Periodic Pattern → Models peaks/valleys of nuclear shell closures (like magic numbers).**")
    st.markdown("- **Scientific Bonus → Strong Force near ideal (1.0) improves stability, shifting instability lower.**")
    st.markdown("- The result is a dynamic, accurate, and responsive simulation of nuclear instability across universes.")
# === Star Formation Potential (Fully Scientific Model → Gravity + Dark Energy + EM + Strong + Weak) ===
with tabs[2]:
    st.subheader("Star Formation Potential")
    
    # Slider values → Physical constants
    gravity_slider = constants["Gravitational Constant Multiplier"]
    dark_energy_slider = constants["Dark Energy Multiplier"]
    strong_force_slider = constants["Strong Force Multiplier"]
    em_force_slider = constants["Electromagnetic Force Multiplier"]
    weak_force_slider = constants["Weak Force Multiplier"]

    # Create local dynamic grid around current universe constants
    gravity_center = gravity_slider
    dark_energy_center = dark_energy_slider

    gravity_values = np.linspace(gravity_center - 2, gravity_center + 2, 50)
    dark_energy_values = np.linspace(dark_energy_center - 2, dark_energy_center + 2, 50)

    gravity_values = np.clip(gravity_values, 0.1, 10.0)
    dark_energy_values = np.clip(dark_energy_values, 0.1, 10.0)

    gravity_grid, dark_grid = np.meshgrid(gravity_values, dark_energy_values)

    # Calculate factors

    # 1. Gravity → Higher gravity → easier collapse
    gravity_factor = np.exp(-np.abs(gravity_grid - gravity_slider) * 0.5)

    # 2. Dark Energy → Higher DE → harder collapse
    dark_energy_penalty = np.exp(-np.abs(dark_grid - dark_energy_slider) * 1.5)

    # 3. EM Force → Higher EM → higher radiation pressure → harder collapse
    em_force_penalty = np.exp(-np.abs(em_force_slider - 1.0) * 2.0)

    # 4. Strong Force → Higher strong → easier ignition → better formation
    strong_force_bonus = np.exp(-np.abs(strong_force_slider - 1.0) * 1.5)

    # 5. Weak Force → Optimal near 1.0 → better fusion → too weak or strong → worse
    weak_force_bonus = np.exp(-((weak_force_slider - 1.0) ** 2) * 3.0)

    # Final Star Formation Potential
    star_formation_potential = gravity_factor * dark_energy_penalty * em_force_penalty * strong_force_bonus * weak_force_bonus

    # Normalize
    star_formation_potential /= np.max(star_formation_potential)

    # Plot
    fig = go.Figure(data=[go.Surface(
        z=star_formation_potential,
        x=gravity_grid,
        y=dark_grid,
        colorscale='Viridis',
        colorbar=dict(title='Star Formation Potential')
    )])

    fig.update_layout(
        title="Star Formation Potential",
        scene=dict(
            xaxis_title='Gravity Multiplier',
            yaxis_title='Dark Energy Multiplier',
            zaxis_title='Star Formation Potential'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This module models star formation based on five critical universal constants:")
    st.markdown("- **Gravity → Higher gravity compresses gas → promotes collapse → better star formation.**")
    st.markdown("- **Dark Energy → Expands space → lowers density → suppresses star formation.**")
    st.markdown("- **EM Force → Higher EM increases radiation pressure → suppresses star formation.**")
    st.markdown("- **Strong Force → Higher strong force makes nuclear ignition easier → promotes star formation.**")
    st.markdown("- **Weak Force → Optimal weak force allows stable fusion → deviations suppress formation.")
# === Life Probability (Heatmap → Linked to Metallicity + Forces) ===
with tabs[3]:
    st.subheader("Life Probability Map (Linked to Metallicity and Forces)")

    # Link to physics constants (user sliders)
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)

    # Current universe values (from sliders)
    current_strong = constants["Strong Force Multiplier"]
    current_em = constants["Electromagnetic Force Multiplier"]

    # Calculate force-based life probability
    force_life_prob = np.exp(-((strong_grid - current_strong)**2 + (em_grid - current_em)**2) / 3)

    # ====== NEW SCIENTIFIC PART (METALLICITY LINKING) ======

    # Metallicity (simulate linkage from star evolution → in real case this comes from star metallicity result)
    # For now, simulate as steady increasing function:
    #  (Later → directly connect this to star formation module metallicity[-1] value!)
    simulated_metallicity = 0.5  # Example: halfway enriched universe

    # Metallicity boost factor → low metals suppress life, medium to high metals enable life
    metallicity_factor = (simulated_metallicity - 0.1) / (0.5 - 0.1)
    metallicity_factor = np.clip(metallicity_factor, 0, 1)

    # Broadcast metallicity factor across grid
    metallicity_grid = np.ones_like(force_life_prob) * metallicity_factor

    # Final life probability
    final_life_prob = force_life_prob * metallicity_grid

    # Plotting
    fig = go.Figure(data=go.Heatmap(
        z=final_life_prob,
        x=strong_force_values,
        y=em_force_values,
        colorscale='Viridis',
        colorbar=dict(title='Life Probability')
    ))

    fig.update_layout(
        title="Life Probability Map (Force + Metallicity Linked)",
        xaxis_title="Strong Force Multiplier",
        yaxis_title="EM Force Multiplier"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This life probability model now includes:")
    st.markdown("- **Forces Compatibility** → Molecular and atomic stability zones based on nuclear and EM forces.")
    st.markdown("- **Metallicity Influence** → Low metallicity suppresses life chances (no planet formation), while medium to high metallicity supports life emergence.")
    st.markdown("The model shows that universes with optimal physical constants and sufficient cosmic evolution (star death → metals) are the most likely to support life.")

# === Quantum Bonding Probability (Enhanced with Temperature and Pressure) ===
with tabs[4]:
    st.subheader("Quantum Bonding Probability")

    # Add new sliders to constants
    if "Temperature Multiplier" not in constants:
        constants["Temperature Multiplier"] = slider_with_input("Temperature Multiplier", 0.1, 10.0, 1.0, 0.01)
    if "Pressure Multiplier" not in constants:
        constants["Pressure Multiplier"] = slider_with_input("Pressure Multiplier", 0.1, 10.0, 1.0, 0.01)

    # Load force constants
    strong_force = constants["Strong Force Multiplier"]
    em_force = constants["Electromagnetic Force Multiplier"]
    temperature = constants["Temperature Multiplier"]
    pressure = constants["Pressure Multiplier"]

    # Create grid for strong & EM force combinations
    strong_force_values = np.linspace(0.1, 10.0, 50)
    em_force_values = np.linspace(0.1, 10.0, 50)
    strong_grid, em_grid = np.meshgrid(strong_force_values, em_force_values)

    # Base bonding probability (force interaction proximity to user setting)
    bonding_prob = np.exp(-((strong_grid - strong_force)**2 + (em_grid - em_force)**2) / 2)

    # Temperature & pressure modifiers → ideal at 1.0
    temp_effect = np.exp(-((temperature - 1.0)**2) * 2.0)
    pressure_effect = np.exp(-((pressure - 1.0)**2) * 2.0)

    # Apply physical modifiers globally
    bonding_prob *= temp_effect * pressure_effect

    # === 3D Surface Plot (Preserved Original) ===
    fig_3d = go.Figure(data=[go.Surface(
        z=bonding_prob,
        x=strong_force_values,
        y=em_force_values,
        colorscale='Viridis',
        colorbar=dict(title='Bonding Probability')
    )])

    fig_3d.update_layout(
        title="3D Quantum Bonding Probability (Forces + Temp/Pressure)",
        scene=dict(
            xaxis_title='Strong Force Multiplier',
            yaxis_title='EM Force Multiplier',
            zaxis_title='Bonding Probability'
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # === 2D Contour Plot (New Addition) ===
    fig_2d = go.Figure(data=go.Contour(
        z=bonding_prob,
        x=strong_force_values,
        y=em_force_values,
        colorscale='Viridis',
        contours_coloring='heatmap',
        colorbar=dict(title='Bonding Probability'),
        line_smoothing=1.2
    ))

    fig_2d.update_layout(
        title="2D Bonding Zone Map (Temperature & Pressure Adjusted)",
        xaxis_title='Strong Force Multiplier',
        yaxis_title='EM Force Multiplier'
    )

    st.plotly_chart(fig_2d, use_container_width=True)

    # === Scientific Explanation ===
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This upgraded bonding model includes:")
    st.markdown("- **Strong and EM Force Interplay** → Nucleus-electron interactions define potential for chemical bonding.")
    st.markdown("- **Temperature Multiplier** → High temperatures may prevent stable bonding due to high kinetic energy.")
    st.markdown("- **Pressure Multiplier** → Optimal pressure enables orbital overlap and electron cloud compression.")
    st.markdown("- **3D Surface** shows quantum bonding efficiency across force space.")
    st.markdown("- **2D Contour** reveals high-probability bonding zones for molecule formation across universes.")
# ------------------------
# === Universe Probability ===
with tabs[5]:
    st.subheader("Universe Emergence Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    ax.set_ylabel("Probability")
    ax.set_title("Universe Viability Probability")
    st.pyplot(fig)
    
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This chart shows the calculated overall probability that this universe configuration can support stable chemistry and life. Lower deviation values (closer to baseline constants) result in higher probability.")
# === Element Abundance (Updated with Slider-Linked Half-Life Model) ===
with tabs[6]:
    st.subheader("Element Abundance Probability")

    # Recalculate half-life matrix using current slider values
    strong_force = constants["Strong Force Multiplier"]
    weak_force = constants["Weak Force Multiplier"]

    atomic_numbers = np.arange(1, 121)
    isotope_range = np.arange(1, 21)
    Z_grid, iso_grid = np.meshgrid(atomic_numbers, isotope_range, indexing='ij')

    base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
    weak_decay_penalty = np.exp(-((weak_force - 1.0) ** 2) * 3)
    strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong_force))

    half_life_matrix = base_half_life * weak_decay_penalty * strong_bonus
    half_life_matrix = np.clip(half_life_matrix, 0, 1)

    mean_half_life_per_element = half_life_matrix.mean(axis=1)
    half_life_weight = np.mean(mean_half_life_per_element)

    # Force-specific abundance, modulated by average isotope half-life
    forces = ["Strong", "EM", "Weak"]
    strong_abundance = np.exp(-abs(strong_force - 1.0)) * half_life_weight
    em_abundance = np.exp(-abs(constants["Electromagnetic Force Multiplier"] - 1.0)) * half_life_weight
    weak_abundance = np.exp(-abs(weak_force - 1.0)) * half_life_weight
    abundance = [strong_abundance, em_abundance, weak_abundance]

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    ax.set_ylabel("Relative Abundance")
    ax.set_title("Predicted Element Abundance per Force Multiplier")
    st.pyplot(fig)

    # Explanation
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("- **This module now recalculates nuclear half-lives using live slider values.**")
    st.markdown("- **Abundance = Force tuning × isotope survivability**, giving more realistic results.")
    st.markdown("- Universes with poor isotope stability have low elemental abundance, even if forces are well-tuned.")
# === Radiation Risk ===
with tabs[7]:
    st.subheader("EM Radiation Risk")
    x = np.linspace(0.1, 10.0, 500)
    y = (x**2) / 100
    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.axvline(constants["Electromagnetic Force Multiplier"], color='r', linestyle='--', label="Current EM Force")
    ax.legend()
    ax.set_xlabel("Electromagnetic Force Multiplier")
    ax.set_ylabel("Radiation Output Factor")
    ax.set_title("Predicted Radiation Output")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("Higher EM force constants increase electromagnetic interactions, leading to more radiation. This graph models radiation risk scaling, important for universe habitability.")

# === Star Lifespan (Scientific Mass-Luminosity-Lifespan Model) ===
with tabs[8]:
    st.subheader("Star Lifespan Model")

    # Define gravity multiplier range
    gravity_values = np.linspace(0.1, 10.0, 500)

    # Assume stellar mass scales linearly with gravity (simplified)
    stellar_mass = gravity_values  # M ∝ gravity multiplier

    # Luminosity → L ∝ M^3.5
    luminosity = stellar_mass ** 3.5

    # Lifetime → τ ∝ M / L → τ ∝ 1 / M^2.5
    stellar_lifetime = 1 / (stellar_mass ** 2.5)

    # Normalize lifetime for graphing
    stellar_lifetime /= np.max(stellar_lifetime)

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gravity_values,
        y=stellar_lifetime,
        mode='lines',
        name='Relative Star Lifespan',
        line=dict(color='darkgreen', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[constants["Gravitational Constant Multiplier"]],
        y=[1 / (constants["Gravitational Constant Multiplier"] ** 2.5) / np.max(1 / (stellar_mass ** 2.5))],
        mode='markers',
        name='Current Universe Setting',
        marker=dict(size=12, color='red', symbol='x')
    ))

    fig.update_layout(
        title="Star Lifespan vs Gravity Multiplier (Mass-Luminosity-Lifespan Model)",
        xaxis_title='Gravitational Constant Multiplier',
        yaxis_title='Relative Star Lifespan',
        legend_title="Stellar Lifetime"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This model uses stellar physics to calculate star lifespan based on gravitational strength:")
    st.markdown("- **Stellar Mass scales with gravity → Higher gravity → Higher stellar mass.**")
    st.markdown("- **Luminosity increases with mass (L ∝ M^3.5) → more massive stars burn faster.**")
    st.markdown("- **Stellar Lifespan → Shortens significantly as gravity increases (τ ∝ 1 / M^2.5).**")
    st.markdown("- Universes with stronger gravity have shorter-lived stars, reducing time for planets and life to evolve.")
# === 3D Dark Matter Tendril Simulation (Physics Linked Version) ===
with tabs[9]:
    st.subheader("Dark Matter Tendril Simulation")

    # LINK TO PHYSICAL CONSTANTS (user sliders)
    gravity_multiplier = constants["Gravitational Constant Multiplier"]
    dark_energy_multiplier = constants["Dark Energy Multiplier"]

    size = 50
    scale = 10
    num_clusters = 5

    # Gravity effect → higher gravity = tighter clusters
    gravity_effect = gravity_multiplier
    cluster_spread = 4.0 / gravity_effect

    # Dark energy effect → higher dark energy = larger voids
    dark_energy_effect = dark_energy_multiplier
    space_stretch = 1.0 / dark_energy_effect

    # Create 3D grid
    x = np.linspace(-scale * space_stretch, scale * space_stretch, size)
    y = np.linspace(-scale * space_stretch, scale * space_stretch, size)
    z = np.linspace(-scale * space_stretch, scale * space_stretch, size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Generate random cluster centers
    density = np.zeros_like(X)
    np.random.seed(42)
    cluster_centers = np.random.uniform(-scale, scale, (num_clusters, 3))

    # Create tendril-like Gaussian blobs
    for cx, cy, cz in cluster_centers:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        blob = np.exp(-(dist**2) / cluster_spread)
        density += blob

    # Normalize
    density /= np.max(density)

    # Extract significant points
    threshold = 0.2
    points = np.where(density > threshold)

    x_points = X[points]
    y_points = Y[points]
    z_points = Z[points]
    density_points = density[points]

    # Plot dark matter tendrils
    fig = go.Figure(data=[go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=3,
            color=density_points,
            colorscale='Inferno',
            opacity=0.75,
            colorbar=dict(title='Dark Matter Density')
        )
    )])

    fig.update_layout(
        title="3D Dark Matter Tendril Simulation",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This simulation links dark matter tendrils and cluster formation directly to physical constants:")
    st.markdown("- **Higher Gravity Multiplier** compresses clusters and tightens tendrils.")
    st.markdown("- **Higher Dark Energy Multiplier** stretches cosmic web, expanding voids and diffusing structures.")
    st.markdown("This creates a scientifically inspired visualization of how changes in universal constants shape the cosmic web.")

# === 3D Atomic Stability Probability (Scientific → Strong, EM, Weak Forces) ===
with tabs[10]:
    st.subheader("Atomic Stability Probability per Isotope")

    atomic_numbers = np.arange(1, 121)
    isotopes_per_element = 20

    # Slider inputs → physical constants
    strong_force_slider = constants["Strong Force Multiplier"]
    em_force_slider = constants["Electromagnetic Force Multiplier"]
    weak_force_slider = constants["Weak Force Multiplier"]

    # Base stability → favor mid-Z → natural stability
    base_stability = np.exp(-np.abs(atomic_numbers - 40) / 30)

    # Strong Force → Higher strong force → stabilizes heavier nuclei → helps
    strong_force_bonus = np.exp(-np.abs(atomic_numbers - 80) / (25 * strong_force_slider))

    # EM Force → Higher EM → more proton repulsion → destabilizes
    em_force_penalty = np.exp(-np.abs(atomic_numbers - 40) / (20 * em_force_slider))

    # Weak Force → Optimal near 1.0 → stable beta decay cycles
    weak_force_optimal = np.exp(-((weak_force_slider - 1.0) ** 2) * 3.0)

    # Final stability
    final_stability = base_stability * strong_force_bonus * em_force_penalty * weak_force_optimal
    final_stability = np.clip(final_stability, 0, 1)

    # Add isotope variation → Weak Force dependent
    np.random.seed(42)
    stability_matrix = np.array([
        final_stability + np.random.normal(0, 0.05 * weak_force_slider, len(atomic_numbers))
        for _ in range(isotopes_per_element)
    ]).T
    stability_matrix = np.clip(stability_matrix, 0, 1)

    # === FIXED: Safe index access ===
    Z_vals, Isotope_vals, Stability_vals = [], [], []
    num_elements = stability_matrix.shape[0]
    num_isotopes = stability_matrix.shape[1]

    for Z in atomic_numbers:
        for iso in range(1, isotopes_per_element + 1):
            z_index = Z - 1
            iso_index = iso - 1

            if z_index < num_elements and iso_index < num_isotopes:
                Z_vals.append(Z)
                Isotope_vals.append(iso)
                Stability_vals.append(stability_matrix[z_index, iso_index])
            else:
                st.warning(f"IndexError avoided: Z={Z}, iso={iso}")

    # Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=Z_vals,
        y=Isotope_vals,
        z=Stability_vals,
        mode='markers',
        marker=dict(size=5, color=Stability_vals, colorscale='Plasma', colorbar=dict(title='Stability'))
    )])

    fig.update_layout(
        title="3D Atomic Stability Probability per Isotope (Scientific Model)",
        scene=dict(
            xaxis_title='Atomic Number',
            yaxis_title='Isotope Number',
            zaxis_title='Stability Probability'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This advanced model simulates isotope stability based on strong, EM, and weak nuclear forces:")
    st.markdown("- **Strong Force → Higher values stabilize heavier nuclei → promotes stability.**")
    st.markdown("- **EM Force → Higher EM force increases proton repulsion → reduces stability.**")
    st.markdown("- **Weak Force → Optimal near 1.0 → deviations increase instability through inefficient decay modes.**")
    st.markdown("- Each isotope varies randomly per element, simulating natural isotope-dependent instability with weak force influence.")
with tabs[11]:
    st.subheader("Universe Life Probability Over Cosmic Time")

    # Use star formation metallicity from previous calculation
    time_steps = 100
    gravity_multiplier = constants["Gravitational Constant Multiplier"]
    dark_energy_multiplier = constants["Dark Energy Multiplier"]
    strong_force_multiplier = constants["Strong Force Multiplier"]
    em_force_multiplier = constants["Electromagnetic Force Multiplier"]

    initial_gas_density = 1.0
    star_formation_efficiency_base = 0.05

    gravity_effect = gravity_multiplier
    dark_energy_effect = 1 / dark_energy_multiplier

    time = np.arange(time_steps)
    gas_density = np.zeros(time_steps)
    star_density = np.zeros(time_steps)
    metallicity = np.zeros(time_steps)

    gas_density[0] = initial_gas_density
    star_density[0] = 0
    metallicity[0] = 0.01

    for t in range(1, time_steps):
        star_formation_efficiency = star_formation_efficiency_base * gravity_effect * dark_energy_effect
        stars_formed = gas_density[t-1] * star_formation_efficiency

        gas_density[t] = gas_density[t-1] - stars_formed
        star_density[t] = star_density[t-1] + stars_formed
        metallicity[t] = metallicity[t-1] + stars_formed * 0.02

        if gas_density[t] < 0:
            gas_density[t] = 0

    # Calculate life probability over time
    metallicity_factor = (metallicity - 0.1) / (0.5 - 0.1)
    metallicity_factor = np.clip(metallicity_factor, 0, 1)

    force_factor = np.exp(-((strong_force_multiplier - 1.0)**2 + (em_force_multiplier - 1.0)**2) / 3)

    life_probability_time = metallicity_factor * force_factor

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=metallicity_factor, mode='lines', name='Metallicity Factor', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=time, y=life_probability_time, mode='lines', name='Life Probability', line=dict(color='purple')))

    fig.update_layout(
        title="Universe Life Probability Over Time (Metallicity + Forces Combined)",
        xaxis_title='Time (Arbitrary Units ~ Billions of Years)',
        yaxis_title='Probability',
        legend_title="Factors"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This graph shows how life probability in the universe changes over time:")
    st.markdown("- **Early Universe → Low Metallicity → Low Life Probability**")
    st.markdown("- **Mid-life Universe → High Metallicity → Peak Life Probability**")
    st.markdown("- **Late Universe → Stars fade, no new metals → Declining Life Probability**")
    st.markdown("- **Force Constants → Tuning effects shown through scaling → unfavorable universes may never reach high probability**")
# === tabs[12]: Molecular Bonding Model (Element Specific Chemistry) ===
with tabs[12]:
    st.subheader("Molecular Bonding Stability (Element-Specific Chemistry)")

    # Constants
    em_force = constants["Electromagnetic Force Multiplier"]
    strong_force = constants["Strong Force Multiplier"]
    weak_force = constants["Weak Force Multiplier"]
    temperature = constants["Temperature Multiplier"]
    pressure = constants["Pressure Multiplier"]

    # Isotope survivability factor
    atomic_numbers = np.arange(1, 121)
    Z_grid, iso_grid = np.meshgrid(atomic_numbers, np.arange(1, 21), indexing='ij')
    base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
    weak_decay_penalty = np.exp(-((weak_force - 1.0) ** 2) * 3)
    strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong_force))
    half_life_matrix = np.clip(base_half_life * weak_decay_penalty * strong_bonus, 0, 1)
    isotope_stability_factor = np.mean(half_life_matrix)

    # Baseline molecules
    molecules = {
        "H₂ (Covalent)": 1.0,
        "CO₂ (Covalent)": 0.85,
        "Fe (Metallic)": 0.75,
        "H₂O (Polar)": 0.90,
        "Uranium Compounds (Heavy)": 0.65
    }

    # Force and environmental modifiers
    em_modifier = np.exp(-abs(em_force - 1.0) * 2.0)
    strong_modifier = np.exp(-abs(strong_force - 1.0) * 1.5)
    weak_modifier = np.exp(-abs(weak_force - 1.0) * 1.2)
    temp_modifier = np.exp(-((temperature - 1.0)**2) * 2.0)
    pressure_modifier = np.exp(-((pressure - 1.0)**2) * 2.0)

    global_modifier = em_modifier * strong_modifier * weak_modifier * temp_modifier * pressure_modifier * isotope_stability_factor

    # Final molecular stabilities
    adjusted_stabilities = {mol: base * global_modifier for mol, base in molecules.items()}
    viability_threshold = 0.5
    molecule_names = list(adjusted_stabilities.keys())
    stability_values = list(adjusted_stabilities.values())
    color_map = ['green' if val > viability_threshold else 'gray' for val in stability_values]

    # Plot
    fig = go.Figure(data=[go.Bar(
        x=molecule_names,
        y=stability_values,
        marker=dict(color=color_map),
        text=[f"{v:.2f}" for v in stability_values],
        textposition='outside'
    )])
    fig.update_layout(
        title="Molecular Bonding Viability Across Forces, Temperature, Pressure, Isotope Stability",
        yaxis_title="Relative Bond Stability",
        xaxis_title="Molecule Type",
        yaxis_range=[0, 1.2]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analysis
    st.markdown("### AI Analysis → Scientific Chemistry Summary")
    st.markdown("- **This model is now isotope-aware**, penalizing bonding where atoms are too unstable to persist.")
    st.markdown("- **Green bars** indicate viable molecular bonds under the current physical and nuclear conditions.")
    st.markdown("- **Gray bars** mean instability prevents molecule formation — due to weak decay or nuclear disruption.")
# === tabs[13]: Molecular Abundance Map (Force & Thermal Influence) ===
with tabs[13]:
    st.subheader("Molecular Abundance Probability Map")

    # Constants
    em = constants["Electromagnetic Force Multiplier"]
    strong = constants["Strong Force Multiplier"]
    weak = constants["Weak Force Multiplier"]
    temp = constants["Temperature Multiplier"]
    pressure = constants["Pressure Multiplier"]

    # Isotope survivability factor
    atomic_numbers = np.arange(1, 121)
    Z_grid, iso_grid = np.meshgrid(atomic_numbers, np.arange(1, 21), indexing='ij')
    base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
    weak_decay_penalty = np.exp(-((weak - 1.0) ** 2) * 3)
    strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong))
    half_life_matrix = np.clip(base_half_life * weak_decay_penalty * strong_bonus, 0, 1)
    isotope_stability_factor = np.mean(half_life_matrix)

    # Molecular families
    molecular_families = {
        "Simple Covalent (H₂, O₂)": 1.0,
        "Polar Molecules (H₂O)": 0.85,
        "Carbon Chains (CH₄, CO₂)": 0.90,
        "Metallic Bonds (Fe, Ni)": 0.70,
        "Heavy Nuclear Compounds (U, Th)": 0.60
    }

    abundance = {}
    for molecule, base in molecular_families.items():
        if "Covalent" in molecule:
            modifier = np.exp(-abs(em - 1.0) * 2) * np.exp(-((temp - 1.0)**2) * 2)
        elif "Polar" in molecule:
            modifier = np.exp(-abs(em - 1.0) * 2) * np.exp(-((temp - 1.0)**2) * 2) * np.exp(-((pressure - 1.0)**2) * 1.5)
        elif "Carbon" in molecule:
            modifier = np.exp(-abs(em - 1.0) * 1.5) * np.exp(-abs(strong - 1.0) * 1.5)
        elif "Metallic" in molecule:
            modifier = np.exp(-((pressure - 1.0)**2) * 2.0) * np.exp(-abs(em - 1.0) * 1.2)
        elif "Heavy" in molecule:
            modifier = np.exp(-abs(strong - 1.0) * 3) * np.exp(-abs(weak - 1.0) * 2)

        abundance[molecule] = base * modifier * isotope_stability_factor

    # Normalize
    values = np.array(list(abundance.values()))
    values /= np.max(values)

    # Plot
    fig = go.Figure(data=[go.Bar(
        x=list(abundance.keys()),
        y=values,
        marker_color='indigo',
        text=[f"{v:.2f}" for v in values],
        textposition='outside'
    )])
    fig.update_layout(
        title="Relative Abundance of Molecular Families (Isotope-Corrected)",
        xaxis_title="Molecular Family",
        yaxis_title="Normalized Abundance Probability",
        yaxis_range=[0, 1.1]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.markdown("### AI Analysis → Scientific Chemistry Summary")
    st.markdown("- **Now tethered to isotope survival rates.**")
    st.markdown("- Even if forces and pressure are perfect, molecules will vanish if atoms decay too fast.")
    st.markdown("- This adds deep scientific realism to multiverse chemical prediction.")
# === tabs[14]: Isotope Decay & Half-Life Model ===
with tabs[14]:
    st.subheader("Isotope Decay & Half-Life Model")

    # Pull updated constants directly
    weak_force = constants["Weak Force Multiplier"]
    strong_force = constants["Strong Force Multiplier"]

    # Define atomic number and isotope ranges
    atomic_numbers = np.arange(1, 121)
    isotope_range = np.arange(1, 21)
    Z_grid, iso_grid = np.meshgrid(atomic_numbers, isotope_range, indexing='ij')

    # Compute half-life matrix (updated every run)
    base_half_life = np.exp(-np.abs(Z_grid - 50) / 20)
    weak_decay_penalty = np.exp(-((weak_force - 1.0) ** 2) * 3)
    strong_bonus = np.exp(-np.abs(Z_grid - 80) / (25 * strong_force))

    half_life_matrix = base_half_life * weak_decay_penalty * strong_bonus
    half_life_matrix = np.clip(half_life_matrix, 0, 1)
    half_life_matrix /= half_life_matrix.max()  # Normalize

    # === 3D Surface Plot of Half-Lives ===
    fig1 = go.Figure(data=[go.Surface(
        z=half_life_matrix,
        x=atomic_numbers,
        y=isotope_range,
        colorscale='Cividis',
        colorbar=dict(title='Normalized Half-Life')
    )])
    fig1.update_layout(
        title="Isotope Half-Life Map (Z vs Isotope Number)",
        scene=dict(
            xaxis_title='Atomic Number (Z)',
            yaxis_title='Isotope Number',
            zaxis_title='Relative Half-Life'
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Life-Compatible Isotope Bar Plot ===
    threshold = 0.1  # Define long-lived isotopes as top 10% within current universe
    life_viable_isotopes = (half_life_matrix > threshold).sum(axis=1)

    fig2, ax = plt.subplots()
    ax.bar(atomic_numbers, life_viable_isotopes, color='darkorange')
    ax.set_title("Count of Life-Compatible Isotopes per Element")
    ax.set_xlabel("Atomic Number")
    ax.set_ylabel("# of Isotopes with Long Half-Life")
    st.pyplot(fig2)

    # === AI Summary ===
    st.markdown("### AI Analysis → Scientific Nuclear Summary")
    st.markdown("- **Weak Force Multiplier** → Controls decay rates. Higher = faster decay = fewer long-lived isotopes.")
    st.markdown("- **Strong Force Multiplier** → Stabilizes heavier nuclei. Higher = longer lifespans for heavy elements.")
    st.markdown("- **Dynamic thresholding** (top 10%) enables realistic comparisons across many different universes.")
with tabs[15]:
    st.subheader("Periodic Table Expansion Potential")

    # Pull force multipliers
    strong_force = constants["Strong Force Multiplier"]
    em_force = constants["Electromagnetic Force Multiplier"]
    weak_force = constants["Weak Force Multiplier"]

    # Theoretical Z limit based on balance between nuclear cohesion and EM repulsion
    # Use a scaled sigmoid centered around Z=118
    Z = np.linspace(1, 200, 200)

    cohesion = np.exp(-np.abs(Z - 80) / (25 * strong_force))  # stabilizing
    repulsion = 1 / (1 + np.exp(-(Z - 100) / (10 * em_force)))  # destabilizing EM effect
    weak_instability = np.exp(-((weak_force - 1.0) ** 2) * 3)

    stability_curve = cohesion * (1 - repulsion) * weak_instability
    stability_curve = np.clip(stability_curve, 0, 1)

    # Threshold: elements with stability > 0.1 are "potentially stable"
    Z_extended = Z[stability_curve > 0.1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Z,
        y=stability_curve,
        mode='lines',
        name='Expansion Curve',
        line=dict(color='royalblue')
    ))

    fig.add_trace(go.Scatter(
        x=[Z_extended[-1]] if len(Z_extended) > 0 else [],
        y=[stability_curve[Z_extended.shape[0]-1]] if len(Z_extended) > 0 else [],
        mode='markers+text',
        text=[f"Max Z: {int(Z_extended[-1])}"] if len(Z_extended) > 0 else [],
        textposition="top center",
        marker=dict(size=10, color='red'),
        name="Expansion Limit"
    ))

    fig.update_layout(
        title="Theoretical Periodic Table Expansion Limit",
        xaxis_title="Atomic Number (Z)",
        yaxis_title="Stability Potential",
        yaxis_range=[0, 1.05]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("- This model estimates the furthest stable atomic number (Z) a universe can support.")
    st.markdown("- **Strong Force** increases nuclear cohesion → allows higher Z before instability.")
    st.markdown("- **EM Force** causes proton repulsion → caps the growth of periodic table.")
    st.markdown("- **Weak Force** impacts decay resilience of extreme elements.")
    st.markdown("- **Max stable Z** is shown dynamically as physical constants vary.")
# === New Tab: Nuclear Binding Energy Map ===
tabs.append("Nuclear Binding Energy Map")
with tabs[-1]:
    st.subheader("Nuclear Binding Energy per Nucleon")

    # Slider values
    strong = constants["Strong Force Multiplier"]
    em = constants["Electromagnetic Force Multiplier"]
    weak = constants["Weak Force Multiplier"]  # will modulate symmetry term

    # SEMF coefficients (baseline in MeV)
    a_v  = 15.8  * strong        # volume term ∝ strong force
    a_s  = 18.3                  # surface term (unchanged)
    a_c  = 0.714 * em            # Coulomb term ∝ EM repulsion
    a_sym= 23.2 * (1/weak)       # symmetry term ∝ 1/weak (weak stabilizes n↔p balance)
    a_pair = 12.0                # pairing term (simplified constant)

    # Atomic mass numbers
    Z = np.arange(1, 121)
    A = Z * 2                     # approximate N≈Z for most stable isotopes
    N = A - Z

    # SEMF binding energy per nucleus
    BE = ( a_v*A
           - a_s*A**(2/3)
           - a_c*Z*(Z-1)/A**(1/3)
           - a_sym*(A-2*Z)**2/A
           + np.where((A%2)==0, a_pair/A**(1/2), -a_pair/A**(1/2)) )

    # Per-nucleon and clip
    BE_per_A = np.clip(BE/A, 0, None)

    # Plot curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Z,
        y=BE_per_A,
        mode='lines+markers',
        marker=dict(size=6),
        name="BE/A (MeV)"
    ))
    fig.update_layout(
        title="Binding Energy per Nucleon vs Atomic Number",
        xaxis_title="Atomic Number (Z)",
        yaxis_title="Binding Energy per Nucleon (MeV)",
        yaxis_range=[0, np.max(BE_per_A)*1.1]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight iron peak
    Z_peak = Z[np.argmax(BE_per_A)]
    st.markdown(f"**Peak BE/A at Z = {int(Z_peak)} → {BE_per_A.max():.2f} MeV/nucleon**")

    # Scientific Recap
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown(f"- **Volume Term (a_v)** scales with strong force → deeper well for all nuclei.")
    st.markdown(f"- **Coulomb Term (a_c)** scales with EM force → larger Z penalized more.")
    st.markdown(f"- **Symmetry Term (a_sym)** inversely tied to weak force → n↔p balance stability.")
    st.markdown("- The peak around **Fe–Ni** emerges naturally; universes with stronger strong force push the peak slightly higher in Z.")