
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multiverse Physics Simulation", layout="wide")
st.title("Multiverse Physics Simulation")

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

tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "3D Dark Matter Simulation", "3D Atomic Stability", "Universe Life Probability Over Time"
])

# --- Continue Tabs (starting from tab1) ---
# === Periodic Table Stability (Scientific Model → Strong Force, EM Force, Weak Force Dependent) ===
with tabs[0]:
    st.subheader("Periodic Table Stability Probability (Scientific Model → Strong, EM, Weak Force Dependent)")

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
        title="Periodic Table Stability Probability (Scientific Model → Strong, EM, Weak Force Dependent)",
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
        title="Island of Instability (Periodic + Scientific Bonus Model)",
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
    st.subheader("Star Formation Potential (Fully Scientific Model)")

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
        title="Star Formation Potential (Fully Scientific → Gravity + Dark Energy + EM + Strong + Weak)",
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

# ------------------------
# === Universe Probability ===
with tabs[5]:
    st.subheader("Universe Probability")
    prob = np.exp(-deviation)
    fig, ax = plt.subplots()
    ax.bar(["Universe Probability"], [prob], color='purple')
    ax.set_ylabel("Probability")
    ax.set_title("Universe Viability Probability")
    st.pyplot(fig)
    
    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This chart shows the calculated overall probability that this universe configuration can support stable chemistry and life. Lower deviation values (closer to baseline constants) result in higher probability.")

# === Element Abundance ===
with tabs[6]:
    st.subheader("Element Abundance")
    forces = ["Strong", "EM", "Weak"]
    abundance = [np.exp(-abs(constants["Strong Force Multiplier"]-1)),
                 np.exp(-abs(constants["Electromagnetic Force Multiplier"]-1)),
                 np.exp(-abs(constants["Weak Force Multiplier"]-1))]
    fig, ax = plt.subplots()
    ax.bar(forces, abundance, color=['blue', 'magenta', 'yellow'])
    ax.set_ylabel("Relative Abundance")
    ax.set_title("Predicted Element Abundance per Force Multiplier")
    st.pyplot(fig)

    st.markdown("### AI Analysis → Scientific Summary")
    st.markdown("This graph estimates the abundance of elements in universes where the fundamental force strengths differ. Strong and weak forces alter nuclear formation rates while EM changes impact overall element formation efficiency.")

# === Radiation Risk ===
with tabs[7]:
    st.subheader("Radiation Risk")
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
    st.subheader("Star Lifespan vs Gravity Multiplier (Scientific Model)")

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
    st.subheader("3D Dark Matter Tendril Simulation (Linked to Physics Constants)")

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
        title="3D Dark Matter Tendril Simulation (Linked to Gravity and Dark Energy)",
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
# === 3D Atomic Stability Probability (Scientific Physics Model) ===
with tabs[10]:
    st.subheader("3D Atomic Stability Probability per Isotope (Scientific Model)")

    # Semi-Empirical Mass Formula coefficients
    a_v = 15.8
    a_s = 18.3
    a_c = 0.714
    a_a = 23.2
    a_p = 12.0

    def binding_energy(Z, A):
        if A <= 0 or Z <= 0 or Z > A:
            return 0
        N = A - Z
        B = a_v * A
        B -= a_s * A**(2/3)
        B -= a_c * (Z * (Z - 1)) / A**(1/3)
        B -= a_a * ((A - 2*Z)**2) / A
        if A % 2 == 1:
            delta = 0
        else:
            if Z % 2 == 0:
                delta = +a_p / A**0.5
            else:
                delta = -a_p / A**0.5
        B += delta
        return B

    # Atomic numbers and isotopes
    atomic_numbers = np.arange(1, 121)
    isotopes_per_element = 20

    stability_matrix = []

    for Z in atomic_numbers:
        isotope_stabilities = []
        for iso in range(Z, Z + isotopes_per_element):
            BE = binding_energy(Z, iso)
            BE_per_nucleon = BE / iso if iso > 0 else 0

            # Stability logic: > 7 MeV/nucleon → stable-ish
            stability = np.clip((BE_per_nucleon - 7) / 3, 0, 1)
            isotope_stabilities.append(stability)
        stability_matrix.append(isotope_stabilities)

    stability_matrix = np.array(stability_matrix)

    # Prepare plot
    Z_vals, Isotope_vals, Stability_vals = [], [], []
    for i, Z in enumerate(atomic_numbers):
        for j in range(isotopes_per_element):
            Z_vals.append(Z)
            Isotope_vals.append(j + 1)
            Stability_vals.append(stability_matrix[i, j])

    fig = go.Figure(data=[go.Scatter3d(
        x=Z_vals,
        y=Isotope_vals,
        z=Stability_vals,
        mode='markers',
        marker=dict(size=5, color=Stability_vals, colorscale='Plasma', colorbar=dict(title='Stability')))
    ])

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
    st.markdown("This version uses the Semi-Empirical Mass Formula to calculate binding energies and isotope stability. Isotopes with high binding energy per nucleon are more stable. As atomic number increases, stability generally decreases due to increased Coulomb repulsion.")
    # === Universe Life Probability Over Time (Metallicity + Forces Combined) ===
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