
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator — ABSOLUTE TRUE FINAL FIXED VERSION")

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

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("This graph shows how electromagnetic force impacts element stability. Higher EM stabilizes lighter elements, but destabilizes heavier ones.")

with tabs[1]:
    st.subheader("Island of Instability (3D)")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Strong force changes shift stability in heavy elements. Higher values compress instability regions, lower values expand them.")

with tabs[2]:
    st.subheader("Star Formation Potential (3D)")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Gravity and dark energy multipliers affect star formation. Stars form under balanced values but fail in extreme conditions.")

with tabs[3]:
    st.subheader("Life Probability (Heatmap)")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Balanced strong and electromagnetic forces are essential for chemistry. This heatmap shows where life-supporting conditions arise.")

with tabs[4]:
    st.subheader("Quantum Bonding (3D)")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Nuclear and electromagnetic force adjustments affect molecular bonding. Strong deviations disrupt molecule formation.")

with tabs[5]:
    st.subheader("Universe Probability")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Deviation from standard constants lowers universe viability. Larger deviations reduce the chance of stable universes.")

with tabs[6]:
    st.subheader("Element Abundance")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Nuclear forces govern nucleosynthesis. This graph shows how elemental abundances shift across universes.")

with tabs[7]:
    st.subheader("Radiation Risk")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Electromagnetic force affects radiation risk. Higher values increase radiation hazards to chemical and biological systems.")

with tabs[8]:
    st.subheader("Star Lifespan")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Gravity controls fuel burn rates in stars. Higher gravity shortens lifespans, lower gravity extends them.")

with tabs[9]:
    st.subheader("2D Dark Matter Simulation")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("This simulation shows dark matter density variations. Fluctuations affect cosmic structure and galaxy formation.")

with tabs[10]:
    st.subheader("3D Atomic Stability")

import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Multiverse Physics Simulation")

# Sidebar - Universe Constants with % change display
st.sidebar.header("Adjust Physical Constants")





def slider_with_percent(label, min_value, max_value, value, step):
    col1, col2 = st.sidebar.columns([3, 1])

    # Text input first for user-defined value
    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed", key=label)

    # Validate user input
    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    # Apply button to sync value
    apply_key = f"{label}_apply"
    if st.sidebar.button("Apply", key=apply_key):
        slider_val = precise_val
    else:
        slider_val = value

    # Always render slider
    slider_val = col1.slider(label, min_value, max_value, slider_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)

    return slider_val

    with col2:
        st.markdown("<span style='font-size:11px;'>User Input</span>", unsafe_allow_html=True)
        precise_val_input = st.text_input("", str(value), label_visibility="collapsed")

    try:
        precise_val = float(precise_val_input)
        if precise_val < min_value:
            precise_val = min_value
        elif precise_val > max_value:
            precise_val = max_value
    except:
        precise_val = value

    slider_val = col1.slider(label, min_value, max_value, precise_val, step)

    percent_change = (slider_val - 1.0) * 100
    st.sidebar.markdown(f"<span style='font-size:12px;'>{label} Change: {percent_change:+.1f}% from baseline</span>", unsafe_allow_html=True)
    return slider_val

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

# Tabs
tabs = st.tabs([
    "Periodic Table Stability (3D)", "Island of Instability (3D)", "Star Formation Potential (3D)", 
    "Life Probability (Heatmap)", "Quantum Bonding (3D)", "Universe Probability",
    "Element Abundance", "Radiation Risk", "Star Lifespan", "2D Dark Matter Simulation", "3D Atomic Stability"
])

# Graphs autogenerated per tab selected

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Nuclear forces influence isotope stability. This graph shows which isotopes remain stable or decay under changed constants.")
