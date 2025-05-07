
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Omega Multiverse Simulator — ABSOLUTE TRUE FINAL FIX VERSION")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("This graph shows how EM force affects element stability. Higher EM stabilizes lighter elements but destabilizes heavier ones.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Strong force changes shift heavy element stability. Larger values shrink and lower values expand instability regions.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Gravity and dark energy balance star formation. Stars form in moderate ranges but fail under extreme conditions.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Strong and EM force balance controls chemistry complexity. Moderate values favor life-permitting reactions.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Bond strength varies with nuclear forces. Extreme force multipliers disrupt molecular stability.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Deviation from standard constants lowers universe viability. Large deviations make stable universes rare.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Nuclear forces drive nucleosynthesis. Force changes shift which elements form or remain abundant.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("High EM force increases radiation damage potential. Low EM decreases destructive interactions.")

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

    st.markdown("### AI Analysis — Scientific Summary")
    st.markdown("Gravity determines fuel burn rates. High gravity shortens stellar life, low gravity extends it.")
