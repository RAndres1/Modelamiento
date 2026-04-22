
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Respuesta Forzada y Sistemas con Excitación Externa",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
def system_derivatives(state, t, m, c, k, F0, omega, forcing_type):
    x, v = state
    if forcing_type == "Senoidal":
        force = F0 * np.sin(omega * t)
    elif forcing_type == "Cosenoidal":
        force = F0 * np.cos(omega * t)
    else:  # Escalón
        force = F0 if t >= 0 else 0.0

    a = (force - c * v - k * x) / m
    return np.array([v, a]), force


def rk4_solve(m, c, k, F0, omega, forcing_type, x0, v0, t_max, n_points):
    t = np.linspace(0, t_max, n_points)
    dt = t[1] - t[0]

    y = np.zeros((n_points, 2))
    y[0] = [x0, v0]
    force_values = np.zeros(n_points)

    for i in range(n_points - 1):
        ti = t[i]
        yi = y[i]

        k1, f1 = system_derivatives(yi, ti, m, c, k, F0, omega, forcing_type)
        k2, _ = system_derivatives(yi + 0.5 * dt * k1, ti + 0.5 * dt, m, c, k, F0, omega, forcing_type)
        k3, _ = system_derivatives(yi + 0.5 * dt * k2, ti + 0.5 * dt, m, c, k, F0, omega, forcing_type)
        k4, _ = system_derivatives(yi + dt * k3, ti + dt, m, c, k, F0, omega, forcing_type)

        y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        force_values[i] = f1

    _, force_values[-1] = system_derivatives(y[-1], t[-1], m, c, k, F0, omega, forcing_type)
    return t, y[:, 0], y[:, 1], force_values


def natural_frequency(m, k):
    return math.sqrt(k / m) if m > 0 and k > 0 else np.nan


def damping_ratio(m, c, k):
    wn = natural_frequency(m, k)
    if wn == 0 or np.isnan(wn):
        return np.nan
    return c / (2 * math.sqrt(k * m))


def response_amplitude(r, zeta, F0, k):
    # X / (F0/k) = 1 / sqrt((1-r^2)^2 + (2ζr)^2)
    denom = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
    return (F0 / k) / denom


def classify_damping(zeta):
    if np.isnan(zeta):
        return "Indeterminado"
    if abs(zeta) < 1e-8:
        return "No amortiguado"
    if zeta < 1:
        return "Subamortiguado"
    if abs(zeta - 1) < 0.03:
        return "Cercano al crítico"
    return "Sobreamortiguado"


def steady_state_estimate(t, x, cycles=4):
    # Aproximación sencilla: amplitud pico-pico en la parte final
    n = len(t)
    start = int(n * 0.75)
    xs = x[start:]
    if len(xs) == 0:
        return np.nan
    return 0.5 * (np.max(xs) - np.min(xs))


def build_mechanical_plot(t_idx, t, x, force, m, c, k):
    fig, ax = plt.subplots(figsize=(8, 3.5))

    wall_x = 0.0
    mass_x = 1.8 + x[t_idx]
    mass_w = 0.6
    mass_h = 0.4

    # wall
    ax.plot([wall_x, wall_x], [-0.8, 0.8], linewidth=3)

    # spring
    spring_x = np.linspace(wall_x + 0.05, mass_x, 16)
    spring_y = np.zeros_like(spring_x)
    for i in range(1, len(spring_x) - 1):
        spring_y[i] = 0.12 if i % 2 == 0 else -0.12
    spring_y[0] = 0
    spring_y[-1] = 0
    ax.plot(spring_x, spring_y, linewidth=2)

    # damper
    ax.plot([wall_x, wall_x + 0.4], [-0.35, -0.35], linewidth=2)
    ax.plot([wall_x + 0.4, wall_x + 0.4], [-0.48, -0.22], linewidth=2)
    ax.plot([wall_x + 0.4, mass_x - 0.15], [-0.35, -0.35], linewidth=2)

    # mass
    rect = plt.Rectangle((mass_x, -mass_h / 2), mass_w, mass_h, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(mass_x + mass_w / 2, 0, "m", ha="center", va="center", fontsize=12)

    # external force arrow
    fx = force[t_idx]
    arrow_len = 0.45 if fx >= 0 else -0.45
    ax.arrow(mass_x + mass_w / 2, 0.5, arrow_len, 0, head_width=0.06, head_length=0.08, length_includes_head=True)
    ax.text(mass_x + mass_w / 2, 0.62, "F(t)", ha="center", va="bottom", fontsize=11)

    ax.set_xlim(-0.2, 4.0)
    ax.set_ylim(-0.9, 0.9)
    ax.set_title(f"Esquema del sistema en t = {t[t_idx]:.2f} s")
    ax.axis("off")
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.title("Respuesta Forzada y Sistemas con Excitación Externa")
st.caption("Simulador interactivo para apoyar una exposición sobre ODE no-homogénea")

with st.sidebar:
    st.header("Parámetros")

    preset = st.selectbox(
        "Caso sugerido",
        [
            "Personalizado",
            "Caso base",
            "Amortiguamiento alto",
            "Casi resonancia",
            "Resonancia aproximada",
            "Fuerza lenta",
            "Fuerza rápida",
        ]
    )

    defaults = {
        "m": 1.0,
        "c": 0.8,
        "k": 9.0,
        "F0": 1.0,
        "omega": 2.2,
        "x0": 0.3,
        "v0": 0.0,
        "forcing_type": "Senoidal",
        "t_max": 25.0,
        "n_points": 2400,
    }

    if preset == "Caso base":
        defaults.update({"m": 1.0, "c": 0.8, "k": 9.0, "F0": 1.0, "omega": 2.2})
    elif preset == "Amortiguamiento alto":
        defaults.update({"m": 1.0, "c": 3.0, "k": 9.0, "F0": 1.0, "omega": 2.2})
    elif preset == "Casi resonancia":
        defaults.update({"m": 1.0, "c": 0.4, "k": 9.0, "F0": 1.0, "omega": 2.8})
    elif preset == "Resonancia aproximada":
        defaults.update({"m": 1.0, "c": 0.15, "k": 9.0, "F0": 1.0, "omega": 3.0})
    elif preset == "Fuerza lenta":
        defaults.update({"m": 1.0, "c": 0.8, "k": 9.0, "F0": 1.0, "omega": 0.8})
    elif preset == "Fuerza rápida":
        defaults.update({"m": 1.0, "c": 0.8, "k": 9.0, "F0": 1.0, "omega": 6.5})

    m = st.slider("m (masa)", 0.2, 5.0, float(defaults["m"]), 0.1)
    c = st.slider("c (amortiguamiento)", 0.0, 5.0, float(defaults["c"]), 0.05)
    k = st.slider("k (rigidez)", 0.5, 25.0, float(defaults["k"]), 0.5)
    F0 = st.slider("F₀ (amplitud de la fuerza)", 0.1, 5.0, float(defaults["F0"]), 0.1)
    omega = st.slider("ω (frecuencia externa)", 0.1, 10.0, float(defaults["omega"]), 0.1)

    st.divider()
    x0 = st.slider("x(0)", -2.0, 2.0, float(defaults["x0"]), 0.05)
    v0 = st.slider("x'(0)", -3.0, 3.0, float(defaults["v0"]), 0.05)
    forcing_type = st.selectbox("Tipo de forzamiento", ["Senoidal", "Cosenoidal", "Escalón"], index=["Senoidal", "Cosenoidal", "Escalón"].index(defaults["forcing_type"]))
    t_max = st.slider("Tiempo máximo", 8.0, 40.0, float(defaults["t_max"]), 1.0)
    n_points = st.slider("Resolución", 600, 4000, int(defaults["n_points"]), 200)

# -----------------------------
# Simulation
# -----------------------------
t, x, v, force = rk4_solve(m, c, k, F0, omega, forcing_type, x0, v0, t_max, n_points)
wn = natural_frequency(m, k)
zeta = damping_ratio(m, c, k)
r = omega / wn if wn > 0 else np.nan
A_est = steady_state_estimate(t, x)

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ωₙ", f"{wn:.3f} rad/s")
col2.metric("ζ", f"{zeta:.3f}")
col3.metric("r = ω/ωₙ", f"{r:.3f}")
col4.metric("Régimen", classify_damping(zeta))
col5.metric("Amplitud final aprox.", f"{A_est:.3f}")

st.latex(r"m\ddot{x} + c\dot{x} + kx = F(t)")
st.write(
    "La solución total puede interpretarse como la suma de una parte transitoria "
    "y una parte estacionaria: "
)
st.latex(r"x(t)=x_h(t)+x_p(t)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Respuesta temporal",
    "Esquema del sistema",
    "Resonancia",
    "Guía para exponer"
])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t, x, label="x(t)")
        ax.plot(t, force / max(F0, 1e-8) * max(np.max(np.abs(x)), 1), linestyle="--", alpha=0.7, label="F(t) reescalada")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Desplazamiento")
        ax.set_title("Respuesta del sistema")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.subheader("Lectura rápida")
        st.write(
            "- Al inicio suele dominar el **transitorio**.\n"
            "- Después, si hay amortiguamiento, la respuesta tiende a una oscilación **estacionaria**.\n"
            "- Si **ω ≈ ωₙ** y el amortiguamiento es pequeño, la amplitud aumenta notablemente."
        )
        if abs(r - 1) < 0.08 and zeta < 0.2:
            st.warning("Estás cerca de una condición de resonancia con poco amortiguamiento.")
        elif omega < wn:
            st.info("La fuerza externa es más lenta que la dinámica natural del sistema.")
        elif omega > wn:
            st.info("La fuerza externa es más rápida que la respuesta natural del sistema.")

    fig2, ax2 = plt.subplots(figsize=(10, 3.8))
    ax2.plot(t, v, label="Velocidad x'(t)")
    ax2.set_xlabel("Tiempo")
    ax2.set_ylabel("Velocidad")
    ax2.set_title("Velocidad del sistema")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

with tab2:
    st.write("Este panel sirve para mostrar visualmente el movimiento del sistema masa-resorte-amortiguador.")
    idx = st.slider("Selecciona el instante", 0, len(t) - 1, min(len(t) - 1, len(t) // 3), 1)
    fig3 = build_mechanical_plot(idx, t, x, force, m, c, k)
    st.pyplot(fig3, use_container_width=True)

    st.write(
        f"En ese instante: x(t) = **{x[idx]:.3f}**, x'(t) = **{v[idx]:.3f}**, F(t) = **{force[idx]:.3f}**"
    )

with tab3:
    st.write("Curva teórica de amplitud estacionaria para un forzamiento armónico.")
    r_vals = np.linspace(0.05, 3.0, 600)
    zeta_safe = max(zeta, 1e-4)
    amp_vals = response_amplitude(r_vals, zeta_safe, F0, k)

    fig4, ax4 = plt.subplots(figsize=(10, 4.5))
    ax4.plot(r_vals, amp_vals, label="Amplitud estacionaria")
    ax4.axvline(1.0, linestyle="--", linewidth=1, label="r = 1")
    ax4.axvline(r, linestyle=":", linewidth=1.5, label=f"Tu caso actual: r = {r:.2f}")
    ax4.set_xlabel("r = ω/ωₙ")
    ax4.set_ylabel("Amplitud")
    ax4.set_title("Curva de resonancia")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4, use_container_width=True)

    st.write(
        "Interpretación: cuando la frecuencia externa se acerca a la frecuencia natural del sistema, "
        "la amplitud puede crecer mucho. A mayor amortiguamiento, el pico se aplana."
    )

with tab4:
    st.subheader("Cómo usar esta app durante la exposición")
    st.write(
        "1. Empieza con el **caso base** para mostrar la ecuación y sus términos.\n"
        "2. Luego cambia a **fuerza lenta** y **fuerza rápida** para comparar comportamientos.\n"
        "3. Después usa **casi resonancia** o **resonancia aproximada** para mostrar el pico de amplitud.\n"
        "4. Usa el panel del esquema para relacionar el gráfico con el sistema físico.\n"
        "5. Explica que el amortiguamiento hace que el transitorio desaparezca más rápido."
    )

    st.subheader("Ideas que conectan con la rúbrica")
    st.write(
        "- **Fenómeno y relevancia:** muestra que la fuerza externa cambia completamente el comportamiento.\n"
        "- **Formulación y análisis:** explica el papel de m, c, k y F(t).\n"
        "- **Solución e interpretación:** conecta la curva con transitorio y estacionario.\n"
        "- **Calidad visual:** usa la simulación para apoyar la exposición oral."
    )

    st.subheader("Sugerencia académica importante")
    st.warning(
        "Úsenla como apoyo visual. El análisis matemático, las conclusiones y la interpretación técnica "
        "deben poder explicarlos ustedes sin depender de la app."
    )
