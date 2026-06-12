import streamlit as st
import gymnasium as gym
import numpy as np
import time
import os

st.set_page_config(
    page_title="CartPole Q-Learning",
    page_icon=":material/smart_toy:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS adicional para mejorar la apariencia del visualizador
st.markdown("""
<style>
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title(":material/smart_toy: CartPole-v1: Agente Q-Learning")
st.markdown("Observa cómo un agente de Inteligencia Artificial equilibra un poste en un carro en movimiento, tomando decisiones basadas en su tabla de aprendizaje.")

st.sidebar.header(":material/settings: Configuración", divider="blue")

# Intentar cargar la Q-Table entrenada
q_table = None
if os.path.exists('q_table.npy'):
    q_table = np.load('q_table.npy')
    st.sidebar.success("Modelo cargado. El agente utilizará su política aprendida.", icon=":material/memory:")
else:
    st.sidebar.warning("No se encontró 'q_table.npy'. El agente usará acciones aleatorias.", icon=":material/warning:")

sim_speed = st.sidebar.slider("Velocidad de simulación (FPS)", min_value=10, max_value=60, value=30, step=5)
delay = 1.0 / sim_speed

st.sidebar.markdown("---")
st.sidebar.markdown("""
**¿Cómo funciona?**
El agente observa 4 variables del entorno en tiempo real:
1. Posición del carro
2. Velocidad del carro
3. Ángulo del poste
4. Velocidad angular

En cada fracción de segundo, consulta su **Q-Table** para decidir la mejor acción: empujar a la **Izquierda** o a la **Derecha**.
""")

# Parámetros para la discretización
BINS = 20
os_low = [-4.8, -2.0, -0.418, -3.5]
os_high = [4.8, 2.0, 0.418, 3.5]

def discretize_state(state):
    ratios = [(state[i] - os_low[i]) / (os_high[i] - os_low[i]) for i in range(len(state))]
    new_state = [int(round((BINS - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(BINS - 1, max(0, x)) for x in new_state]
    return tuple(new_state)

col_main, col_side = st.columns([2, 1], gap="large")

with col_side:
    with st.container(border=True):
        st.subheader(":material/sports_esports: Control")
        start_btn = st.button("Iniciar Simulación", type="primary", use_container_width=True, icon=":material/play_arrow:")
        
    with st.container(border=True):
        st.subheader(":material/monitoring: Telemetría en Vivo")
        col_m1, col_m2 = st.columns(2)
        metric_reward = col_m1.empty()
        metric_action = col_m2.empty()
        metric_angle = col_m1.empty()
        metric_pos = col_m2.empty()
        
        metric_reward.metric("Recompensa", "0")
        metric_action.metric("Acción", "-")
        metric_angle.metric("Ángulo", "0.0°")
        metric_pos.metric("Posición", "0.0")
        
        st.divider()
        status_placeholder = st.empty()
        status_placeholder.info("Esperando inicio...", icon=":material/hourglass_empty:")

with col_main:
    with st.container(border=True):
        st.subheader(":material/desktop_windows: Visualizador del Entorno")
        img_placeholder = st.empty()
        
        if not start_btn:
            try:
                env = gym.make('CartPole-v1', render_mode='rgb_array')
                env.reset()
                img_placeholder.image(env.render(), use_container_width=True)
                env.close()
            except:
                pass

if start_btn:
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_cont, _ = env.reset()
    done = False
    total_reward = 0
    
    status_placeholder.info("Simulando entorno...", icon=":material/sync:")
    
    while not done:
        if q_table is not None:
            state_idx = discretize_state(state_cont)
            action = int(np.argmax(q_table[state_idx]))
        else:
            action = env.action_space.sample()
            
        action_text = "➡️ Derecha" if action == 1 else "⬅️ Izquierda"
            
        state_cont, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Actualizar métricas
        angle_deg = state_cont[2] * (180.0 / np.pi)
        
        metric_reward.metric("Recompensa", f"{total_reward:.0f}")
        metric_action.metric("Acción", action_text)
        metric_angle.metric("Ángulo", f"{angle_deg:.1f}°")
        metric_pos.metric("Posición", f"{state_cont[0]:.2f}")
        
        img = env.render()
        img_placeholder.image(img, use_container_width=True)
        
        time.sleep(delay)
        
    status_placeholder.success("¡Episodio finalizado!", icon=":material/flag:")
    env.close()
