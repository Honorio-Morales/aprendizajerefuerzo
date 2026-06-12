import streamlit as st
import gymnasium as gym
import numpy as np
import time
import os

st.set_page_config(
    page_title="CartPole Q-Learning",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.header(":material/smart_toy: CartPole-v1 Q-Learning", divider="blue")
st.markdown("Demostración interactiva del agente entrenado equilibrando el poste.")

st.sidebar.header(":material/settings: Configuración", divider="blue")

# Intentar cargar la Q-Table entrenada
q_table = None
if os.path.exists('q_table.npy'):
    q_table = np.load('q_table.npy')
    st.sidebar.success("Modelo cargado correctamente. El agente jugará de forma inteligente.", icon=":material/check_circle:")
else:
    st.sidebar.warning("No se encontró 'q_table.npy'. El agente usará acciones aleatorias.", icon=":material/warning:")

# Parámetros para la discretización
BINS = 20
os_low = [-4.8, -2.0, -0.418, -3.5]
os_high = [4.8, 2.0, 0.418, 3.5]

def discretize_state(state):
    ratios = [(state[i] - os_low[i]) / (os_high[i] - os_low[i]) for i in range(len(state))]
    new_state = [int(round((BINS - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(BINS - 1, max(0, x)) for x in new_state]
    return tuple(new_state)

col_metrics, col_vis = st.columns([1, 2], gap="large")

with col_metrics:
    st.markdown("### Controles")
    start_btn = st.button("Iniciar Simulación", type="primary", use_container_width=True, icon=":material/play_arrow:")
    
    st.markdown("### Métricas en Vivo")
    metric_placeholder = st.empty()
    metric_placeholder.metric(label="Recompensa Total", value="0")
    status_placeholder = st.empty()

with col_vis:
    st.markdown("### Visualización")
    img_placeholder = st.empty()
    # Mostrar una imagen inicial
    if not start_btn:
        try:
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            env.reset()
            img_placeholder.image(env.render(), use_container_width=True)
            env.close()
        except:
            st.info("Visualizador listo.", icon=":material/visibility:")

if start_btn:
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_cont, _ = env.reset()
    done = False
    total_reward = 0
    
    status_placeholder.info("Simulación en curso...", icon=":material/sync:")
    
    while not done:
        if q_table is not None:
            state_idx = discretize_state(state_cont)
            action = int(np.argmax(q_table[state_idx]))
        else:
            action = env.action_space.sample()
            
        state_cont, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        img = env.render()
        img_placeholder.image(img, use_container_width=True)
        metric_placeholder.metric(label="Recompensa Total", value=f"{total_reward:.0f}", delta=f"+{reward:.0f}")
        time.sleep(0.02)
        
    status_placeholder.success("¡Simulación terminada!", icon=":material/done_all:")
    env.close()

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado con Streamlit y Gymnasium")
