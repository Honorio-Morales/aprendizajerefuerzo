
import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

st.title("🕹️ CartPole-v1 Q-Learning Agent")
st.sidebar.header("Configuración")

# Intentar cargar la Q-Table entrenada
q_table = None
if os.path.exists('q_table.npy'):
    q_table = np.load('q_table.npy')
    st.sidebar.success("✅ Q-Table cargada correctamente. El agente jugará con lo aprendido.")
else:
    st.sidebar.warning("⚠️ No se encontró 'q_table.npy'. El agente usará acciones aleatorias. ¡Asegúrate de subir este archivo!")

# Parámetros para la discretización
ALPHA = 0.1
GAMMA = 0.99
BINS = 20

os_low = [-4.8, -2.0, -0.418, -3.5]
os_high = [4.8, 2.0, 0.418, 3.5]

def discretize_state(state):
    ratios = [(state[i] - os_low[i]) / (os_high[i] - os_low[i]) for i in range(len(state))]
    new_state = [int(round((BINS - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(BINS - 1, max(0, x)) for x in new_state]
    return tuple(new_state)

# Botón para ejecutar demo
if st.button('Ejecutar Demo del Agente'):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_cont, _ = env.reset()
    done = False
    total_reward = 0
    
    img_placeholder = st.empty()
    
    while not done:
        # Decidir acción basada en la Q-Table si existe, si no, usar random
        if q_table is not None:
            state_idx = discretize_state(state_cont)
            action = int(np.argmax(q_table[state_idx]))
        else:
            action = env.action_space.sample()
            
        state_cont, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        img = env.render()
        img_placeholder.image(img, caption=f"Reward: {total_reward}")
        time.sleep(0.02)
        
    st.success(f"¡Simulación terminada! Reward total: {total_reward}")
    env.close()
