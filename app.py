import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide", page_title="Algoritmo SARSA en FrozenLake")

st.title("Implementación del Algoritmo SARSA en FrozenLake-v1")
st.markdown("Una aplicación interactiva para explorar el algoritmo SARSA en el entorno `FrozenLake-v1` de Gymnasium.")

# --- Hiperparámetros configurables --- #
st.sidebar.header("Configuración de Hiperparámetros")

alpha = st.sidebar.slider("Tasa de Aprendizaje (alpha)", 0.01, 1.0, 0.1, 0.01)
gamma = st.sidebar.slider("Factor de Descuento (gamma)", 0.01, 0.99, 0.99, 0.01)
epsilon_inicial = st.sidebar.slider("Épsilon Inicial (exploración)", 0.0, 1.0, 1.0, 0.01)
epsilon_min = st.sidebar.slider("Épsilon Mínimo", 0.0, 0.1, 0.01, 0.001)
epsilon_decay = st.sidebar.slider("Tasa de Decaimiento de Épsilon", 0.0001, 0.01, 0.001, 0.0001)
num_episodios = st.sidebar.slider("Número de Episodios", 100, 10000, 5000, 100)
max_pasos_por_episodio = st.sidebar.slider("Máx. Pasos por Episodio", 50, 500, 100, 10)

# --- Funciones SARSA --- #

def elegir_accion_epsilon_greedy(estado, tabla_q, epsilon, num_acciones):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_acciones - 1) # Acción aleatoria
    else:
        return np.argmax(tabla_q[estado, :]) # Mejor acción


@st.cache_resource # Cachea el entorno para evitar reinicializaciones costosas
def obtener_entorno():
    return gym.make('FrozenLake-v1', is_slippery=True)


def entrenar_sarsa(
    alpha,
    gamma,
    epsilon_inicial,
    epsilon_min,
    epsilon_decay,
    num_episodios,
    max_pasos_por_episodio,
):
    entorno = obtener_entorno()
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n
    tabla_q = np.zeros((num_estados, num_acciones))
    recompensas_por_episodio = []
    epsilon = epsilon_inicial

    progreso_bar = st.progress(0, text="Entrenando SARSA...")

    for episodio in range(num_episodios):
        estado_actual, info = entorno.reset()
        terminado = False
        truncado = False
        recompensa_acumulada = 0
        pasos_en_episodio = 0

        accion_actual = elegir_accion_epsilon_greedy(estado_actual, tabla_q, epsilon, num_acciones)

        while not terminado and not truncado and pasos_en_episodio < max_pasos_por_episodio:
            siguiente_estado, recompensa, terminado, truncado, info = entorno.step(accion_actual)
            recompensa_acumulada += recompensa

            accion_siguiente = elegir_accion_epsilon_greedy(siguiente_estado, tabla_q, epsilon, num_acciones)

            antiguo_q = tabla_q[estado_actual, accion_actual]
            siguiente_q_valor = tabla_q[siguiente_estado, accion_siguiente]

            nuevo_q_valor = antiguo_q + alpha * (recompensa + gamma * siguiente_q_valor - antiguo_q)
            tabla_q[estado_actual, accion_actual] = nuevo_q_valor

            estado_actual = siguiente_estado
            accion_actual = accion_siguiente
            pasos_en_episodio += 1

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        recompensas_por_episodio.append(recompensa_acumulada)
        progreso_bar.progress((episodio + 1) / num_episodios, text=f"Episodio {episodio + 1}/{num_episodios}")

    entorno.close()
    progreso_bar.empty()
    return tabla_q, recompensas_por_episodio

# --- Botón para iniciar el entrenamiento --- #
if st.sidebar.button("Iniciar Entrenamiento SARSA"):
    st.subheader("Resultados del Entrenamiento")
    tabla_q_final, recompensas = entrenar_sarsa(
        alpha,
        gamma,
        epsilon_inicial,
        epsilon_min,
        epsilon_decay,
        num_episodios,
        max_pasos_por_episodio,
    )

    st.success("¡Entrenamiento SARSA completado!")

    st.markdown("### Tabla Q Final (primeras filas)")
    st.dataframe(tabla_q_final[:5, :]) # Mostrar solo las primeras 5 filas para evitar saturación

    st.markdown("### Gráfica de Recompensa Acumulada por Episodio")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(recompensas)
    ax1.set_title('Recompensa Acumulada por Episodio durante el Entrenamiento SARSA')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa Acumulada')
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown("### Gráfica de Recompensa Promedio Móvil")
    ventana = 100
    recompensa_suavizada = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(recompensa_suavizada)
    ax2.set_title(f'Recompensa Promedio Móvil (Ventana {ventana}) durante el Entrenamiento SARSA')
    ax2.set_xlabel('Episodio (suavizado)')
    ax2.set_ylabel('Recompensa Promedio')
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("### Análisis de la Convergencia")
    st.markdown("**Interpretación del Aprendizaje (Curva ascendente y estabilización):**")
    st.markdown("Si la curva de recompensa promedio móvil muestra una **tendencia ascendente** y luego **se estabiliza** en un valor alto, indica que el agente está aprendiendo y convergiendo a una política efectiva. Esto significa que está encontrando rutas para alcanzar el objetivo (G) y evitar los agujeros (H).")
    st.markdown("**Inestabilidad o Estancamiento y relación con Hiperparámetros:**")
    st.markdown("Si las curvas son muy fluctuantes (inestabilidad) o se mantienen bajas (estancamiento), es posible que los hiperparámetros (alpha, gamma, epsilon) necesiten ajustarse. Una `alpha` muy alta puede causar inestabilidad, mientras que una `epsilon` insuficiente puede llevar a un estancamiento en un óptimo local.")

else:
    st.info("Ajusta los hiperparámetros en la barra lateral izquierda y haz clic en 'Iniciar Entrenamiento SARSA' para comenzar.")

st.markdown("--- ")
st.markdown("**Nota sobre el Entorno FrozenLake-v1:**")
st.markdown("El entorno `FrozenLake-v1` (`is_slippery=True`) simula un suelo congelado donde el agente debe navegar desde el inicio (S) hasta el objetivo (G). Las casillas 'F' son seguras, 'H' son agujeros. La propiedad `is_slippery=True` introduce aleatoriedad: la acción deseada solo ocurre con una probabilidad, y en otras ocasiones el agente se desliza a una casilla adyacente aleatoria. Esto hace que el problema sea más desafiante y requiere que el agente aprenda una política robusta.")