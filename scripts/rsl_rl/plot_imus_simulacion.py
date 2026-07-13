import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch

from isaaclab.app import AppLauncher

# 1. Configurar el lanzador de la App de Isaac Sim obligatoriamente primero
parser = argparse.ArgumentParser(description="Script para comparar IMUs en Dynabot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. El resto de las importaciones de Isaac Lab van ACÁ abajo
from isaaclab.environments.managed_grid_env import ManagedGridEnv
# Asegúrate de importar TU clase de configuración donde agregaste las dos IMUs:
from tu_modulo_config_path import VelocityRoughEnvCfg  

def main():
    # Creamos el entorno usando tu configuración modoficada
    cfg = VelocityRoughEnvCfg()
    env = ManagedGridEnv(cfg)
    
    env.reset()
    logs = []
    
    print("Corriendo simulación para recolectar datos...")
    # Corremos 500 pasos de simulación (aprox 2.5 segundos de datos a 200Hz)
    for step in range(500):
        # Generamos acciones aleatorias o sinusoidales suaves para mover al perro
        actions = torch.sin(torch.ones(env.num_envs, env.action_manager.total_action_dim) * step * 0.1) * 0.5
        env.step(actions)
        
        # Extraemos los datos brutos de la escena de simulación
        imu_centro = env.unwrapped.scene["imu_center"].data
        imu_adelante = env.unwrapped.scene["imu_front"].data
        
        # Guardamos en un diccionario el paso actual
        logs.append({
            "step": step,
            "gyro_centro_y": imu_centro.gyro[0, 1].item(),      # Velocidad Angular Y (Pitch)
            "gyro_adelante_y": imu_adelante.gyro[0, 1].item(),  # Deberían ser iguales
            "accel_centro_z": imu_centro.accel[0, 2].item(),    # Aceleración Vertical Z
            "accel_adelante_z": imu_adelante.accel[0, 2].item(),# ¡Esta debería tener más ruido/picos!
        })
        
    env.close()
    
    # 3. Procesar y Graficar con Matplotlib
    df = pd.DataFrame(logs)
    
    # Crear ventana con dos gráficos (Subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Gráfico 1: Giroscopio (Velocidad Angular en Y)
    ax1.plot(df["step"], df["gyro_centro_y"], label="IMU Centro (base_link)", alpha=0.8, color="tab:blue")
    ax1.plot(df["step"], df["gyro_adelante_y"], label="IMU Adelante (Head Offset)", linestyle="--", alpha=0.8, color="tab:orange")
    ax1.set_title("Comparativa: Velocidad Angular (Pitch - Eje Y)")
    ax1.set_ylabel("Rad/s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Acelerómetro (Aceleración Lineal en Z)
    ax2.plot(df["step"], df["accel_centro_z"], label="IMU Centro (base_link)", alpha=0.7, color="tab:blue")
    ax2.plot(df["step"], df["accel_adelante_z"], label="IMU Adelante (Head Offset)", alpha=0.7, color="tab:red")
    ax2.set_title("Comparativa: Aceleración Vertical (Eje Z)")
    ax2.set_xlabel("Pasos de Simulación (Steps)")
    ax2.set_ylabel("m/s²")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Mostrando gráficos...")
    plt.show()

if __name__ == "__main__":
    main()
    simulation_app.close()