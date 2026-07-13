import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch

from isaaclab.app import AppLauncher

## ./isaaclab.sh -p -c "import sys; print(sys.executable)"
# 1. Configurar el lanzador (Debe ir PRIMERO obligatoriamente)
parser = argparse.ArgumentParser(description="Script para comparar IMUs en Dynabot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Importaciones de Isaac Lab (Deben ir DESPUÉS de lanzar la app)
from isaaclab.environments.manager_based_env import ManagerBasedEnv
# REEMPLAZA ESTA LÍNEA por la ruta real a tu entorno de Dynabot:
# Ejemplo: from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg as VelocityRoughEnvCfg
from tu_modulo_config_path import VelocityRoughEnvCfg  

def main():
    print("🚀 Inicializando entorno de simulación...")
    cfg = VelocityRoughEnvCfg()
    
    # Forzamos a que solo cree 1 robot para que cargue ultra rápido
    cfg.scene.num_envs = 1 
    
    # Usamos la clase base correcta de Isaac Lab
    env = ManagerBasedEnv(cfg)
    
    env.reset()
    logs = []
    
    print("Recolectando datos (500 pasos)... No cierres la ventana.")
    
    for step in range(500):
        # Generar acciones compatibles con el dispositivo correcto (CUDA/CPU)
        actions = torch.sin(torch.ones(env.num_envs, env.action_manager.total_action_dim, device=env.device) * step * 0.1) * 0.4
        
        # Avanzar la física
        env.step(actions)
        
        # ¡CRUCIAL! Renderizar el frame para que la ventana no diga "No Responde"
        env.render()
        
        # Extraer datos brutos de las IMUs de la escena
        imu_centro = env.unwrapped.scene["imu_center"].data
        imu_adelante = env.unwrapped.scene["imu_front"].data
        
        # Registrar lecturas en el diccionario
        logs.append({
            "step": step,
            "gyro_centro_y": imu_centro.gyro[0, 1].item(),      # Pitch Centro
            "gyro_adelante_y": imu_adelante.gyro[0, 1].item(),  # Pitch Adelante (Deberían ser iguales)
            "accel_centro_z": imu_centro.accel[0, 2].item(),    # Aceleración Z Centro
            "accel_adelante_z": imu_adelante.accel[0, 2].item(),# Aceleración Z Adelante (Mucho más ruidosa)
        })
        
    print("Simulación terminada. Cerrando entorno...")
    env.close()
    
    # 3. Procesar y Graficar
    df = pd.DataFrame(logs)
    print("Generando gráficos comparativos...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Gráfico de Giroscopio (Velocidad Angular)
    ax1.plot(df["step"], df["gyro_centro_y"], label="IMU Centro (base_link)", alpha=0.8, color="tab:blue")
    ax1.plot(df["step"], df["gyro_adelante_y"], label="IMU Adelante (Offset)", linestyle="--", alpha=0.8, color="tab:orange")
    ax1.set_title("Comparativa: Velocidad Angular (Pitch - Eje Y)")
    ax1.set_ylabel("Rad/s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de Acelerómetro (Aceleración Lineal)
    ax2.plot(df["step"], df["accel_centro_z"], label="IMU Centro (base_link)", alpha=0.7, color="tab:blue")
    ax2.plot(df["step"], df["accel_adelante_z"], label="IMU Adelante (Offset)", alpha=0.7, color="tab:red")
    ax2.set_title("Comparativa: Aceleración Vertical (Eje Z)")
    ax2.set_xlabel("Pasos de Simulación (Steps)")
    ax2.set_ylabel("m/s²")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    simulation_app.close()