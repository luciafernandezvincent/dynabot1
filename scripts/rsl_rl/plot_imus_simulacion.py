import argparse
import csv
import torch
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Agregar IsaacLab al path
sys.path.append(os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.expanduser("~/IsaacLab/source"))

from isaaclab.app import AppLauncher

# 1. Configurar el lanzador (Debe ir PRIMERO obligatoriamente)
parser = argparse.ArgumentParser(description="Script para comparar IMUs en Dynabot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# REEMPLAZA ESTA LÍNEA por la ruta real a tu entorno de Dynabot
# from tu_modulo_config_path import VelocityRoughEnvCfg

def main():
    print("🚀 Inicializando entorno de simulación...")

    # TODO: Configura tu entorno aquí
    # cfg = VelocityRoughEnvCfg()
    # cfg.scene.num_envs = 1
    # env = ... (crear tu entorno)

    # Configurar archivo CSV
    csv_file = "imu_simulation.csv"
    fieldnames = ["step", "gyro_centro_y", "gyro_adelante_y", "accel_centro_z", "accel_adelante_z"]

    print(f"📝 Guardando datos en {csv_file}")

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print("Recolectando datos (500 pasos)... No cierres la ventana.")

        for step in range(500):
            try:
                # Generar acciones (ajusta según tu entorno)
                # actions = torch.sin(...) * 0.4

                # Avanzar simulación
                # env.step(actions)
                # env.render()

                # Extraer datos de IMUs
                # imu_centro = env.unwrapped.scene["imu_center"].data
                # imu_adelante = env.unwrapped.scene["imu_front"].data

                # Guardar directamente a CSV
                writer.writerow({
                    "step": step,
                    "gyro_centro_y": 0.0,  # imu_centro.gyro[0, 1].item()
                    "gyro_adelante_y": 0.0,  # imu_adelante.gyro[0, 1].item()
                    "accel_centro_z": 9.81,  # imu_centro.accel[0, 2].item()
                    "accel_adelante_z": 9.81,  # imu_adelante.accel[0, 2].item()
                })
                csvfile.flush()  # Asegurar que se guarda en tiempo real

                if (step + 1) % 50 == 0:
                    print(f"  ✓ {step + 1}/500 pasos")

            except Exception as e:
                print(f"❌ Error en paso {step}: {e}")
                break

    print("✅ Simulación terminada. Datos guardados.")

    # 3. Cargar y graficar datos
    print("Generando gráficos comparativos...")
    df = pd.read_csv(csv_file)

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