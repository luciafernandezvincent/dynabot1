#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación de filtros Madgwick vs Kalman para datos de IMU de simulación.
No requiere ROS, funciona directamente con datos de sensor_results.
"""

import numpy as np
import csv
from datetime import datetime
from pathlib import Path
import math
import json


class Madgwick:
    """Implementación simple del filtro Madgwick sin dependencias externas"""

    def __init__(self, sampleperiod=1.0/100, beta=0.1):
        self.sampleperiod = sampleperiod
        self.beta = beta  # Algorithm gain

    def updateIMU(self, q, gyr, acc):
        """
        Update orientation estimate using gyroscope and accelerometer.

        Args:
            q: Quaternion [w, x, y, z]
            gyr: Gyroscope data [gx, gy, gz] in rad/s
            acc: Accelerometer data [ax, ay, az] in m/s²

        Returns:
            Updated quaternion
        """
        q = np.array(q)
        gyr = np.array(gyr)
        acc = np.array(acc)

        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm == 0:
            return q
        acc = acc / acc_norm

        # Extract quaternion components
        w, x, y, z = q

        # Compute objective function and Jacobian
        f = np.array([
            2*(x*z - w*y) - acc[0],
            2*(w*x + y*z) - acc[1],
            2*(0.5 - x**2 - y**2) - acc[2]
        ])

        J = np.array([
            [-2*y, 2*z, -2*w, 2*x],
            [2*x, 2*w, 2*z, 2*y],
            [0, -4*x, -4*y, 0]
        ])

        # Compute gradient descent step
        step = J.T @ f
        step = step / np.linalg.norm(step)

        # Compute rate of change of quaternion
        q_dot = 0.5 * np.array([
            -x*gyr[0] - y*gyr[1] - z*gyr[2],
            w*gyr[0] + y*gyr[2] - z*gyr[1],
            w*gyr[1] + z*gyr[0] - x*gyr[2],
            w*gyr[2] + x*gyr[1] - y*gyr[0]
        ])

        # Apply feedback
        q_dot = q_dot - self.beta * step

        # Integrate to yield quaternion
        q = q + q_dot * self.sampleperiod
        q = q / np.linalg.norm(q)

        return q


class IMUFilterComparison:
    """Compare Madgwick vs Kalman filter for IMU orientation estimation"""

    def __init__(self, sample_rate=100, accel_offset=None, gyro_offset=None):
        """
        Inicializar comparador de filtros.

        Args:
            sample_rate: Frecuencia de muestreo en Hz
            accel_offset: Offset de calibración para aceleración [x, y, z]
            gyro_offset: Offset de calibración para giroscopio [x, y, z]
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

        # Calibration offsets (como en RealInterface::imu_cb)
        self.accel_offset = accel_offset if accel_offset is not None else [0.0, 0.0, 0.0]
        self.gyro_offset = gyro_offset if gyro_offset is not None else [0.0, 0.0, 0.0]

        # Madgwick filter
        self.madgwick = Madgwick(sampleperiod=self.dt)
        self.q_madgwick = np.array([1.0, 0.0, 0.0, 0.0])

        # Kalman filter
        self.q_kalman = np.array([1.0, 0.0, 0.0, 0.0])
        self.P_kalman = np.eye(4) * 0.1  # State covariance
        self.P_kalman_prev = self.P_kalman.copy()

        # Kalman parameters - tuned for IMU
        self.Q_kalman = np.eye(4) * 0.001  # Process noise (gyro drift) - menor = más confianza en gyro
        self.R_kalman = np.eye(3) * 0.1    # Measurement noise (accel) - menor = más confianza en accel

        # Convergence detection
        self.kalman_converged = False
        self.convergence_threshold = 1e-6
        self.convergence_counter = 0
        self.convergence_frames_needed = 100

        # Parameter file
        self.params_file = Path.home() / ".kalman_params.json"
        self.load_kalman_params()

        if self.kalman_converged:
            print("✓ Parámetros Kalman convergidos cargados")
        else:
            print("⚙ Modo calibración Kalman activado")

        # CSV setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = Path.home() / f"imu_filter_comparison_{timestamp}.csv"
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow([
            'step',
            'gx', 'gy', 'gz',
            'ax', 'ay', 'az',
            'madgwick_roll_deg', 'madgwick_pitch_deg', 'madgwick_yaw_deg',
            'kalman_roll_deg', 'kalman_pitch_deg', 'kalman_yaw_deg',
            'kalman_converged', 'kalman_P_change', 'convergence_counter'
        ])
        self.csv_file_handle.flush()
        print(f"📝 Guardando comparación de filtros en: {self.csv_file}")

    def load_kalman_params(self):
        """Load Kalman parameters from JSON if they exist"""
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r') as f:
                    params = json.load(f)
                self.P_kalman = np.array(params['P_kalman'])
                self.Q_kalman = np.array(params['Q_kalman'])
                self.R_kalman = np.array(params['R_kalman'])
                self.kalman_converged = True
            except Exception as e:
                print(f"⚠ No se pudieron cargar parámetros Kalman: {e}")

    def save_kalman_params(self):
        """Save converged Kalman parameters to JSON"""
        try:
            params = {
                'P_kalman': self.P_kalman.tolist(),
                'Q_kalman': self.Q_kalman.tolist(),
                'R_kalman': self.R_kalman.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.params_file, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"✓ Parámetros Kalman guardados en: {self.params_file}")
        except Exception as e:
            print(f"✗ Error guardando parámetros Kalman: {e}")

    def check_convergence(self):
        """Check if Kalman filter has converged"""
        if self.kalman_converged:
            return

        P_change = np.linalg.norm(self.P_kalman - self.P_kalman_prev)

        if P_change < self.convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        self.P_kalman_prev = self.P_kalman.copy()

        if self.convergence_counter >= self.convergence_frames_needed:
            self.kalman_converged = True
            self.save_kalman_params()
            print("✓ Filtro Kalman convergió!")

    def quat_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        w, x, y, z = q

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def quat_to_rotmat(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
        ])

    def project_gravity(self, q):
        """Project gravity to body frame using quaternion"""
        R = self.quat_to_rotmat(q)
        g_world = np.array([0.0, 0.0, -9.81])
        g_body = R.T @ g_world
        return g_body

    def kalman_predict(self, gyro, dt=None):
        """Kalman filter prediction step with gyro integration"""
        if dt is None:
            dt = self.dt

        if dt <= 0:
            return

        omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * self.quat_multiply(self.q_kalman, omega_quat)
        self.q_kalman = self.q_kalman + q_dot * dt
        self.q_kalman = self.q_kalman / np.linalg.norm(self.q_kalman)

        if not self.kalman_converged:
            F = np.eye(4)
            self.P_kalman = F @ self.P_kalman @ F.T + self.Q_kalman
            self.check_convergence()

    def kalman_update(self, accel):
        """Extended Kalman Filter update step with accelerometer"""
        accel_norm = accel / (np.linalg.norm(accel) + 1e-8)

        w, x, y, z = self.q_kalman

        # Expected measurement: gravity vector in body frame
        # h(q) = [2(xz - wy), 2(wx + yz), 2(0.5 - x² - y²)]
        h = np.array([
            2 * (x * z - w * y),
            2 * (w * x + y * z),
            2 * (0.5 - x**2 - y**2)
        ])

        # Innovation (residual)
        y_innov = accel_norm - h

        # Jacobian H: dh/dq (3x4 matrix)
        H = np.array([
            [-2*y, 2*z, -2*w, 2*x],
            [2*x, 2*w, 2*z, 2*y],
            [-4*x, -4*y, 0, 0]
        ])

        # Kalman gain
        S = H @ self.P_kalman @ H.T + self.R_kalman
        try:
            K = self.P_kalman @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 3))
            return

        # State update
        dq = K @ y_innov
        self.q_kalman = self.q_kalman + dq
        self.q_kalman = self.q_kalman / np.linalg.norm(self.q_kalman)

        # Covariance update
        if not self.kalman_converged:
            I = np.eye(4)
            self.P_kalman = (I - K @ H) @ self.P_kalman

    def quat_multiply(self, q1, q2):
        """Multiply two quaternions: q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def preprocess_imu(self, gyro_raw, accel_raw):
        """
        Preprocesar datos de IMU como en RealInterface::imu_cb.

        Args:
            gyro_raw: Giroscopio crudo [gx, gy, gz] en deg/s (o unidades del sensor)
            accel_raw: Aceleración cruda [ax, ay, az] en m/s²

        Returns:
            gyro_processed: [gx, gy, gz] en rad/s
            accel_processed: [ax, ay, az] en m/s² (remapeado)
        """
        # Restar offsets de calibración
        accel_x = accel_raw[0] - self.accel_offset[0]
        accel_y = accel_raw[1] - self.accel_offset[1]
        accel_z = accel_raw[2] - self.accel_offset[2]

        gyro_x = gyro_raw[0] - self.gyro_offset[0]
        gyro_y = gyro_raw[1] - self.gyro_offset[1]
        gyro_z = gyro_raw[2] - self.gyro_offset[2]

        # Convertir giroscopio a radianes por segundo (como en el robot)
        # El robot hace: (gyro_z/180)*M_PI, etc.
        gyro_processed = np.array([
            (gyro_z / 180.0) * np.pi,
            (gyro_x / 180.0) * np.pi,
            (gyro_y / 180.0) * np.pi
        ])

        # Remapear aceleración (como en el robot)
        # El robot hace: x = -accel_z, y = -accel_x, z = -accel_y
        accel_processed = np.array([
            -accel_z,
            -accel_x,
            -accel_y
        ])

        return gyro_processed, accel_processed

    def process_imu_data(self, gyro_raw, accel_raw, step, preprocess=True):
        """
        Procesar datos de IMU con ambos filtros.

        Args:
            gyro_raw: Array [gx, gy, gz] en deg/s (crudo del sensor)
            accel_raw: Array [ax, ay, az] en m/s² (crudo del sensor)
            step: Número de paso/frame
            preprocess: Si aplicar preprocesamiento como en el robot real

        Returns:
            Dict con resultados de ambos filtros
        """
        gyro_raw = np.array(gyro_raw)
        accel_raw = np.array(accel_raw)

        # Preprocesar como en RealInterface::imu_cb
        if preprocess:
            gyro, accel = self.preprocess_imu(gyro_raw, accel_raw)
        else:
            gyro = gyro_raw
            accel = accel_raw

        # Update Madgwick
        self.q_madgwick = self.madgwick.updateIMU(self.q_madgwick, gyr=gyro, acc=accel)

        # Update Kalman: Predict + Update
        self.kalman_predict(gyro)
        self.kalman_update(accel)

        # Convert to Euler angles
        madgwick_roll, madgwick_pitch, madgwick_yaw = self.quat_to_euler(self.q_madgwick)
        kalman_roll, kalman_pitch, kalman_yaw = self.quat_to_euler(self.q_kalman)

        # Convert to degrees
        madgwick_roll_deg = math.degrees(madgwick_roll)
        madgwick_pitch_deg = math.degrees(madgwick_pitch)
        madgwick_yaw_deg = math.degrees(madgwick_yaw)
        kalman_roll_deg = math.degrees(kalman_roll)
        kalman_pitch_deg = math.degrees(kalman_pitch)
        kalman_yaw_deg = math.degrees(kalman_yaw)

        # Calculate P change
        P_change = np.linalg.norm(self.P_kalman - self.P_kalman_prev)

        # Save to CSV
        self.csv_writer.writerow([
            step,
            f"{gyro[0]:.6f}", f"{gyro[1]:.6f}", f"{gyro[2]:.6f}",
            f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
            f"{madgwick_roll_deg:.6f}", f"{madgwick_pitch_deg:.6f}", f"{madgwick_yaw_deg:.6f}",
            f"{kalman_roll_deg:.6f}", f"{kalman_pitch_deg:.6f}", f"{kalman_yaw_deg:.6f}",
            f"{int(self.kalman_converged)}", f"{P_change:.9f}", f"{self.convergence_counter}"
        ])
        self.csv_file_handle.flush()

        return {
            'madgwick': {
                'roll_deg': madgwick_roll_deg,
                'pitch_deg': madgwick_pitch_deg,
                'yaw_deg': madgwick_yaw_deg,
                'q': self.q_madgwick.copy()
            },
            'kalman': {
                'roll_deg': kalman_roll_deg,
                'pitch_deg': kalman_pitch_deg,
                'yaw_deg': kalman_yaw_deg,
                'q': self.q_kalman.copy()
            }
        }

    def close(self):
        """Close CSV file"""
        if self.csv_file_handle and not self.csv_file_handle.closed:
            self.csv_file_handle.close()
            print(f"✓ Datos guardados en: {self.csv_file}")

    def __del__(self):
        self.close()


def process_imu_csv(csv_file, accel_offset=None, gyro_offset=None):
    """
    Procesar datos de IMU desde un archivo CSV de sensor_results.

    Args:
        csv_file: Ruta al archivo CSV con datos de IMU
        accel_offset: Offset de calibración para aceleración
        gyro_offset: Offset de calibración para giroscopio
    """
    import pandas as pd

    print(f"📂 Leyendo datos de IMU desde: {csv_file}")

    # Leer CSV
    df = pd.read_csv(csv_file)
    print(f"   Total de muestras: {len(df)}")

    # Crear comparador
    filter_comp = IMUFilterComparison(
        sample_rate=100,
        accel_offset=accel_offset or [0.0, 0.0, 0.0],
        gyro_offset=gyro_offset or [0.0, 0.0, 0.0]
    )

    print("\n📊 Procesando datos de sensor_results...")

    # Procesar cada fila
    for step_num, row in enumerate(df.itertuples(index=False)):
        # Los datos del CSV ya están en rad/s y m/s²
        gyro = np.array([float(row.av_x), float(row.av_y), float(row.av_z)])
        accel = np.array([float(row.la_x), float(row.la_y), float(row.la_z)])

        # Procesar (sin preprocesamiento adicional, los datos ya están listos)
        result = filter_comp.process_imu_data(gyro, accel, step_num, preprocess=False)

        if step_num % max(1, len(df) // 10) == 0:  # Mostrar 10 veces durante el procesamiento
            print(f"  Step {step_num}/{len(df)}: Madgwick={result['madgwick']['roll_deg']:.2f}° "
                  f"| Kalman={result['kalman']['roll_deg']:.2f}°")

    # Cerrar
    filter_comp.close()
    print("✓ ¡Listo!")
    return filter_comp.csv_file


# Ejemplo de uso con datos de simulación
if __name__ == "__main__":
    print("🚀 Comparador de Filtros IMU (Madgwick vs Kalman)")
    print("=" * 60)

    # Ruta al archivo de sensor_results
    csv_file = Path.home() / "Proyectos/dynabot1/sensor_results/20260415-134534/imu.csv"

    if csv_file.exists():
        output_csv = process_imu_csv(csv_file)
        print(f"\n📊 Resultados guardados en: {output_csv}")
    else:
        print(f"✗ Archivo no encontrado: {csv_file}")
        print("\n💡 Uso manual:")
        print("   from imu_filter_comparison import process_imu_csv")
        print("   process_imu_csv('/ruta/a/imu.csv')")
