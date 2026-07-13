import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_imu_data(csv_path, output_dir=None):
    """Plot IMU data from CSV file."""
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'IMU Data from {Path(csv_path).parent.name}', fontsize=16)

    # Angular velocity plot
    ax = axes[0, 0]
    ax.plot(df.index, df['av_x'], label='X', alpha=0.7)
    ax.plot(df.index, df['av_y'], label='Y', alpha=0.7)
    ax.plot(df.index, df['av_z'], label='Z', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linear acceleration plot
    ax = axes[0, 1]
    ax.plot(df.index, df['la_x'], label='X', alpha=0.7)
    ax.plot(df.index, df['la_y'], label='Y', alpha=0.7)
    ax.plot(df.index, df['la_z'], label='Z', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Linear Acceleration (m/s²)')
    ax.set_title('Linear Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Angular velocity magnitude
    ax = axes[1, 0]
    av_mag = np.sqrt(df['av_x']**2 + df['av_y']**2 + df['av_z']**2)
    ax.plot(av_mag, linewidth=1, color='C0', alpha=0.7)
    ax.fill_between(df.index, av_mag, alpha=0.3, color='C0')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Magnitude (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.grid(True, alpha=0.3)

    # Linear acceleration magnitude
    ax = axes[1, 1]
    la_mag = np.sqrt(df['la_x']**2 + df['la_y']**2 + df['la_z']**2)
    ax.plot(la_mag, linewidth=1, color='C1', alpha=0.7)
    ax.fill_between(df.index, la_mag, alpha=0.3, color='C1')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Magnitude (m/s²)')
    ax.set_title('Linear Acceleration Magnitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'imu_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print statistics
    print(f"\nIMU Data Statistics:")
    print(f"Number of samples: {len(df)}")
    print(f"\nAngular Velocity (rad/s):")
    print(f"  X - Mean: {df['av_x'].mean():.4f}, Std: {df['av_x'].std():.4f}, Range: [{df['av_x'].min():.4f}, {df['av_x'].max():.4f}]")
    print(f"  Y - Mean: {df['av_y'].mean():.4f}, Std: {df['av_y'].std():.4f}, Range: [{df['av_y'].min():.4f}, {df['av_y'].max():.4f}]")
    print(f"  Z - Mean: {df['av_z'].mean():.4f}, Std: {df['av_z'].std():.4f}, Range: [{df['av_z'].min():.4f}, {df['av_z'].max():.4f}]")
    print(f"\nLinear Acceleration (m/s²):")
    print(f"  X - Mean: {df['la_x'].mean():.4f}, Std: {df['la_x'].std():.4f}, Range: [{df['la_x'].min():.4f}, {df['la_x'].max():.4f}]")
    print(f"  Y - Mean: {df['la_y'].mean():.4f}, Std: {df['la_y'].std():.4f}, Range: [{df['la_y'].min():.4f}, {df['la_y'].max():.4f}]")
    print(f"  Z - Mean: {df['la_z'].mean():.4f}, Std: {df['la_z'].std():.4f}, Range: [{df['la_z'].min():.4f}, {df['la_z'].max():.4f}]")

    plt.show()

if __name__ == "__main__":
    # Default to most recent imu.csv with data
    # sensor_results_dir = Path(__file__).parent.parent.parent / "sensor_results"
    sensor_results_dir = Path(__file__).parent.parent.parent /"sensor_results/20260415-140421"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find the most recent imu.csv
        imu_files = list(sensor_results_dir.glob("imu.csv"))
        imu_files_with_data = [f for f in imu_files if f.stat().st_size > 100]
        if imu_files_with_data:
            csv_path = sorted(imu_files_with_data)[-1]
        else:
            print("No imu.csv files found with data")
            sys.exit(1)

    print(f"Plotting IMU data from: {csv_path}")
    plot_imu_data(csv_path)
