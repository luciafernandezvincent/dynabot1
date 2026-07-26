import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from scipy.stats import norm

def plot_imu_data(csv_path, output_dir=None):
    """Plot IMU data from CSV file."""
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'IMU Measurements', fontsize=16)

    # Angular velocity plot
    ax = axes[0]
    ax.plot(df.index, df['gyro_x'], label='X', alpha=0.7)
    ax.plot(df.index, df['gyro_y'], label='Y', alpha=0.7)
    ax.plot(df.index, df['gyro_z'], label='Z', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linear acceleration plot
    ax = axes[1]
    ax.plot(df.index, df['accel_x'], label='X', alpha=0.7)
    ax.plot(df.index, df['accel_y'], label='Y', alpha=0.7)
    ax.plot(df.index, df['accel_z'], label='Z', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Linear Acceleration (g)')
    ax.set_title('Linear Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'imu_measurements.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print statistics
    print(f"\nIMU Data Statistics:")
    print(f"Number of samples: {len(df)}")
    print(f"\nAngular Velocity (deg/s):")
    print(f"  X - Mean: {df['gyro_x'].mean():.4f}, Std: {df['gyro_x'].std():.4f}, Range: [{df['gyro_x'].min():.4f}, {df['gyro_x'].max():.4f}]")
    print(f"  Y - Mean: {df['gyro_y'].mean():.4f}, Std: {df['gyro_y'].std():.4f}, Range: [{df['gyro_y'].min():.4f}, {df['gyro_y'].max():.4f}]")
    print(f"  Z - Mean: {df['gyro_z'].mean():.4f}, Std: {df['gyro_z'].std():.4f}, Range: [{df['gyro_z'].min():.4f}, {df['gyro_z'].max():.4f}]")
    print(f"\nLinear Acceleration (g):")
    print(f"  X - Mean: {df['accel_x'].mean():.4f}, Std: {df['accel_x'].std():.4f}, Range: [{df['accel_x'].min():.4f}, {df['accel_x'].max():.4f}]")
    print(f"  Y - Mean: {df['accel_y'].mean():.4f}, Std: {df['accel_y'].std():.4f}, Range: [{df['accel_y'].min():.4f}, {df['accel_y'].max():.4f}]")
    print(f"  Z - Mean: {df['accel_z'].mean():.4f}, Std: {df['accel_z'].std():.4f}, Range: [{df['accel_z'].min():.4f}, {df['accel_z'].max():.4f}]")

    # Save mean/std of each variable to a .txt file with 10 decimals
    variables = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    stats_output_path = output_dir / 'imu_stats.txt'
    with open(stats_output_path, 'w') as f:
        f.write(f"Number of samples: {len(df)}\n\n")
        for var in variables:
            f.write(f"{var} - Mean: {df[var].mean():.10f}, Std: {df[var].std():.10f}\n")
    print(f"Stats saved to: {stats_output_path}")

    # Histograms with fitted normal distribution for each IMU variable
    xlabels = {
        'gyro_x': 'Angular velocity x (deg/s)',
        'gyro_y': 'Angular velocity y (deg/s)',
        'gyro_z': 'Angular velocity z (deg/s)',
        'accel_x': 'Linear acceleration x (g)',
        'accel_y': 'Linear acceleration y (g)',
        'accel_z': 'Linear acceleration z (g)',
    }
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(18, 10))
    fig_hist.suptitle(f'IMU Measurements Distributions', fontsize=16)

    for ax, var in zip(axes_hist.flat, variables):
        mean = df[var].mean()
        std = df[var].std()
        x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
        y = norm.pdf(x, loc=mean, scale=std)
        ax.hist(df[var], density=True, alpha=0.6, color='C0')
        ax.plot(x, y, color='C1', label=f"N({mean:.4f}, {std:.4f})")
        ax.set_title(xlabels[var])
        ax.set_xlabel(xlabels[var])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    hist_output_path = output_dir / 'imu_histograms.png'
    plt.savefig(hist_output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram plot saved to: {hist_output_path}")

    #plt.show()



if __name__ == "__main__":
    # Default to most recent imu.csv with data
    # sensor_results_dir = Path(__file__).parent.parent.parent / "sensor_results"
    # sensor_results_dir = Path(__file__).parent.parent.parent /"sensor_results/20260415-140421"
    # if len(sys.argv) > 1:
    #     csv_path = sys.argv[1]
    # else:
    #     # Find the most recent imu.csv
    #     imu_files = list(sensor_results_dir.glob("imu.csv"))
    #     imu_files_with_data = [f for f in imu_files if f.stat().st_size > 100]
    #     if imu_files_with_data:
    #         csv_path = sorted(imu_files_with_data)[-1]
    #     else:
    #         print("No imu.csv files found with data")
    #         sys.exit(1)
    csv_path = "sensor_results/log_imu_static_100.csv"
    print(f"Plotting IMU data from: {csv_path}")
    plot_imu_data(csv_path)
