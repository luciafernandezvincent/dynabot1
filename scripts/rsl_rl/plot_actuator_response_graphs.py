import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "actuator_response_results"
OUTPUT_DIR = os.path.join(BASE, "compare_sim_real")

# (sim_joint_name, real_joint_key, real_csv_key)
JOINT_MAP = [
    ("base_to_front_right_shoulder", "FRshoulder", "FRshoulder"),
    ("front_right_shoulder_to_arm",  "FRarm",      "FRarm"),
    ("front_right_arm_to_hand", "FRfoot",     "FRfoot"),
]


def center_ylim_on_zero(ax):
    ymin, ymax = ax.get_ylim()
    bound = max(abs(ymin), abs(ymax))
    ax.set_ylim(-bound, bound)


def set_ylim_to_curves(ax, series, xlim, margin=0.0):
    """Set ylim to the actual min/max the curves take within the visible x-range."""
    xmin, xmax = xlim
    values = []
    for x, y in series:
        mask = (x >= xmin) & (x <= xmax)
        if np.any(mask):
            values.append(y[mask])
    ymin, ymax = np.concatenate(values).min(), np.concatenate(values).max()
    ax.set_ylim(ymin - margin, ymax + margin)


def find_step_index(values: np.ndarray, threshold_frac: float = 0.1) -> int:
    """Return the first index where the signal changes abruptly (step onset)."""
    diffs = np.abs(np.diff(values))
    total_range = np.ptp(values)
    threshold = threshold_frac * total_range if total_range > 1e-6 else 0.05
    candidates = np.where(diffs > threshold)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def load_sim(sim_joint: str, sim_joint_graph, t0: float | None = None):
    df = pd.read_csv(f"actuator_response_results/sim_dyna/{sim_joint}/actuator_response_{sim_joint}.csv")
    df = df[df["joint"] == sim_joint_graph].reset_index(drop=True)

    time = df["time"].to_numpy(dtype=float)
    targets = df["joint_pos_target"].to_numpy(dtype=float)

    if t0 is None:
        step_idx = find_step_index(targets) # First sample where joint_pos_target changes from initial
        t0 = time[step_idx]
    step_idx = int(np.argmin(np.abs(time - t0)))
    pos0 = df["joint_pos"].iloc[step_idx]
    target0 = targets[step_idx]

    return {
        "time":   time - t0,
        "pos":    np.rad2deg(df["joint_pos"].to_numpy(dtype=float) - pos0),
        "target": np.rad2deg(targets - target0),
    }, t0


def load_real(real_key: str, csv_key: str, t0: float | None = None):
    df = pd.read_csv(f"actuator_response_results/real_dyna/log_step_curve_{csv_key}.csv")

    req_df  = df[["timestamp_request",  f"{real_key}_request"]].dropna()
    meas_df = df[["timestamp_measured", f"{real_key}_measured"]].dropna()

    t_req   = req_df["timestamp_request"].to_numpy(dtype=float)
    req_deg = req_df[f"{real_key}_request"].to_numpy(dtype=float)

    t_meas   = meas_df["timestamp_measured"].to_numpy(dtype=float)
    meas_deg = meas_df[f"{real_key}_measured"].to_numpy(dtype=float)

    if t0 is None:
        step_idx = find_step_index(req_deg)
        t0 = t_req[step_idx]
    req_step_idx = int(np.argmin(np.abs(t_req - t0)))
    req0_deg = req_deg[req_step_idx]
    meas_step_idx = int(np.argmin(np.abs(t_meas - t0)))
    meas0_deg = meas_deg[meas_step_idx]

    return {
        "time_req":  t_req  - t0,
        "time_meas": t_meas - t0,
        "req": req_deg  - req0_deg,
        "meas": meas_deg - meas0_deg,
    }, t0


def plot_joint_position(sim_joint: str, real_key: str, csv_key: str):
    
    fig, axs = plt.subplots(3, 1,figsize=(13, 10))
    # Moved joint
    sim,  t0_sim  = load_sim(sim_joint, sim_joint)
    real, t0_real = load_real(real_key, csv_key)
    axs[0].plot(sim["time"], sim["target"], color="tab:blue",   linestyle="--", label="Sim — target")
    axs[0].plot(sim["time"], sim["pos"],    color="tab:blue",                   label="Sim — position")
    axs[0].plot(real["time_req"],  real["req"],  color="tab:orange", linestyle="--", label="Real — request")
    axs[0].plot(real["time_meas"], real["meas"], color="tab:orange",                 label="Real — measured")

    axs[0].set_xlabel("Time relative to step [s]")
    axs[0].set_ylabel("Position change [deg]")
    axs[0].set_title(f"Actuator response — {sim_joint}")
    axs[0].autoscale(tight=True)
    axs[0].set_xlim(-0.2, 1)
    set_ylim_to_curves(axs[0], [
        (sim["time"],      sim["target"]),
        (sim["time"],      sim["pos"]),
        (real["time_req"],  real["req"]),
        (real["time_meas"], real["meas"]),
    ], (-0.2, 1), margin=5.0)
    axs[0].legend()
    axs[0].grid(True)

    # Other joints
    filtered_joint_map = [joint for joint in JOINT_MAP if sim_joint not in joint]
    i = 1
    for other_sim_joint, other_real_key, _ in filtered_joint_map:
        sim,  _ = load_sim(sim_joint, other_sim_joint, t0=t0_sim)
        real, _ = load_real(other_real_key, csv_key, t0=t0_real)
        axs[i].plot(sim["time"], sim["target"], color="tab:blue",   linestyle="--", label="Sim — target")
        axs[i].plot(sim["time"], sim["pos"],    color="tab:blue",                   label="Sim — position")
        axs[i].plot(real["time_req"],  real["req"],  color="tab:orange", linestyle="--", label="Real — request")
        axs[i].plot(real["time_meas"], real["meas"], color="tab:orange",                 label="Real — measured")

        axs[i].set_xlabel("Time relative to step [s]")
        axs[i].set_ylabel("Position change [deg]")
        axs[i].set_title(f"Actuator response — {other_sim_joint}")
        axs[i].autoscale(tight=True)
        axs[i].set_xlim(-0.2, 1)
        center_ylim_on_zero(axs[i])
        axs[i].legend()
        axs[i].grid(True)
        i += 1

    # Save graph
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{sim_joint}_compare.png")
    fig.savefig(out_path, dpi=300)
    #plt.show()
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def main():
    for sim_joint, real_key, csv_key in JOINT_MAP:
        plot_joint_position(sim_joint, real_key, csv_key)

    # plot_joint_position("base_to_front_right_shoulder", "FRshoulder", "FRshoulder")
    # plot_joint_position("front_right_shoulder_to_arm",  "FRarm",      "FRarm")
    # plot_joint_position("front_right_arm_to_hand", "FRfoot",     "FRfoot")


if __name__ == "__main__":
    main()
