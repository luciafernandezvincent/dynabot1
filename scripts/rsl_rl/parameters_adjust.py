"""
Ajusta los parametros de actuador (stiffness, damping) para que la respuesta
en simulacion se parezca a la del robot real.
 
Para cada articulacion:
  1. Carga la curva real (medida) y detecta el escalon.
  2. Le ajusta una respuesta al escalon de 2do orden  ->  (wn, zeta).
  3. Con J = armature, despeja:
         stiffness = wn^2 * J
         damping   = 2 * zeta * wn * J
  4. (Cross-check) Ajusta tambien la curva de sim actual y reporta la
     inercia efectiva implicada (J_eff = stiffness_actual / wn_sim^2),
     para ver si tu 'armature' esta bien elegido.
 
Requisitos: numpy, pandas, scipy
"""
 
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
 
# ----------------------------------------------------------------------------
# CONFIG  -- editá esto con tus articulaciones y los parametros ACTUALES
# ----------------------------------------------------------------------------
BASE = "actuator_response_results"
 
# label, sim_joint (carpeta/csv de sim), real_key (col en csv real), csv_key (nombre archivo real),
# armature (J), stiffness_actual, damping_actual
JOINT_MAP = [
    # ("etiqueta",        "sim_joint",                     "real_key",  "csv_key",   J,     k_act, d_act)
    ("shoulder",         "base_to_front_right_shoulder",  "FRshoulder", "FRshoulder", 0.06, 80.0, 0.5),
    ("shoulder_to_arm",  "front_right_shoulder_to_arm",   "FRarm",      "FRarm",      0.06, 26.0, 2.0),
    ("arm_to_hand",      "front_right_arm_to_hand",        "FRfoot",     "FRfoot",     0.06, 26.0, 2.0),
]
 
# Ventana de tiempo (s, relativa al escalon) que se usa para ajustar.
# Conviene que cubra la subida + el asentamiento. None = usa todo lo disponible.
FIT_WINDOW = 1.0
 
 
# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def find_step_index(values: np.ndarray, threshold_frac: float = 0.1) -> int:
    """Primer indice donde la senal cambia bruscamente (inicio del escalon)."""
    diffs = np.abs(np.diff(values))
    total_range = np.ptp(values)
    threshold = threshold_frac * total_range if total_range > 1e-6 else 0.05
    candidates = np.where(diffs > threshold)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0
 
 
def second_order_step(t, wn, zeta):
    """Respuesta al escalon unitario de wn^2 / (s^2 + 2*zeta*wn*s + wn^2).
    Valida para cualquier zeta (sub-, criticamente y sobre-amortiguado)."""
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    m = t >= 0
    tt = t[m]
 
    s = np.sqrt(complex(zeta * zeta - 1.0))
    p1 = -wn * (zeta - s)
    p2 = -wn * (zeta + s)
 
    if abs(p1 - p2) < 1e-9:               # raiz doble (criticamente amortiguado)
        y[m] = 1.0 - np.exp(-wn * tt) * (1.0 + wn * tt)
    else:
        yt = 1.0 + (p2 * np.exp(p1 * tt) - p1 * np.exp(p2 * tt)) / (p1 - p2)
        y[m] = np.real(yt)
    return y
 
 
def fit_second_order(t, y, fit_window=None):
    """Ajusta y(t) = K * step(wn, zeta).  Devuelve (wn, zeta, K, R2)."""
    if fit_window is not None:
        m = t <= fit_window
        t, y = t[m], y[m]
 
    if len(t) < 5:
        return None
 
    # estimacion inicial de wn a partir del tiempo al 63% del valor final
    yf = np.median(y[-max(1, len(y) // 10):])
    if yf == 0:
        yf = 1.0
    reach = np.where(y >= 0.632 * yf)[0]
    t63 = t[reach[0]] if len(reach) > 0 and t[reach[0]] > 0 else (t[-1] / 3.0 if t[-1] > 0 else 0.1)
    wn0 = 2.0 / t63 if t63 > 0 else 20.0
 
    p0 = [wn0, 0.7, yf]
    bounds = ([0.5, 0.05, 0.3], [500.0, 5.0, 3.0])
 
    def model(tt, wn, zeta, K):
        return K * second_order_step(tt, wn, zeta)
 
    try:
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=30000)
    except Exception as e:
        print(f"    [WARN] el ajuste no convergio: {e}")
        return None
 
    yhat = model(t, *popt)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return popt[0], popt[1], popt[2], r2
 
 
# ----------------------------------------------------------------------------
# Carga de curvas (normalizadas a escalon unitario 0 -> 1)
# ----------------------------------------------------------------------------
def load_real_curve(csv_key, real_key):
    path = os.path.join(BASE, "real_dyna", f"log_step_curve_{csv_key}.csv")
    if not os.path.exists(path):
        print(f"    [SKIP] no existe el CSV real: {path}")
        return None
 
    df = pd.read_csv(path)
    req_col, meas_col = f"{real_key}_request", f"{real_key}_measured"
    req_df = df[["timestamp_request", req_col]].dropna()
    meas_df = df[["timestamp_measured", meas_col]].dropna()
 
    t_req = req_df["timestamp_request"].to_numpy(float)
    req_deg = req_df[req_col].to_numpy(float)
    t_meas = meas_df["timestamp_measured"].to_numpy(float)
    meas_deg = meas_df[meas_col].to_numpy(float)
 
    step_idx = find_step_index(req_deg)
    t0 = t_req[step_idx]
    req0 = req_deg[step_idx]
    amp = req_deg[-1] - req0                     # amplitud comandada (con signo)
    if abs(amp) < 1e-6:
        print("    [SKIP] amplitud del escalon real ~ 0")
        return None
 
    meas0 = meas_deg[int(np.argmin(np.abs(t_meas - t0)))]
    t = t_meas - t0
    y = (meas_deg - meas0) / amp                 # normalizado 0 -> 1
    m = t >= 0
    return t[m], y[m], amp
 
 
def load_sim_curve(sim_joint):
    path = os.path.join(BASE, "sim_dyna", sim_joint, f"actuator_response_{sim_joint}.csv")
    if not os.path.exists(path):
        return None
 
    df = pd.read_csv(path)
    df = df[df["joint"] == sim_joint].reset_index(drop=True)
    if df.empty or "joint_pos_target" not in df.columns:
        return None
 
    targets = df["joint_pos_target"].to_numpy(float)
    step_idx = find_step_index(targets)
    t0 = df["time"].iloc[step_idx]
    pos0 = df["joint_pos"].iloc[step_idx]
    tgt0 = targets[step_idx]
    amp = targets[-1] - tgt0
    if abs(amp) < 1e-6:
        return None
 
    t = df["time"].to_numpy(float) - t0
    y = (df["joint_pos"].to_numpy(float) - pos0) / amp
    m = t >= 0
    return t[m], y[m], amp
 
 
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("AJUSTE DE PARAMETROS DE ACTUADOR (sim -> real)")
    print("=" * 72)
 
    for label, sim_joint, real_key, csv_key, J, k_act, d_act in JOINT_MAP:
        print(f"\n--- {label} ---")
 
        # ---- ajuste de la curva REAL ----
        real = load_real_curve(csv_key, real_key)
        if real is None:
            continue
        t_r, y_r, amp_r = real
        fit_r = fit_second_order(t_r, y_r, FIT_WINDOW)
        if fit_r is None:
            continue
        wn_r, zeta_r, K_r, r2_r = fit_r
 
        print(f"  Real:  wn = {wn_r:6.2f} rad/s   zeta = {zeta_r:5.3f}   "
              f"(K = {K_r:4.2f}, R2 = {r2_r:5.3f}, escalon = {amp_r:.1f} deg)")
        if r2_r < 0.9:
            print("         [!] R2 bajo: la curva real no es un 2do orden limpio "
                  "(retardo/forma en S). Los valores son aproximados.")
 
        # ---- parametros sugeridos (J = armature) ----
        k_new = wn_r ** 2 * J
        d_new = 2.0 * zeta_r * wn_r * J
        print(f"  -> con armature (J) = {J:.4f}:")
        print(f"        stiffness = {k_new:7.2f}   (antes {k_act})")
        print(f"        damping   = {d_new:7.3f}   (antes {d_act})")
 
        # ---- cross-check con la curva de sim ACTUAL ----
        sim = load_sim_curve(sim_joint)
        if sim is not None:
            t_s, y_s, amp_s = sim
            fit_s = fit_second_order(t_s, y_s, FIT_WINDOW)
            if fit_s is not None:
                wn_s, zeta_s, K_s, r2_s = fit_s
                J_eff = k_act / (wn_s ** 2) if wn_s > 0 else float("nan")
                print(f"  Sim actual: wn = {wn_s:6.2f}   zeta = {zeta_s:5.3f}   (R2 = {r2_s:5.3f})")
                print(f"        -> inercia efectiva implicada J_eff = {J_eff:.4f}")
                if J_eff > 0 and abs(J_eff - J) / J_eff > 0.25:
                    print(f"        [!] J_eff difiere bastante de tu armature ({J}).")
                    print(f"            Para mas precision, reemplaza J por {J_eff:.4f} arriba:")
                    print(f"               stiffness = {wn_r**2 * J_eff:7.2f}")
                    print(f"               damping   = {2.0*zeta_r*wn_r*J_eff:7.3f}")
 
    print("\n" + "=" * 72)
    print("Nota: stiffness en Nm/rad, damping en Nm*s/rad. 'armature' se deja igual.")
    print("Si el R2 del real es bajo, ajusta a mano alrededor de estos valores.")
    print("=" * 72)
 
 
if __name__ == "__main__":
    main()