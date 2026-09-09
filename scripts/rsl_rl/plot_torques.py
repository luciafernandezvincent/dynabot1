"""Grafica el torques.csv que produce scripts/rsl_rl/eval.py.

Por cada articulacion dibuja las dos series que guarda el eval:

  - computed: el par que pide el controlador PD.
  - applied:  el par que el modelo de actuador realmente entrega.

Donde se separan, el motor esta saturado (recorto el pedido) y la curva de applied ES el techo
real en ese instante. Ojo: con DCMotorCfg ese techo NO es la constante effort_limit, es la curva
par-velocidad, asi que baja cuando la articulacion gira rapido. La linea de effort_limit se dibuja
solo como referencia del maximo absoluto, no como el limite vigente en cada momento.

Uso:
    python scripts/rsl_rl/plot_torques.py
    python scripts/rsl_rl/plot_torques.py <ruta/al/torques.csv> --tmin 8 --tmax 12
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = "logs/rsl_rl/anymal_d_flat/sin_delay/exp_112_epochs_8/eval/torques.csv"

#: Linea horizontal de referencia, en Nm. NO es lo que usa la simulacion: los DCMotorCfg de
#: source/dynabot1/dynabot1/assets/dynabot.py tienen effort_limit=14.5, asi que la columna
#: "saturated" del CSV recorta contra 14.5 (y contra la curva par-velocidad). Poner aca el limite
#: del motor REAL sirve justamente para ver el hueco: cuantos picos que la sim da por buenos
#: quedarian fuera del alcance del hardware. Se puede cambiar con --effort-limit.
EFFORT_LIMIT_NM = 9.0

COLOR_COMPUTED = "tab:orange"
COLOR_APPLIED = "tab:blue"
COLOR_SATURATED = "tab:red"
COLOR_OVER_REF = "tab:purple"


def joint_names_from_columns(df: pd.DataFrame) -> list[str]:
    """Deriva las articulaciones de la cabecera en vez de hardcodearlas.

    El CSV trae las patas que se hayan pedido con --torque_legs, asi que la cantidad de columnas
    cambia entre corridas.
    """
    return [col[: -len("_applied_Nm")] for col in df.columns if col.endswith("_applied_Nm")]


def summarize(df: pd.DataFrame, joints: list[str], effort_limit: float) -> pd.DataFrame:
    """Tabla por articulacion: cuanto par mueve, cuanto saturo en sim y cuanto pasa la referencia.

    Las dos ultimas columnas miden cosas distintas y conviene no confundirlas: "saturacion_%" es lo
    que la simulacion recorto de verdad (contra effort_limit=14.5 y la curva par-velocidad), y
    "sobre_ref_%" es cuanto se pasa del limite de referencia que se le pase a este script.
    """
    rows = []
    for joint in joints:
        applied = df[f"{joint}_applied_Nm"]
        computed = df[f"{joint}_computed_Nm"]
        saturated = df[f"{joint}_saturated"].astype(bool)
        clip = (computed.abs() - applied.abs())[saturated]
        over_ref = applied.abs() > effort_limit
        rows.append({
            "joint": joint,
            "|applied| medio": applied.abs().mean(),
            "|applied| max": applied.abs().max(),
            "sat_sim": int(saturated.sum()),
            "saturacion_%": 100.0 * saturated.mean(),
            "recorte max [Nm]": float(clip.max()) if len(clip) else 0.0,
            f"sobre_{effort_limit:g}Nm": int(over_ref.sum()),
            "sobre_ref_%": 100.0 * over_ref.mean(),
        })
    return pd.DataFrame(rows)


def plot_torques(
    csv_path: str, out_path: str | None, tmin: float | None, tmax: float | None, effort_limit: float
):
    df = pd.read_csv(csv_path)
    joints = joint_names_from_columns(df)
    if not joints:
        raise SystemExit(f"[ERROR] {csv_path} no tiene columnas *_applied_Nm")

    summary = summarize(df, joints, effort_limit)
    print(f"\n[INFO] {csv_path}: {len(df)} steps, {len(joints)} articulaciones")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # el recorte del eje x se hace despues de resumir, para que la tabla siempre describa la
    # corrida completa aunque se este mirando una ventana chica
    if tmin is not None:
        df = df[df["time_s"] >= tmin]
    if tmax is not None:
        df = df[df["time_s"] <= tmax]

    fig, axes = plt.subplots(len(joints), 1, figsize=(13, 2.2 * len(joints)), sharex=True)
    axes = [axes] if len(joints) == 1 else list(axes)

    for ax, joint in zip(axes, joints):
        time = df["time_s"]
        applied = df[f"{joint}_applied_Nm"]
        computed = df[f"{joint}_computed_Nm"]
        saturated = df[f"{joint}_saturated"].astype(bool)

        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        for sign in (1, -1):
            ax.axhline(
                sign * effort_limit, color="0.6", lw=0.8, ls=":", zorder=0,
                label=f"referencia +-{effort_limit:g} Nm" if sign == 1 else None,
            )
        # computed va debajo y mas grueso: donde no hay saturacion queda tapado por applied, y
        # asoma exactamente en los tramos recortados
        ax.plot(time, computed, color=COLOR_COMPUTED, lw=1.8, alpha=0.9, label="computed (pedido PD)")
        ax.plot(time, applied, color=COLOR_APPLIED, lw=1.0, label="applied (entregado)")
        # dos marcas distintas a proposito: lo que la sim recorto de verdad, y lo que se pasa del
        # limite de referencia sin que la sim lo note (motor real mas debil que el configurado)
        over_ref = applied.abs() > effort_limit
        if over_ref.any():
            ax.plot(
                time[over_ref], applied[over_ref], "o", color=COLOR_OVER_REF, ms=4,
                mfc="none", mew=1.2, label=f"sobre referencia ({int(over_ref.sum())})", zorder=4,
            )
        if saturated.any():
            ax.plot(
                time[saturated], applied[saturated], "o", color=COLOR_SATURATED, ms=7,
                mfc="none", mew=1.6, label=f"saturado en sim ({int(saturated.sum())})", zorder=5,
            )

        row = summary.loc[summary["joint"] == joint].iloc[0]
        ax.set_title(
            f"{joint}  —  pico {row['|applied| max']:.1f} Nm  |  saturado en sim {row['saturacion_%']:.2f}%"
            f"  |  sobre {effort_limit:g} Nm {row['sobre_ref_%']:.2f}%",
            fontsize=10, loc="left",
        )
        ax.set_ylabel("Torque [Nm]")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Tiempo de simulacion [s]")
    fig.tight_layout()

    out_path = out_path or os.path.join(os.path.dirname(csv_path), "torques.png")
    fig.savefig(out_path, dpi=200)
    print(f"\n[SAVED] {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="torques.csv generado por eval.py")
    parser.add_argument("--out", default=None, help="PNG de salida (default: torques.png junto al CSV)")
    parser.add_argument("--tmin", type=float, default=None, help="Recorta el grafico desde este tiempo [s]")
    parser.add_argument("--tmax", type=float, default=None, help="Recorta el grafico hasta este tiempo [s]")
    parser.add_argument("--show", action="store_true", help="Abre la ventana interactiva ademas de guardar")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # guardar sin display (el eval suele correr en servidor headless)

    fig = plot_torques(args.csv, args.out, args.tmin, args.tmax, EFFORT_LIMIT_NM)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
