"""
Paso 4: PSD y potencia por bandas en cada estímulo (sesión de música).

Para cada sujeto, sesión y canal de CANALES_ANALISIS:
  1. Corta la sesión en ventanas de 1 s (igual que el paso 3) y se queda con las
     aceptadas según resultados/03_artefactos/ventanas.csv.
  2. Calcula la PSD de cada ventana (Welch, ventana Hann de 1 s -> resolución 1 Hz)
     y las promedia: PSD del estímulo.
  3. Con esa PSD promedio calcula, para cada banda (theta, alfa, beta):
       potencia absoluta  = integral de la PSD en la banda            [µV²]
       potencia relativa  = potencia de la banda / potencia 1–45 Hz   [sin unidad]
       cambio_pct         = % de cambio de la relativa respecto a sin_musica
  4. Guarda también la potencia de cada ventana (sirve para la estadística).

Guarda en resultados/04_psd/:
  - psd_promedio.csv:      sujeto, sesion, canal, frecuencia_Hz, psd_uV2_Hz
  - potencia_bandas.csv:   una fila por sujeto, sesión, canal y banda
  - potencia_ventanas.csv: una fila por ventana aceptada, canal y banda
  - psd_<sujeto>.png:      PSD de los 4 estímulos, un panel por canal
  - bandas_<sujeto>.png:   potencia relativa por estímulo, un panel por banda
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import config as cfg

DIR_ARTEFACTOS = cfg.DIR_RESULTADOS / "03_artefactos"
DIR_PSD = cfg.DIR_RESULTADOS / "04_psd"
DIR_PSD.mkdir(parents=True, exist_ok=True)

COLORES = ["tab:gray", "tab:green", "tab:red", "tab:purple"]
COLORES_BANDAS = {"theta": "tab:blue", "alfa": "tab:orange", "beta": "gold"}
fs = cfg.FS_FINAL
muestras_por_ventana = int(cfg.DURACION_VENTANA * fs)


def potencia(f, psd, banda):
    """Integral de la PSD entre banda[0] y banda[1] [µV²]."""
    df = f[1] - f[0]
    dentro = (f >= banda[0]) & (f <= banda[1])
    return psd[..., dentro].sum(axis=-1) * df


archivo_ventanas = DIR_ARTEFACTOS / "ventanas.csv"
if not archivo_ventanas.exists():
    raise FileNotFoundError(f"No existe {archivo_ventanas}. Correr primero el paso 3.")
ventanas_info = pd.read_csv(archivo_ventanas)

filas_psd = []
filas_bandas = []
filas_ventanas = []

for sujeto in cfg.SUJETOS_ACTIVOS:
    print(f"{sujeto}:")
    psd_sujeto = {}                         # (sesion, canal) -> PSD promedio, para las figuras

    for i, nombre in enumerate(cfg.SESIONES):
        # 1. Cargar la sesión y cortarla en ventanas, igual que en el paso 3
        ruta = cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{nombre}.csv"
        x = pd.read_csv(ruta)[cfg.CANALES].to_numpy()
        n_ventanas = len(x) // muestras_por_ventana
        x = x[:n_ventanas * muestras_por_ventana]
        ventanas = x.reshape(n_ventanas, muestras_por_ventana, len(cfg.CANALES))

        for canal in cfg.CANALES_ANALISIS:
            j = cfg.CANALES.index(canal)
            info = ventanas_info[(ventanas_info["sujeto"] == sujeto)
                                 & (ventanas_info["sesion"] == nombre)
                                 & (ventanas_info["canal"] == canal)]
            if len(info) != n_ventanas:
                raise ValueError(f"{sujeto} {nombre} {canal}: el paso 3 tiene {len(info)} "
                                 f"ventanas y acá hay {n_ventanas}. Volver a correr el paso 3.")
            aceptadas = info["aceptada"].to_numpy()
            limpias = ventanas[aceptadas, :, j]          # (ventanas aceptadas x muestras)
            print(f"  {nombre} {canal}: {len(limpias)} de {n_ventanas} ventanas aceptadas")
            if len(limpias) == 0:
                continue

            # 2. PSD de cada ventana y promedio
            f, psd_ventanas = signal.welch(limpias, fs=fs, nperseg=cfg.NPERSEG_WELCH, axis=-1)
            psd = psd_ventanas.mean(axis=0)
            psd_sujeto[(nombre, canal)] = psd

            for frec, valor in zip(f, psd):
                filas_psd.append({"sujeto": sujeto, "sesion": nombre, "canal": canal,
                                  "frecuencia_Hz": frec, "psd_uV2_Hz": valor})

            # 3. Potencia por banda de la PSD promedio
            total = potencia(f, psd, cfg.BANDA_TOTAL)
            for banda, limites in cfg.BANDAS.items():
                absoluta = potencia(f, psd, limites)
                filas_bandas.append({
                    "sujeto": sujeto, "sesion": nombre, "canal": canal, "banda": banda,
                    "n_ventanas_aceptadas": len(limpias),
                    "pct_ventanas_descartadas": 100 * (1 - len(limpias) / n_ventanas),
                    "potencia_abs_uV2": absoluta,
                    "potencia_rel": absoluta / total,
                })

            # 4. Potencia de cada ventana aceptada
            total_v = potencia(f, psd_ventanas, cfg.BANDA_TOTAL)
            numeros = info["ventana"].to_numpy()[aceptadas]
            inicios = info["inicio_sesion_s"].to_numpy()[aceptadas]
            for banda, limites in cfg.BANDAS.items():
                absoluta_v = potencia(f, psd_ventanas, limites)
                for k in range(len(limpias)):
                    filas_ventanas.append({
                        "sujeto": sujeto, "sesion": nombre, "canal": canal, "banda": banda,
                        "ventana": numeros[k], "inicio_sesion_s": inicios[k],
                        "potencia_abs_uV2": absoluta_v[k],
                        "potencia_rel": absoluta_v[k] / total_v[k],
                    })

    # --- Figura 1: PSD de los 4 estímulos, un panel por canal ---
    canales = cfg.CANALES_ANALISIS
    fig, axs = plt.subplots(1, len(canales), figsize=(7 * len(canales), 5),
                            sharey=True, squeeze=False)
    for ax, canal in zip(axs[0], canales):
        for banda, (lo, hi) in cfg.BANDAS.items():
            ax.axvspan(lo, hi, color=COLORES_BANDAS[banda], alpha=0.12)
            ax.text((lo + hi) / 2, 0.98, banda, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
        for i, nombre in enumerate(cfg.SESIONES):
            if (nombre, canal) in psd_sujeto:
                en_rango = (f >= cfg.BANDA_TOTAL[0]) & (f <= cfg.BANDA_TOTAL[1])
                ax.semilogy(f[en_rango], psd_sujeto[(nombre, canal)][en_rango],
                            color=COLORES[i], lw=1.5, label=nombre)
        ax.set_xlim(cfg.BANDA_TOTAL)
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_title(canal)
        ax.grid(alpha=0.3, which="both")
    axs[0, 0].set_ylabel("PSD [µV²/Hz]")
    for ax in axs[0]:                    # leyenda en el primer panel que tenga curvas
        if ax.get_lines():
            ax.legend()
            break
    fig.suptitle(f"{sujeto} – PSD promedio por estímulo (ventanas de 1 s aceptadas)")
    fig.tight_layout()
    fig.savefig(DIR_PSD / f"psd_{sujeto}.png", dpi=150)
    plt.close(fig)

# --- Tablas ---
tabla_bandas = pd.DataFrame(filas_bandas)

# % de cambio de la potencia relativa respecto a la sesión de referencia
referencia = tabla_bandas[tabla_bandas["sesion"] == cfg.SESION_REFERENCIA]
referencia = referencia[["sujeto", "canal", "banda", "potencia_rel"]]
referencia = referencia.rename(columns={"potencia_rel": "potencia_rel_referencia"})
tabla_bandas = tabla_bandas.merge(referencia, on=["sujeto", "canal", "banda"], how="left")
tabla_bandas["cambio_pct"] = 100 * (tabla_bandas["potencia_rel"]
                                    / tabla_bandas["potencia_rel_referencia"] - 1)
tabla_bandas = tabla_bandas.drop(columns="potencia_rel_referencia")

pd.DataFrame(filas_psd).round(6).to_csv(DIR_PSD / "psd_promedio.csv", index=False)
tabla_bandas.round(4).to_csv(DIR_PSD / "potencia_bandas.csv", index=False)
pd.DataFrame(filas_ventanas).round(6).to_csv(DIR_PSD / "potencia_ventanas.csv", index=False)

# --- Figura 2: potencia relativa por estímulo, un panel por banda ---
canales = cfg.CANALES_ANALISIS
ancho = 0.8 / len(canales)
posiciones = np.arange(len(cfg.SESIONES))
for sujeto in cfg.SUJETOS_ACTIVOS:
    t = tabla_bandas[tabla_bandas["sujeto"] == sujeto]
    fig, axs = plt.subplots(1, len(cfg.BANDAS), figsize=(5 * len(cfg.BANDAS), 4.5))
    for ax, banda in zip(axs, cfg.BANDAS):
        for c, canal in enumerate(canales):
            d = t[(t["banda"] == banda) & (t["canal"] == canal)].set_index("sesion")
            d = d.reindex(cfg.SESIONES)
            x = posiciones + (c - (len(canales) - 1) / 2) * ancho
            barras = ax.bar(x, d["potencia_rel"], ancho, label=canal)
            for barra, cambio, nombre in zip(barras, d["cambio_pct"], cfg.SESIONES):
                if nombre != cfg.SESION_REFERENCIA and not np.isnan(cambio):
                    ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                            f"{cambio:+.0f}%", ha="center", va="bottom", fontsize=7,
                            rotation=90 if len(canales) > 2 else 0)
        ax.set_xticks(posiciones)
        ax.set_xticklabels(cfg.SESIONES, rotation=20)
        ax.set_title(f"{banda} {cfg.BANDAS[banda][0]:g}–{cfg.BANDAS[banda][1]:g} Hz")
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("Potencia relativa (banda / 1–45 Hz)")
    axs[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for ax in axs:                       # lugar arriba para los porcentajes
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    fig.suptitle(f"{sujeto} – Potencia relativa por estímulo "
                 f"(% = cambio respecto a {cfg.SESION_REFERENCIA})")
    fig.tight_layout()
    fig.savefig(DIR_PSD / f"bandas_{sujeto}.png", dpi=150)
    plt.close(fig)

# --- Resumen en pantalla ---
print("\nPotencia relativa y % de cambio respecto a la referencia:")
print(tabla_bandas[["sujeto", "canal", "banda", "sesion", "potencia_rel", "cambio_pct"]]
      .round(3).to_string(index=False))