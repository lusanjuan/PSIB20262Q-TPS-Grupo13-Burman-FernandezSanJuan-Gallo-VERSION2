"""
Paso 3: rechazo de artefactos por amplitud pico a pico.

Para cada sujeto, sesión y canal:
  1. Corta la señal en ventanas de DURACION_VENTANA (sin solapamiento).
     Las muestras del final que no completan una ventana se descartan.
  2. Calcula la amplitud pico a pico (máximo - mínimo) de cada ventana.
  3. Marca como aceptada la ventana si pico a pico <= UMBRAL_PP.

Guarda en resultados/03_artefactos/:
  - ventanas.csv: una fila por ventana y canal (sujeto, sesion, canal, ventana,
    inicio y fin en s desde el inicio de la sesión, pico a pico, aceptada).
  - descartadas.csv: % de ventanas descartadas por sujeto, sesión y canal.
  - histograma_pico_a_pico.png: pico a pico de todas las ventanas, por canal.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

DIR_ARTEFACTOS = cfg.DIR_RESULTADOS / "03_artefactos"
DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)

muestras_por_ventana = int(cfg.DURACION_VENTANA * cfg.FS_FINAL)    # 1 s * 128 Hz = 128
filas = []

for sujeto in cfg.SUJETOS_ACTIVOS:
    print(f"{sujeto}:")
    for i, nombre in enumerate(cfg.SESIONES):

        # Cargar la sesión (paso 2)
        ruta = cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{nombre}.csv"
        x = pd.read_csv(ruta)[cfg.CANALES].to_numpy()

        # 1. Cortar en ventanas: array (ventanas x muestras x canales)
        n_ventanas = len(x) // muestras_por_ventana
        x = x[:n_ventanas * muestras_por_ventana]
        ventanas = x.reshape(n_ventanas, muestras_por_ventana, len(cfg.CANALES))

        # 2. Pico a pico de cada ventana y canal: array (ventanas x canales)
        pp = ventanas.max(axis=1) - ventanas.min(axis=1)

        # 3. Una fila por ventana y canal
        for j, canal in enumerate(cfg.CANALES):
            for k in range(n_ventanas):
                filas.append({
                    "sujeto": sujeto,
                    "sesion": nombre,
                    "canal": canal,
                    "ventana": k + 1,
                    "inicio_sesion_s": k * cfg.DURACION_VENTANA,
                    "fin_sesion_s": (k + 1) * cfg.DURACION_VENTANA,
                    "pico_a_pico_uV": pp[k, j],
                    "aceptada": pp[k, j] <= cfg.UMBRAL_PP,
                })
        print(f"  {i + 1}. {nombre}: {n_ventanas} ventanas")

# --- Tabla por ventana ---
tabla = pd.DataFrame(filas)
tabla.round(3).to_csv(DIR_ARTEFACTOS / "ventanas.csv", index=False)

# --- % de ventanas descartadas por sujeto, sesión y canal ---
descartadas = (100 * (1 - tabla.groupby(["sujeto", "sesion", "canal"], sort=False)["aceptada"].mean()))
descartadas = descartadas.rename("pct_descartadas").reset_index()
descartadas.round(1).to_csv(DIR_ARTEFACTOS / "descartadas.csv", index=False)

# --- Histograma de pico a pico, un panel por canal ---
fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
bins = np.logspace(1, 4, 61)          # de 10 a 10000 µV, escala logarítmica
for ax, canal in zip(axs.flat, cfg.CANALES):
    pp = tabla.loc[tabla["canal"] == canal, "pico_a_pico_uV"]
    pct = 100 * np.mean(pp > cfg.UMBRAL_PP)
    ax.hist(pp, bins=bins, color="C0", alpha=0.8)
    ax.axvline(cfg.UMBRAL_PP, color="r", ls="--", label=f"umbral {cfg.UMBRAL_PP:g} µV")
    ax.axvline(pp.median(), color="k", ls=":", label=f"mediana {pp.median():.0f} µV")
    ax.set_xscale("log")
    ax.set_title(f"{canal} – {pct:.0f}% de las ventanas supera el umbral")
    ax.legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel(f"Amplitud pico a pico por ventana de {cfg.DURACION_VENTANA:g} s [µV]")
for ax in axs[:, 0]:
    ax.set_ylabel("Cantidad de ventanas")
fig.suptitle(f"Amplitud pico a pico – sujetos: {', '.join(cfg.SUJETOS_ACTIVOS)}")
fig.tight_layout()
fig.savefig(DIR_ARTEFACTOS / "histograma_pico_a_pico.png", dpi=150)
plt.close(fig)

# --- Resumen en pantalla ---
resumen = descartadas.pivot_table(index="sujeto", columns="canal", values="pct_descartadas")
print(f"\n% de ventanas descartadas (umbral {cfg.UMBRAL_PP:g} µV, promedio de las 4 sesiones):")
print(resumen[cfg.CANALES].round(1).to_string())