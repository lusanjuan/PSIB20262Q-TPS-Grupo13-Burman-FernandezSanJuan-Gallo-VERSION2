"""
Paso 3: rechazo de artefactos por amplitud pico a pico.

Para cada sujeto, sesión y canal:
  1. Corta la sesión separada (paso 2) en ventanas de DURACION_VENTANA (sin solapamiento).
     Las muestras del final que no completan una ventana se descartan.
  2. Calcula la amplitud pico a pico (máximo - mínimo) de cada ventana.
  3. Marca la ventana como aceptada si UMBRAL_PP_MIN <= pico a pico <= UMBRAL_PP.
     - Por encima de UMBRAL_PP (150 µV): artefacto grande (parpadeo, movimiento).
     - Por debajo de UMBRAL_PP_MIN (5 µV): ventana plana (canal sin contacto).
     Son los límites de amplitud del EEG (5-150 µV).

La señal no se modifica: solo se decide qué ventanas se usan en los pasos siguientes.

Guarda en resultados/03_artefactos/:
  - ventanas.csv: una fila por ventana y canal (sujeto, sesion, canal, ventana,
    inicio y fin en s desde el inicio de la sesión, pico a pico, aceptada).
  - descartadas.csv: % de ventanas descartadas por sujeto, sesión y canal.
  - histograma_pico_a_pico.png: pico a pico de todas las ventanas, por canal.
  - senales/<sujeto>/<sujeto>_<n>_<sesion>.png: la señal de cada sesión con los 4 canales,
    y las ventanas rechazadas coloreadas (rojo: > UMBRAL_PP; azul: < UMBRAL_PP_MIN).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

DIR_ARTEFACTOS = cfg.DIR_RESULTADOS / "03_artefactos"
DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
DIR_SENALES = DIR_ARTEFACTOS / "senales"
GUARDAR_SENALES = True       # False = no generar las figuras de cada sesión (es más rápido)
LIMITE_Y = 300               # µV: el eje vertical de esas figuras se corta en ±LIMITE_Y

muestras_por_ventana = int(cfg.DURACION_VENTANA * cfg.FS_FINAL)    # 1 s * 128 Hz = 128
filas = []


def tramos(marcadas):
    """Une ventanas marcadas seguidas en tramos (inicio, fin) en segundos, para dibujar menos."""
    resultado = []
    for k in np.flatnonzero(marcadas):
        ini, fin = k * cfg.DURACION_VENTANA, (k + 1) * cfg.DURACION_VENTANA
        if resultado and abs(resultado[-1][1] - ini) < 1e-9:
            resultado[-1][1] = fin
        else:
            resultado.append([ini, fin])
    return resultado


def figura_senal(sujeto, i, nombre, x, pp):
    """Los 4 canales de la sesión, con las ventanas rechazadas coloreadas."""
    t = np.arange(len(x)) / cfg.FS_FINAL
    fig, axs = plt.subplots(len(cfg.CANALES), 1, figsize=(20, 2.2 * len(cfg.CANALES)), sharex=True)
    for j, (ax, canal) in enumerate(zip(axs, cfg.CANALES)):
        grande = pp[:, j] > cfg.UMBRAL_PP
        plana = pp[:, j] < cfg.UMBRAL_PP_MIN
        for ini, fin in tramos(grande):
            ax.axvspan(ini, fin, color="red", alpha=0.25, lw=0)
        for ini, fin in tramos(plana):
            ax.axvspan(ini, fin, color="blue", alpha=0.25, lw=0)
        ax.plot(t, x[:, j], color="black", lw=0.3)
        ax.set_ylim(-LIMITE_Y, LIMITE_Y)
        ax.set_ylabel(f"{canal}\n[µV]")
        ax.text(1.005, 0.5, f"rojo {100 * grande.mean():.0f} %\nazul {100 * plana.mean():.0f} %",
                transform=ax.transAxes, va="center", fontsize=9)
    axs[-1].set_xlabel("Tiempo desde el inicio de la sesión [s]")
    axs[-1].set_xlim(0, t[-1])
    axs[0].set_title(f"{sujeto} - {nombre}.  Rojo: pico a pico > {cfg.UMBRAL_PP:g} µV.  "
                     f"Azul: pico a pico < {cfg.UMBRAL_PP_MIN:g} µV (canal plano).", loc="left")
    fig.tight_layout()
    (DIR_SENALES / sujeto).mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_SENALES / sujeto / f"{sujeto}_{i + 1}_{nombre}.png", dpi=120)
    plt.close(fig)


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
                    "aceptada": cfg.UMBRAL_PP_MIN <= pp[k, j] <= cfg.UMBRAL_PP,
                })
        if GUARDAR_SENALES:
            figura_senal(sujeto, i, nombre, x, pp)
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
bins = np.logspace(0, 4, 81)          # de 1 a 10000 µV, escala logarítmica
for ax, canal in zip(axs.flat, cfg.CANALES):
    t_c = tabla[tabla["canal"] == canal]
    pct = 100 * np.mean(~t_c["aceptada"])
    ax.hist(t_c["pico_a_pico_uV"], bins=bins, color="C0", alpha=0.8)
    ax.axvline(cfg.UMBRAL_PP, color="r", ls="--", label=f"umbral {cfg.UMBRAL_PP:g} µV")
    ax.axvline(cfg.UMBRAL_PP_MIN, color="r", ls=":", label=f"mínimo {cfg.UMBRAL_PP_MIN:g} µV")
    ax.axvline(t_c["pico_a_pico_uV"].median(), color="k", ls=":",
               label=f"mediana {t_c['pico_a_pico_uV'].median():.0f} µV")
    ax.set_xscale("log")
    ax.set_title(f"{canal} – se descarta el {pct:.0f}% de las ventanas")
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
print(f"\n% de ventanas descartadas ({cfg.UMBRAL_PP_MIN:g}–{cfg.UMBRAL_PP:g} µV, "
      "promedio de las 4 sesiones):")
print(resumen[cfg.CANALES].round(1).to_string())