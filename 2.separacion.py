"""
Paso 2: separacion en sesiones.

Para cada sujeto:
  1. Carga la señal filtrada (paso 1) y su archivo de tags.
  2. Verifica que haya 8 tags, en orden creciente y dentro del rango del EEG.
     Si algo falla, avisa y saltea al sujeto.
  3. Corta la señal en las 4 sesiones y guarda cada una en un CSV:
     resultados/02_separacion/<sujeto>/<sujeto>_<n>_<sesion>.csv
  4. Hace una figura con TP9 completo y las sesiones marcadas en colores.

Al final guarda una tabla con la duración de cada sesión por sujeto.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

COLORES = ["tab:gray", "tab:green", "tab:red", "tab:purple"]
N_TAGS = 2 * len(cfg.SESIONES)          # inicio y fin de cada sesión = 8

cfg.DIR_SEPARACION.mkdir(parents=True, exist_ok=True)
filas = []                              # para la tabla de duraciones

for sujeto in cfg.SUJETOS_ACTIVOS:
    print(f"{sujeto}:")

    # 1. Cargar señal filtrada y tags (una columna de tiempos unix en s, sin encabezado)
    datos = np.load(cfg.DIR_FILTRADO / f"{sujeto}_filtrado.npz")
    t = datos["t"]
    x = datos["x"]
    tags = np.loadtxt(cfg.archivo_tags(sujeto), ndmin=1)

    # 2. Verificar los tags
    if len(tags) != N_TAGS:
        print(f"  AVISO: hay {len(tags)} tags (se esperaban {N_TAGS}). Se saltea.")
        continue
    if np.any(np.diff(tags) <= 0):
        print("  AVISO: los tags no están en orden creciente. Se saltea.")
        continue
    if tags[0] < t[0] or tags[-1] > t[-1]:
        print("  AVISO: hay tags fuera del rango del EEG. Se saltea.")
        continue

    carpeta = cfg.DIR_SEPARACION / sujeto
    carpeta.mkdir(exist_ok=True)
    fila = {"sujeto": sujeto}

    # 3. Cortar y guardar cada sesión en CSV
    for i, nombre in enumerate(cfg.SESIONES):
        inicio = tags[2 * i]
        fin = tags[2 * i + 1]
        dentro = (t >= inicio) & (t <= fin)

        sesion = pd.DataFrame(x[dentro], columns=cfg.CANALES)
        sesion.insert(0, cfg.COL_TIEMPO, t[dentro])                 # tiempo unix [s]
        sesion.insert(1, "tiempo_sesion", t[dentro] - inicio)       # s desde el inicio de la sesión
        sesion.to_csv(carpeta / f"{sujeto}_{i + 1}_{nombre}.csv",
                      index=False, float_format="%.6f")

        duracion = (fin - inicio) / 60
        fila[f"{nombre}_min"] = round(duracion, 2)
        print(f"  {i + 1}. {nombre}: {duracion:.2f} min, {dentro.sum()} muestras")

    filas.append(fila)

    # 4. Figura: TP9 completo con las sesiones sombreadas
    canal = cfg.CANALES.index("TP9")
    minutos = (t - t[0]) / 60

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(minutos, x[:, canal], lw=0.3, color="k")
    for i, nombre in enumerate(cfg.SESIONES):
        inicio = (tags[2 * i] - t[0]) / 60
        fin = (tags[2 * i + 1] - t[0]) / 60
        ax.axvspan(inicio, fin, color=COLORES[i], alpha=0.25, label=f"{i + 1}. {nombre}")

    # Que un artefacto grande no aplaste toda la señal
    limite = np.percentile(np.abs(x[:, canal]), 99.5)
    ax.set_ylim(-1.5 * limite, 1.5 * limite)
    ax.set_xlabel("Tiempo desde el inicio del registro [min]")
    ax.set_ylabel("TP9 [µV]")
    ax.set_title(f"{sujeto} – TP9 filtrado y sesiones")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(carpeta / f"sesiones_{sujeto}.png", dpi=150)
    plt.close(fig)

# Tabla de duraciones
tabla = pd.DataFrame(filas)
tabla.to_csv(cfg.DIR_SEPARACION / "duracion_sesiones.csv", index=False)
print("\nDuración de cada sesión [min]:")
print(tabla.to_string(index=False))