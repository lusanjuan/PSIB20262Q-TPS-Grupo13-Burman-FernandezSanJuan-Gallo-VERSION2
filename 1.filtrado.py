"""
Paso 1: filtrado y submuestreo.

Para cada sujeto:
  1. Carga el EEG (tiempos + 4 canales).
  2. Aplica un pasabanda Butterworth de 1 a 50 Hz, de fase cero. El ruido
     de línea (60 Hz) queda fuera de la banda de paso, así que no hace
     falta un notch.
  3. Baja la frecuencia de muestreo de 256 a 128 Hz.
  4. Guarda el resultado en resultados/01_filtrado/<sujeto>_filtrado.npz

Para el sujeto de ejemplo guarda dos figuras de TP9, antes y después:
la señal en el tiempo (10 s) y el espectro (5 min).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import config as cfg


def figuras_control(sujeto, t, x, t_f, x_f):
    canal = cfg.CANALES.index("TP9")
    inicio = t[0] + 600          # minuto 10: al principio hay saturaciones

    # --- Figura 1: 10 s de señal ---
    antes = (t >= inicio) & (t < inicio + 10)
    despues = (t_f >= inicio) & (t_f < inicio + 10)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t[antes] - inicio, x[antes, canal], lw=0.7)
    ax[0].set_title(f"{sujeto} – TP9 original ({cfg.FS_ORIGINAL} Hz)")
    ax[1].plot(t_f[despues] - inicio, x_f[despues, canal], lw=0.7, color="C1")
    ax[1].set_title(f"{sujeto} – TP9 filtrado y submuestreado ({cfg.FS_FINAL} Hz)")
    ax[0].set_ylabel("Amplitud [µV]")
    ax[1].set_ylabel("Amplitud [µV]")
    ax[1].set_xlabel("Tiempo [s] (desde el minuto 10)")
    fig.tight_layout()
    fig.savefig(cfg.DIR_FILTRADO / f"control_filtrado_{sujeto}.png", dpi=150)
    plt.close(fig)

    # --- Figura 2: espectro de 5 min (Welch con ventanas de 4 s) ---
    antes = (t >= inicio) & (t < inicio + 300)
    despues = (t_f >= inicio) & (t_f < inicio + 300)
    f, p = signal.welch(x[antes, canal], fs=cfg.FS_ORIGINAL, nperseg=4 * cfg.FS_ORIGINAL)
    f_f, p_f = signal.welch(x_f[despues, canal], fs=cfg.FS_FINAL, nperseg=4 * cfg.FS_FINAL)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f, p, lw=0.9, label="Original")
    ax.semilogy(f_f, p_f, lw=0.9, color="C1", label="Filtrado y submuestreado")
    ax.axvline(cfg.F_BAJA, color="k", ls="--", lw=0.8)
    ax.axvline(cfg.F_ALTA, color="k", ls="--", lw=0.8)
    ax.axvline(cfg.F_LINEA, color="r", ls=":", lw=0.8, label="60 Hz (red eléctrica)")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Densidad espectral de potencia [µV²/Hz]")
    ax.set_title(f"{sujeto} – Espectro de TP9 antes y después del filtrado")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.DIR_FILTRADO / f"espectro_filtrado_{sujeto}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
cfg.DIR_FILTRADO.mkdir(parents=True, exist_ok=True)

# Los filtros se diseñan una sola vez y sirven para todos los sujetos
sos_pasabanda = signal.butter(cfg.ORDEN_FILTRO, [cfg.F_BAJA, cfg.F_ALTA],
                              btype="bandpass", fs=cfg.FS_ORIGINAL, output="sos")
factor = cfg.FS_ORIGINAL // cfg.FS_FINAL      # 256 / 128 = 2

for sujeto in cfg.SUJETOS_ACTIVOS:
    print(f"{sujeto}:")

    # 1. Cargar
    df = pd.read_csv(cfg.archivo_eeg(sujeto), usecols=[cfg.COL_TIEMPO] + cfg.CANALES)
    df = df.dropna()                      # un solo NaN arruinaría todo el filtrado
    t = df[cfg.COL_TIEMPO].to_numpy()
    x = df[cfg.CANALES].to_numpy(dtype=float)

    # Controles: frecuencia de muestreo real y huecos en la grabación
    fs_real = (len(t) - 1) / (t[-1] - t[0])
    salto_max = np.diff(t).max() * 1000
    print(f"  fs estimada: {fs_real:.2f} Hz | salto máximo entre muestras: {salto_max:.1f} ms")

    # 2. Pasabanda de fase cero (ida y vuelta)
    x_filt = signal.sosfiltfilt(sos_pasabanda, x, axis=0)

    # 3. Submuestrear (resample_poly trae su propio filtro antialiasing)
    x_f = signal.resample_poly(x_filt, up=1, down=factor, axis=0)
    t_f = t[::factor]

    # 4. Guardar
    np.savez(cfg.DIR_FILTRADO / f"{sujeto}_filtrado.npz",
             t=t_f, x=x_f, canales=cfg.CANALES, fs=cfg.FS_FINAL)
    print(f"  {len(t)} -> {len(t_f)} muestras, guardado")

    figuras_control(sujeto, t, x, t_f, x_f)
    print("  figuras de control guardadas")