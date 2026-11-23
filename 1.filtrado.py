"""
Paso 1: filtrado y submuestreo.

Para cada sujeto:
  1. Carga el EEG (tiempos + 4 canales).
  2. Saca el ruido de línea de 60 Hz (notch).
  3. Aplica un pasabanda Butterworth de 1 a 50 Hz, de fase cero.
  4. Baja la frecuencia de muestreo de 256 a 128 Hz.
  5. Guarda el resultado en resultados/01_filtrado/<sujeto>_filtrado.npz

Para cada sujeto guarda dos figuras de control con los 4 canales, antes y después:
  - control_filtrado_<sujeto>.png: la señal en el tiempo (10 s)
  - espectro_filtrado_<sujeto>.png: el espectro (5 min)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import config as cfg


def figuras_control(sujeto, t, x, t_f, x_f):
    # Minuto 10 (al principio hay saturaciones); si el registro es más corto, la mitad
    inicio = t[0] + min(600, (t[-1] - t[0]) / 2)

    # --- Figura 1: 10 s de señal, una fila por canal (izquierda original, derecha filtrada) ---
    antes = (t >= inicio) & (t < inicio + 10)
    despues = (t_f >= inicio) & (t_f < inicio + 10)

    fig, axs = plt.subplots(len(cfg.CANALES), 2, figsize=(16, 2.4 * len(cfg.CANALES)), sharex=True)
    for j, canal in enumerate(cfg.CANALES):
        axs[j, 0].plot(t[antes] - inicio, x[antes, j], lw=0.7)
        axs[j, 1].plot(t_f[despues] - inicio, x_f[despues, j], lw=0.7, color="C1")
        axs[j, 0].set_ylabel(f"{canal}\n[µV]")
    axs[0, 0].set_title(f"Original ({cfg.FS_ORIGINAL} Hz)")
    axs[0, 1].set_title(f"Filtrado y submuestreado ({cfg.FS_FINAL} Hz)")
    for ax in axs[-1]:
        ax.set_xlabel("Tiempo [s] (desde el minuto 10)")
    fig.suptitle(f"{sujeto} – 10 s de señal antes y después del filtrado", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.DIR_FILTRADO / f"control_filtrado_{sujeto}.png", dpi=150)
    plt.close(fig)

    # --- Figura 2: espectro de 5 min (Welch con ventanas de 4 s), un panel por canal ---
    antes = (t >= inicio) & (t < inicio + 300)
    despues = (t_f >= inicio) & (t_f < inicio + 300)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for j, (ax, canal) in enumerate(zip(axs.flat, cfg.CANALES)):
        f, p = signal.welch(x[antes, j], fs=cfg.FS_ORIGINAL, nperseg=4 * cfg.FS_ORIGINAL)
        f_f, p_f = signal.welch(x_f[despues, j], fs=cfg.FS_FINAL, nperseg=4 * cfg.FS_FINAL)
        ax.semilogy(f, p, lw=0.9, label="Original")
        ax.semilogy(f_f, p_f, lw=0.9, color="C1", label="Filtrado y submuestreado")
        ax.axvline(cfg.F_BAJA, color="k", ls="--", lw=0.8)
        ax.axvline(cfg.F_ALTA, color="k", ls="--", lw=0.8)
        ax.axvline(cfg.F_LINEA, color="r", ls=":", lw=0.8, label="60 Hz (red eléctrica)")
        ax.set_title(canal)
    axs[0, 0].legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("Frecuencia [Hz]")
    for ax in axs[:, 0]:
        ax.set_ylabel("PSD [µV²/Hz]")
    fig.suptitle(f"{sujeto} – Espectro antes y después del filtrado (5 min)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.DIR_FILTRADO / f"espectro_filtrado_{sujeto}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
cfg.DIR_FILTRADO.mkdir(parents=True, exist_ok=True)

# Los filtros se diseñan una sola vez y sirven para todos los sujetos
b_notch, a_notch = signal.iirnotch(cfg.F_LINEA, Q=30, fs=cfg.FS_ORIGINAL)
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

    # 2 y 3. Notch + pasabanda (los dos de fase cero: ida y vuelta)
    x_filt = signal.filtfilt(b_notch, a_notch, x, axis=0)
    x_filt = signal.sosfiltfilt(sos_pasabanda, x_filt, axis=0)

    # 4. Submuestrear (resample_poly trae su propio filtro antialiasing)
    x_f = signal.resample_poly(x_filt, up=1, down=factor, axis=0)
    t_f = t[::factor]

    # 5. Guardar
    np.savez(cfg.DIR_FILTRADO / f"{sujeto}_filtrado.npz",
             t=t_f, x=x_f, canales=cfg.CANALES, fs=cfg.FS_FINAL)
    print(f"  {len(t)} -> {len(t_f)} muestras, guardado")

    figuras_control(sujeto, t, x, t_f, x_f)
    print("  figuras de control guardadas")