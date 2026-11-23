"""
Video para la presentación: reproduce un tramo de una sesión "como si fuera en vivo"
para A9 (el sujeto del paper) y para un sujeto que se comporta distinto.

No modifica nada de los pasos 1 a 9. Lee las sesiones separadas (paso 2) y los
bloques de n-back (paso 5). Guarda en resultados/video/:
  video_<A9>_vs_<otro>.mp4   (o .gif si no hay ffmpeg)
  ultimo_cuadro.png          el último cuadro, por si el video no anda en el aula
  eleccion_sujeto.csv        cambio de beta de cada sujeto (de ahí sale el contraste)

Qué muestra, para cada sujeto (una columna cada uno):
  Arriba:  los últimos 10 s de la señal del CANAL, que va pasando. En rojo los
           parpadeos detectados; en gris las ventanas de 1 s que el umbral descarta.
  Abajo:   la potencia beta relativa de cada ventana de 1 s aceptada (puntos), la
           media de la sesión sin música (línea punteada) y la media acumulada de la
           sesión que se reproduce (línea gruesa). El texto dice cuánto se aleja esa
           media de la referencia, en %.

Cómo se elige el sujeto de contraste (si SUJETO_CONTRASTE = None): se calcula para
cada sujeto el cambio de beta relativa en el CANAL entre SESION y la referencia, y
se toma el que cambia más en la dirección contraria a A9 (o, si ninguno va en contra,
el que menos cambia). La tabla queda en eleccion_sujeto.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import signal

import config as cfg

# --- Parámetros del video ---
SUJETO = "A9"
SUJETO_CONTRASTE = None        # None = se elige solo; o poner por ejemplo "A3"
SESION = "musica_ia"           # sesión que se reproduce
CANAL = "TP9"                  # canal que analizó el paper
INICIO_S = 60                  # desde qué segundo de la sesión
DURACION_S = 120               # cuántos segundos de sesión se muestran
VELOCIDAD = 2                  # 2 = el video va al doble de velocidad (120 s de sesión -> 60 s de video)
FPS = 10                       # cuadros por segundo del video
VENTANA_VISIBLE = 10           # s de señal que se ven arriba

FS = cfg.FS_FINAL
REF = cfg.SESION_REFERENCIA
BETA = cfg.BANDAS["beta"]
DIR_SALIDA = cfg.DIR_RESULTADOS / "video"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)
largo = int(cfg.DURACION_VENTANA * FS)
bloques = pd.read_csv(cfg.DIR_RESULTADOS / "05_nback" / "bloques.csv")


def leer(sujeto, sesion):
    i = cfg.SESIONES.index(sesion)
    ruta = cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{sesion}.csv"
    return pd.read_csv(ruta)[cfg.CANALES].to_numpy()


def parpadeos(x):
    """Igual que en analisis_parpadeos.py: picos negativos comunes a TP9 y TP10."""
    s = -(x[:, cfg.CANALES.index("TP9")] + x[:, cfg.CANALES.index("TP10")]) / 2
    sos = signal.butter(4, (1, 10), btype="bandpass", fs=FS, output="sos")
    s = signal.sosfiltfilt(sos, s)
    ruido = np.median(np.abs(s - np.median(s))) / 0.6745
    picos, _ = signal.find_peaks(s, height=max(60, 5 * ruido), distance=int(0.3 * FS),
                                 width=(0.03 * FS, 0.6 * FS))
    return picos / FS


def beta_por_ventana(x_canal):
    """Beta relativa de cada ventana de 1 s y si la ventana pasa el umbral pico a pico."""
    n = len(x_canal) // largo
    v = x_canal[:n * largo].reshape(n, largo)
    aceptada = (v.max(axis=1) - v.min(axis=1)) <= cfg.UMBRAL_PP
    f, psd = signal.welch(v, fs=FS, nperseg=cfg.NPERSEG_WELCH, axis=-1)
    df = f[1] - f[0]
    banda = psd[:, (f >= BETA[0]) & (f <= BETA[1])].sum(axis=1) * df
    total = psd[:, (f >= cfg.BANDA_TOTAL[0]) & (f <= cfg.BANDA_TOTAL[1])].sum(axis=1) * df
    return banda / total, aceptada


def cambio_beta(sujeto):
    """% de cambio de la beta relativa media (ventanas aceptadas) entre SESION y REF."""
    j = cfg.CANALES.index(CANAL)
    medias = []
    for s in [REF, SESION]:
        b, ok = beta_por_ventana(leer(sujeto, s)[:, j])
        medias.append(b[ok].mean() if ok.sum() >= 20 else np.nan)
    return 100 * (medias[1] / medias[0] - 1)


# ===========================================================================
# 1. Elegir el sujeto de contraste
# ===========================================================================
filas = []
for s in cfg.SUJETOS:
    try:
        filas.append({"sujeto": s, "cambio_beta_pct": cambio_beta(s)})
    except FileNotFoundError:
        pass
tabla = pd.DataFrame(filas).dropna()
tabla.round(2).to_csv(DIR_SALIDA / "eleccion_sujeto.csv", index=False)
print(f"Cambio de beta relativa en {CANAL}, {SESION} vs {REF}:")
print(tabla.round(1).to_string(index=False))

cambio_a9 = tabla.loc[tabla["sujeto"] == SUJETO, "cambio_beta_pct"].iloc[0]
if SUJETO_CONTRASTE is None:
    otros = tabla[tabla["sujeto"] != SUJETO]
    contrarios = otros[np.sign(otros["cambio_beta_pct"]) != np.sign(cambio_a9)]
    if len(contrarios):
        SUJETO_CONTRASTE = contrarios.loc[contrarios["cambio_beta_pct"].abs().idxmax(), "sujeto"]
    else:
        SUJETO_CONTRASTE = otros.loc[otros["cambio_beta_pct"].abs().idxmin(), "sujeto"]
print(f"\nSujeto de contraste: {SUJETO_CONTRASTE}")


# ===========================================================================
# 2. Preparar los datos de cada sujeto
# ===========================================================================
def preparar(sujeto):
    j = cfg.CANALES.index(CANAL)
    x_ref = leer(sujeto, REF)[:, j]
    b_ref, ok_ref = beta_por_ventana(x_ref)
    x = leer(sujeto, SESION)
    b, ok = beta_por_ventana(x[:, j])
    fin = min(INICIO_S + DURACION_S, len(x) / FS)
    return {
        "sujeto": sujeto,
        "t": np.arange(len(x)) / FS,
        "x": x[:, j],
        "parpadeos": parpadeos(x),
        "beta": b, "aceptada": ok,
        "t_ventana": np.arange(len(b)) * cfg.DURACION_VENTANA + cfg.DURACION_VENTANA / 2,
        "beta_ref": b_ref[ok_ref].mean(),
        "fin": fin,
        "bloques": bloques[(bloques["sujeto"] == sujeto) & (bloques["sesion"] == SESION)],
    }


datos = [preparar(SUJETO), preparar(SUJETO_CONTRASTE)]
fin_comun = min(d["fin"] for d in datos)
tiempos = np.arange(INICIO_S + VENTANA_VISIBLE, fin_comun, VELOCIDAD / FPS)


def tarea_en(d, t):
    b = d["bloques"]
    actual = b[(b["inicio_sesion_s"] <= t) & (b["fin_sesion_s"] >= t)]
    return actual["tarea"].iloc[0] if len(actual) else "pausa"


# ===========================================================================
# 3. Armar la figura
# ===========================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1]})
elementos = []

for col, d in enumerate(datos):
    ax_s, ax_b = axs[0, col], axs[1, col]

    # Señal completa (se mueve cambiando los límites del eje)
    tramo = (d["t"] >= INICIO_S - 1) & (d["t"] <= d["fin"] + 1)
    ax_s.plot(d["t"][tramo], d["x"][tramo], color="black", lw=0.6)
    # Solo se dibujan las marcas del tramo que se reproduce (así el video se arma más rápido)
    desde, hasta = INICIO_S - 1, d["fin"] + 1
    for k in np.flatnonzero(~d["aceptada"]):
        if desde <= k * cfg.DURACION_VENTANA <= hasta:
            ax_s.axvspan(k * cfg.DURACION_VENTANA, (k + 1) * cfg.DURACION_VENTANA,
                         color="gray", alpha=0.25, lw=0)
    for p in d["parpadeos"]:
        if desde <= p <= hasta:
            ax_s.axvspan(p - 0.2, p + 0.4, color="red", alpha=0.3, lw=0)
    tope = max(100, np.percentile(np.abs(d["x"]), 99.5) * 1.1)
    ax_s.set_ylim(-tope, tope)
    ax_s.set_ylabel(f"{CANAL} [µV]")
    ax_s.set_xlabel("Tiempo [s]")
    titulo = ax_s.set_title("", loc="left")

    # Beta por ventana
    puntos = ax_b.scatter([], [], s=12, color="gray")
    ax_b.axhline(d["beta_ref"], color="black", ls=":", lw=1.5, label=f"media {REF}")
    linea_media, = ax_b.plot([], [], color="black", lw=2.5, label=f"media acumulada {SESION}")
    ahora = ax_b.axvline(INICIO_S, color="red", lw=1)
    visibles = (d["t_ventana"] >= INICIO_S) & (d["t_ventana"] <= fin_comun) & d["aceptada"]
    ax_b.set_xlim(INICIO_S, fin_comun)
    ax_b.set_ylim(0, np.percentile(d["beta"][visibles], 98) * 1.3 if visibles.any() else 1)
    ax_b.set_xlabel("Tiempo [s]")
    ax_b.set_ylabel(f"Beta relativa ({CANAL})")
    ax_b.legend(loc="upper left", fontsize=8)
    texto = ax_b.text(0.99, 0.95, "", transform=ax_b.transAxes, ha="right", va="top", fontsize=11)

    elementos.append({"ax_s": ax_s, "titulo": titulo, "puntos": puntos, "linea": linea_media,
                      "ahora": ahora, "texto": texto})

fig.suptitle(f"Sesión {SESION} reproducida a velocidad x{VELOCIDAD}. "
             "Rojo: parpadeos. Gris: ventanas descartadas (> "
             f"{cfg.UMBRAL_PP:g} µV pico a pico).", fontsize=11)
fig.tight_layout()


def actualizar(t_ahora):
    for d, e in zip(datos, elementos):
        e["ax_s"].set_xlim(t_ahora - VENTANA_VISIBLE, t_ahora)
        n_parp = int(((d["parpadeos"] >= INICIO_S) & (d["parpadeos"] <= t_ahora)).sum())
        minutos = (t_ahora - INICIO_S) / 60
        e["titulo"].set_text(f"{d['sujeto']} - {SESION} - tarea: {tarea_en(d, t_ahora)} - "
                             f"{n_parp} parpadeos ({n_parp / max(minutos, 1e-9):.0f}/min)")

        hasta = (d["t_ventana"] >= INICIO_S) & (d["t_ventana"] + 0.5 <= t_ahora) & d["aceptada"]
        e["puntos"].set_offsets(np.c_[d["t_ventana"][hasta], d["beta"][hasta]])
        if hasta.any():
            media = np.cumsum(d["beta"][hasta]) / np.arange(1, hasta.sum() + 1)
            e["linea"].set_data(d["t_ventana"][hasta], media)
            cambio = 100 * (media[-1] / d["beta_ref"] - 1)
            e["texto"].set_text(f"{cambio:+.0f} % respecto de {REF}")
        e["ahora"].set_xdata([t_ahora, t_ahora])
    return []


# ===========================================================================
# 4. Guardar
# ===========================================================================
anim = animation.FuncAnimation(fig, actualizar, frames=tiempos, interval=1000 / FPS)
nombre = f"video_{SUJETO}_vs_{SUJETO_CONTRASTE}"
if animation.writers.is_available("ffmpeg"):
    ruta = DIR_SALIDA / f"{nombre}.mp4"
    anim.save(ruta, writer=animation.FFMpegWriter(fps=FPS, bitrate=2500), dpi=110)
else:
    ruta = DIR_SALIDA / f"{nombre}.gif"
    print("No se encontró ffmpeg: se guarda como gif (pesa más).")
    anim.save(ruta, writer=animation.PillowWriter(fps=FPS), dpi=80)

actualizar(tiempos[-1])
fig.savefig(DIR_SALIDA / "ultimo_cuadro.png", dpi=150)
plt.close(fig)
print(f"Video guardado en {ruta} ({len(tiempos) / FPS:.0f} s)")