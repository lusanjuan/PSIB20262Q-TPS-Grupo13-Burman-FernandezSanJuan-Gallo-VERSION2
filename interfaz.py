"""
Interfaz paso a paso: sigue un canal de un sujeto por todo el procesamiento.

Uso (desde la carpeta del TP, después de correr los pasos 1 a 8):
    streamlit run paso_a_paso.py

No recalcula el pipeline: lee los archivos de resultados/ (pasos 1 a 8).
Solo calcula dos cosas para mostrarlas: el espectro antes y después del filtrado
y el rango de la señal en cada segundo (para el gráfico del registro completo).


python -m pip install streamlit
python -m streamlit run .\interfaz.py

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator, FuncFormatter
import streamlit as st
from scipy import signal

import config as cfg

RES = cfg.DIR_RESULTADOS
SES = cfg.SESIONES
MUSICAS = SES[1:]
NOM = {"sin_musica": "Sin música", "musica_relajante": "Relajante",
       "musica_excitante": "Excitante", "musica_ia": "IA"}
CORTO = ["Sin", "Relaj.", "Excit.", "IA"]
MARCADORES = [("beta", "TP9"), ("beta", "TP10"), ("theta", "AF7"), ("theta", "AF8")]
MIN_VENTANAS = 20

# Colores (los de la presentación)
NAVY, PINK, BLUE, VIOLET = "#0b3a5c", "#e0457b", "#2f6fdb", "#7b61ff"
GRIS, GRIS_CLARO, ROJO, ROJO_CLARO = "#7d8b99", "#d7e2ec", "#d62839", "#f7c5c9"
C_SES = ["#7d8b99", "#1f9a8a", "#ea7a1c", "#7b61ff"]
C_TAREA = {"1-back": BLUE, "3-back": PINK}
C_BANDA = {"theta": "#ece7ff", "beta": "#e2eeff"}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9fb0c0", "axes.labelcolor": "#4b6479", "xtick.color": "#4b6479",
    "ytick.color": "#4b6479", "axes.grid": True, "grid.color": "#eef3f8", "axes.axisbelow": True,
    "figure.facecolor": "white", "legend.frameon": False,
})

PASOS = ["1. Señal cruda", "2. Filtrado", "3. Separación", "4. Ventaneo y rechazo",
         "5. Ventanas por tarea", "6. PSD por sesión", "7. PSD por tarea", "8. Del sujeto al grupo"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo el EEG crudo…")
def cargar_crudo(sujeto):
    df = pd.read_csv(cfg.archivo_eeg(sujeto), usecols=[cfg.COL_TIEMPO] + cfg.CANALES).dropna()
    return df[cfg.COL_TIEMPO].to_numpy(), df[cfg.CANALES].to_numpy(float)


@st.cache_data
def cargar_filtrado(sujeto):
    d = np.load(cfg.DIR_FILTRADO / f"{sujeto}_filtrado.npz")
    return d["t"], d["x"]


@st.cache_data
def cargar_sesion(sujeto, sesion):
    i = SES.index(sesion)
    return pd.read_csv(cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{sesion}.csv")[cfg.CANALES]


@st.cache_data
def leer(ruta):
    return pd.read_csv(ruta)


@st.cache_data
def rango_por_segundo(sujeto):
    """Percentiles 10 y 90 de cada segundo del registro filtrado (para dibujarlo entero)."""
    t, x = cargar_filtrado(sujeto)
    df = pd.DataFrame(x, columns=cfg.CANALES)
    df["s"] = (t - t[0]).astype(int)
    g = df.groupby("s")
    return g.quantile(0.10), g.quantile(0.90)


@st.cache_data
def espectros(sujeto, canal):
    """Welch de 5 min desde el minuto 10, antes y después del filtrado."""
    t, x = cargar_crudo(sujeto)
    tf, xf = cargar_filtrado(sujeto)
    j = cfg.CANALES.index(canal)
    i0, k0 = np.searchsorted(t, t[0] + 600), np.searchsorted(tf, t[0] + 600)
    n = 300 * cfg.FS_ORIGINAL
    f0, p0 = signal.welch(x[i0:i0 + n, j], fs=cfg.FS_ORIGINAL, nperseg=4 * cfg.FS_ORIGINAL)
    f1, p1 = signal.welch(xf[k0:k0 + n // 2, j], fs=cfg.FS_FINAL, nperseg=4 * cfg.FS_FINAL)
    return f0, p0, f1, p1


def ventanas(sujeto, sesion, canal):
    v = leer(RES / "03_artefactos" / "ventanas.csv")
    v = v[(v.sujeto == sujeto) & (v.sesion == sesion) & (v.canal == canal)].sort_values("ventana")
    return v.pico_a_pico_uV.to_numpy(), v.aceptada.to_numpy(bool)


def bloques(sujeto, sesion):
    b = leer(RES / "05_nback" / "bloques.csv")
    return b[(b.sujeto == sujeto) & (b.sesion == sesion)]


def tarea_de_cada_ventana(sujeto, sesion, n):
    """Igual que el paso 5: la ventana es del bloque si cae completa dentro de él."""
    tarea = np.full(n, None, dtype=object)
    for b in bloques(sujeto, sesion).itertuples():
        k = np.arange(n)
        tarea[(k >= b.inicio_sesion_s) & (k + 1 <= b.fin_sesion_s)] = b.tarea
    return tarea


def tramo_con_rechazos(ok):
    """Elige un tramo de 60 s con unas 4 ventanas descartadas, para que se vea el criterio."""
    mejor, inicio = 1e9, 0
    for a in range(30, max(31, len(ok) - 60), 10):
        r = (~ok[a:a + 60]).sum()
        puntaje = abs(r - 4) + (0 if r > 0 else 10)
        if puntaje < mejor:
            mejor, inicio = puntaje, a
    return inicio


def bandas(ax, arriba=True):
    for b in ["theta", "beta"]:
        lo, hi = cfg.BANDAS[b]
        ax.axvspan(lo, hi, color=C_BANDA[b], zorder=0)
        if arriba:
            ax.text((lo + hi) / 2, 1.01, b, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color="#4b6479")


def eje_log(ax, subs=(1, 2, 5)):
    """Marcas en el eje logarítmico (1-2-5 por defecto), escritas como números comunes."""
    ax.yaxis.set_major_locator(LogLocator(subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fmt(v, d=1):
    return "–" if v is None or pd.isna(v) else f"{v:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Pasos: cada uno devuelve (figura, texto en markdown)
# ---------------------------------------------------------------------------
def paso_crudo(sujeto, canal, segundos):
    t, x = cargar_crudo(sujeto)
    i0 = np.searchsorted(t, t[0] + 600)
    y = x[i0:i0 + segundos * cfg.FS_ORIGINAL, cfg.CANALES.index(canal)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(np.arange(len(y)) / cfg.FS_ORIGINAL, y, color=GRIS, lw=0.8)
    ax.set_xlim(0, segundos)
    ax.set_xlabel("Tiempo [s] (minuto 10 del registro)")
    ax.set_ylabel(f"{canal} [µV]")
    texto = f"""### Lo que mide el electrodo
Así llega **{canal}** desde la vincha Muse: {cfg.FS_ORIGINAL} muestras por segundo, referidas a Fpz.
La señal está corrida hacia arriba (media de {fmt(y.mean(), 0)} µV) y tiene componentes rápidas
que no son actividad cerebral: ruido de la red eléctrica, actividad muscular y deriva del electrodo.

El primer paso es filtrarla."""
    return fig, texto


def paso_filtrado(sujeto, canal, segundos):
    t, x = cargar_crudo(sujeto)
    tf, xf = cargar_filtrado(sujeto)
    j = cfg.CANALES.index(canal)
    i0, k0 = np.searchsorted(t, t[0] + 600), np.searchsorted(tf, t[0] + 600)
    crudo = x[i0:i0 + segundos * cfg.FS_ORIGINAL, j]
    filt = xf[k0:k0 + segundos * cfg.FS_FINAL, j]
    f0, p0, f1, p1 = espectros(sujeto, canal)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"hspace": 0.4})
    a1.plot(np.arange(len(crudo)) / cfg.FS_ORIGINAL, crudo - crudo.mean(), color=GRIS, lw=0.7,
            alpha=0.7, label="cruda (sin la media)")
    a1.plot(np.arange(len(filt)) / cfg.FS_FINAL, filt, color=NAVY, lw=1.2,
            label=f"filtrada, {cfg.FS_FINAL} Hz")
    a1.set_xlim(0, segundos)
    a1.set_xlabel("Tiempo [s]")
    a1.set_ylabel("µV")
    a1.legend(loc="upper right", ncol=2)

    a2.axvspan(cfg.F_BAJA, cfg.F_ALTA, color="#eef3f8", zorder=0)
    a2.semilogy(f0, p0, color=GRIS, lw=1.2, label="cruda")
    a2.semilogy(f1, p1, color=NAVY, lw=1.8, label="filtrada")
    a2.axvline(cfg.F_ALTA, color="#4b6479", ls=":", lw=1)
    a2.text(cfg.F_ALTA + 0.5, 0.97, f"{cfg.F_ALTA:g} Hz", transform=a2.get_xaxis_transform(),
            va="top", fontsize=9, color="#4b6479")
    a2.set_xlim(0, 70)
    a2.set_xlabel("Frecuencia [Hz]")
    a2.set_ylabel("PSD [µV²/Hz]")
    a2.legend(loc="upper right")
    eje_log(a2, subs=(1,))
    texto = f"""### Pasabanda y submuestreo
Dos operaciones:

1. **Butterworth pasabanda {cfg.F_BAJA:g}–{cfg.F_ALTA:g} Hz** (orden {cfg.ORDEN_FILTRO}), aplicado ida y
   vuelta (fase cero) para no correr la señal en el tiempo. Saca el offset y las derivas lentas
   (por debajo de 1 Hz) y todo lo rápido (por encima de 50 Hz, incluido el ruido de red).
2. **Submuestreo de {cfg.FS_ORIGINAL} a {cfg.FS_FINAL} Hz**: alcanza para analizar hasta 45 Hz.

Abajo, el espectro de 5 minutos: entre 1 y 45 Hz, donde están theta y beta, la señal no cambia;
por encima de 50 Hz cae."""
    return fig, texto


def paso_separacion(sujeto, canal):
    lo, hi = rango_por_segundo(sujeto)
    tf, _ = cargar_filtrado(sujeto)
    tags = np.loadtxt(cfg.archivo_tags(sujeto), ndmin=1) - tf[0]
    minutos = lo.index.to_numpy() / 60
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, s in enumerate(SES):
        ax.axvspan(tags[2 * i] / 60, tags[2 * i + 1] / 60, color=C_SES[i], alpha=0.22,
                   label=f"{i + 1}. {NOM[s]}")
    ax.fill_between(minutos, lo[canal], hi[canal], color=NAVY, lw=0, alpha=0.85)
    ax.set_ylim(-60, 60)
    ax.set_xlim(0, minutos[-1])
    ax.set_xlabel("Tiempo desde el inicio del registro [min]")
    ax.set_ylabel(f"{canal} filtrado [µV]\n(rango 10–90 % de cada segundo)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4)
    duraciones = [(tags[2 * i + 1] - tags[2 * i]) / 60 for i in range(len(SES))]
    texto = f"""### Cortar las cuatro sesiones
El registro de {sujeto} dura {fmt(minutos[-1], 0)} minutos. Los 8 tags (inicio y fin de cada sesión)
marcan dónde sonó cada música; lo que queda entre sesiones (pausas, instrucciones) se descarta.

""" + "\n".join(f"- **{NOM[s]}**: {fmt(d)} min" for s, d in zip(SES, duraciones)) + """

El orden de las sesiones fue siempre el mismo para todos los sujetos."""
    return fig, texto


def paso_ventaneo(sujeto, canal, sesion):
    pp, ok = ventanas(sujeto, sesion, canal)
    x = cargar_sesion(sujeto, sesion)[canal].to_numpy()
    ini = tramo_con_rechazos(ok)
    y = x[ini * cfg.FS_FINAL:(ini + 60) * cfg.FS_FINAL]
    U = cfg.UMBRAL_PP

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.45})
    for k in range(60):
        if not ok[ini + k]:
            a1.axvspan(k, k + 1, color=ROJO_CLARO, lw=0, zorder=0)
    for k in range(61):
        a1.axvline(k, color="#eef3f8", lw=0.6, zorder=0)
    a1.plot(np.arange(len(y)) / cfg.FS_FINAL, y, color=NAVY, lw=0.7)
    a1.set_xlim(0, 60)
    a1.set_ylim(-200, 200)
    a1.grid(False)
    a1.set_xlabel(f"Tiempo [s] (desde el segundo {ini} de la sesión)")
    a1.set_ylabel(f"{canal} [µV]")
    a1.set_title(f"En rojo: ventanas descartadas (pico a pico > {U:g} µV)", loc="left", fontsize=10,
                 color="#4b6479")

    minutos = (np.arange(len(pp)) + 0.5) / 60
    a2.vlines(minutos, 0, np.minimum(pp, 400), colors=np.where(ok, GRIS, ROJO), lw=0.8)
    a2.axhline(U, color=ROJO, ls="--", lw=1)
    a2.text(1.0, U, f" {U:g} µV", transform=a2.get_yaxis_transform(), va="center", color=ROJO, fontsize=9)
    a2.set_xlim(0, len(pp) / 60)
    a2.set_ylim(0, 400)
    a2.set_xlabel("Minuto de la sesión")
    a2.set_ylabel("pico a pico [µV]")
    rechazadas = int((~ok).sum())
    texto = f"""### Ventanas de 1 s y umbral
Cada sesión se corta en ventanas de {cfg.DURACION_VENTANA:g} s sin solapamiento. A cada una se le mide
la amplitud pico a pico:

$$pp = \\max(x) - \\min(x)$$

Si supera **{U:g} µV** (el límite superior de amplitud del EEG) hay un parpadeo o un movimiento,
y la ventana entera se descarta.

**{rechazadas} de {len(ok)} ventanas descartadas** ({fmt(100 * rechazadas / len(ok))} %) en
{NOM[sesion].lower()}.

La señal no se modifica: solo se elige qué ventanas entran a la PSD. Arriba, 60 s de la sesión;
abajo, el pico a pico de todas sus ventanas."""
    return fig, texto


def paso_tareas(sujeto, canal, sesion):
    pp, ok = ventanas(sujeto, sesion, canal)
    n = len(ok)
    tarea = tarea_de_cada_ventana(sujeto, sesion, n)
    b = bloques(sujeto, sesion)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [1, 2], "hspace": 0.5})
    for r in b.itertuples():
        a1.broken_barh([(r.inicio_sesion_s, r.fin_sesion_s - r.inicio_sesion_s)], (1.15, 0.85),
                       color=C_TAREA[r.tarea])
    colores = [ROJO if not o else C_TAREA.get(t, GRIS_CLARO) for t, o in zip(tarea, ok)]
    a1.bar(np.arange(n) + 0.5, 0.85, width=1.0, color=colores, linewidth=0)
    a1.set_yticks([0.42, 1.57], ["ventanas", "bloques"])
    a1.set_ylim(0, 2)
    a1.set_xlim(0, n)
    a1.grid(False)
    a1.set_xlabel("Segundo de la sesión")
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLUE, PINK, GRIS_CLARO, ROJO]],
              labels=["1-back", "3-back", "fuera de tarea", "descartada"],
              loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4)

    for r in b.itertuples():
        a2.scatter(r.inicio_sesion_s, r.aciertos_pct, color=C_TAREA[r.tarea], s=45, zorder=3)
    a2.axhline(100, color="#4b6479", ls=":", lw=1)
    a2.set_xlim(0, n)
    a2.set_ylim(40, 105)
    a2.set_xlabel("Segundo de la sesión (inicio del bloque)")
    a2.set_ylabel("Aciertos del bloque [%]")

    def cuenta(t):
        es = tarea == t
        return int(es.sum()), int((es & ok).sum())
    n1, a1n = cuenta("1-back")
    n3, a3n = cuenta("3-back")
    fuera = int(sum(t is None for t in tarea))
    texto = f"""### Cada ventana, a su tarea
Con los tiempos de E-Prime se arman los bloques de 1-back y 3-back. Una ventana pertenece a un
bloque si cae **completa** dentro de él; las de instrucciones y pausas quedan fuera de tarea.

- **1-back**: {a1n} ventanas aceptadas de {n1}
- **3-back**: {a3n} ventanas aceptadas de {n3}
- **Fuera de tarea**: {fuera}

Arriba: los bloques y las ventanas de {canal}. Abajo: los aciertos de cada bloque; el 1-back está
casi siempre en el techo."""
    return fig, texto


def paso_psd_sesion(sujeto, canal):
    psd = leer(RES / "04_psd" / "psd_promedio.csv")
    psd = psd[(psd.sujeto == sujeto) & (psd.canal == canal) & psd.frecuencia_Hz.between(*cfg.BANDA_TOTAL)]
    pot = leer(RES / "04_psd" / "potencia_bandas.csv")
    pot = pot[(pot.sujeto == sujeto) & (pot.canal == canal)]

    fig = plt.figure(figsize=(10, 7.5))
    g = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(g[0, :])
    bandas(ax)
    for i, s in enumerate(SES):
        d = psd[psd.sesion == s].sort_values("frecuencia_Hz")
        ax.semilogy(d.frecuencia_Hz, d.psd_uV2_Hz, color=C_SES[i], lw=2, label=NOM[s])
    ax.set_xlim(*cfg.BANDA_TOTAL)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [µV²/Hz]")
    ax.legend(loc="upper right", ncol=2)
    eje_log(ax)
    for k, (b, color) in enumerate([("theta", VIOLET), ("beta", BLUE)]):
        a = fig.add_subplot(g[1, k])
        v = [pot[(pot.banda == b) & (pot.sesion == s)].potencia_rel.mean() for s in SES]
        a.plot(range(4), v, color=color, lw=2, marker="o", ms=7)
        a.set_xticks(range(4), CORTO)
        a.set_ylim(0, np.nanmax(v) * 1.25)
        a.set_ylabel(f"{b} relativa")
    texto = f"""### Del tiempo a la frecuencia
A cada ventana aceptada se le calcula la PSD con Welch (ventana Hann de 1 s, resolución de 1 Hz)
y se promedian todas las de la sesión.

$$P_{{rel}} = \\frac{{P_{{banda}}}}{{P_{{1-45\\,Hz}}}}$$

Abajo, la potencia relativa de theta y beta en cada sesión. La relativa compensa las diferencias
de contacto del electrodo entre sesiones."""
    return fig, texto


def paso_psd_tarea(sujeto, canal, sesion):
    psd = leer(RES / "05_nback" / "psd_nback.csv")
    psd = psd[(psd.sujeto == sujeto) & (psd.canal == canal) & psd.frecuencia_Hz.between(*cfg.BANDA_TOTAL)]
    pot = leer(RES / "05_nback" / "potencia_nback.csv")
    pot = pot[(pot.sujeto == sujeto) & (pot.canal == canal)].copy()
    pot.loc[pot.n_ventanas_aceptadas < MIN_VENTANAS, "potencia_rel"] = np.nan

    fig = plt.figure(figsize=(10, 7.5))
    g = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(g[0, :])
    bandas(ax)
    for tarea, color in C_TAREA.items():
        d = psd[(psd.sesion == sesion) & (psd.tarea == tarea)].sort_values("frecuencia_Hz")
        ax.semilogy(d.frecuencia_Hz, d.psd_uV2_Hz, color=color, lw=2, label=tarea)
    ax.set_xlim(*cfg.BANDA_TOTAL)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [µV²/Hz]")
    ax.legend(loc="upper right", title=NOM[sesion])
    eje_log(ax)
    for k, b in enumerate(["theta", "beta"]):
        a = fig.add_subplot(g[1, k])
        maximo = 0
        for tarea, color in C_TAREA.items():
            v = [pot[(pot.banda == b) & (pot.sesion == s) & (pot.tarea == tarea)].potencia_rel.mean() for s in SES]
            a.plot(range(4), v, color=color, lw=2, marker="o", ms=6)
            maximo = max(maximo, np.nanmax(v))
        a.set_xticks(range(4), CORTO)
        a.set_ylim(0, maximo * 1.25)
        a.set_ylabel(f"{b} relativa")
    texto = """### 1-back y 3-back por separado
Con las ventanas etiquetadas en el paso 5 se calcula una PSD para cada tarea dentro de cada sesión.
Así la música se compara con la tarea fija.

Abajo, theta y beta relativas de las dos tareas en las cuatro sesiones. La pregunta del TP:
¿cambian con la música en 3-back?"""
    return fig, texto


def paso_grupo(sujeto):
    nb = leer(RES / "05_nback" / "potencia_nback.csv").copy()
    nb.loc[nb.n_ventanas_aceptadas < MIN_VENTANAS, "potencia_rel"] = np.nan
    t3 = nb[nb.tarea == "3-back"].pivot_table(index=["sujeto", "canal", "banda"], columns="sesion",
                                              values="potencia_rel", dropna=False)
    cambio = 100 * (t3[MUSICAS].div(t3["sin_musica"], axis=0) - 1)
    p3b = leer(RES / "08_estadistica" / "3b_musica_eeg_3back.csv")

    def estrellas(p):
        return "" if pd.isna(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    fig, axs = plt.subplots(1, 4, figsize=(12, 4.4), gridspec_kw={"wspace": 0.45})
    friedman = {}
    rng = np.random.default_rng(0)
    for ax, (banda, canal) in zip(axs, MARCADORES):
        m = cambio.xs((canal, banda), level=["canal", "banda"])
        for suj, fila in m.iterrows():
            yo = suj == sujeto
            dx = 0 if yo else rng.uniform(-0.15, 0.15, 3)
            ax.scatter(np.arange(3) + dx, fila, s=60 if yo else 18, color=PINK if yo else GRIS,
                       edgecolor="white", lw=0.8, zorder=4 if yo else 2)
        color = VIOLET if banda == "theta" else BLUE
        ax.plot(range(3), m.mean(), color=color, lw=2.5, marker="D", ms=7, zorder=3)
        ax.axhline(0, color="#4b6479", lw=0.8)
        lim = min(250, np.ceil(np.nanmax(np.abs(m.to_numpy())) * 1.1 / 50) * 50)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(3), CORTO[1:])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"{banda} {canal}", color=color, fontweight="bold", pad=20)
        fila = p3b[(p3b.banda == banda) & (p3b.canal == canal)]
        for k, s in enumerate(MUSICAS):
            p = fila[fila.sesion == s].p_holm
            ax.text(k, 1.0, estrellas(p.iloc[0]) if len(p) else "", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#4b6479")
        friedman[f"{banda} {canal}"] = fila.p_friedman_4_sesiones.iloc[0] if len(fila) else np.nan
    axs[0].set_ylabel("Cambio de potencia en 3-back\nvs sin música [%]")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PINK, ms=8),
                        plt.Line2D([], [], marker="o", ls="", color=GRIS, ms=5),
                        plt.Line2D([], [], marker="D", color=NAVY, lw=2)],
               labels=[sujeto, "otros sujetos", "media del grupo"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    texto = f"""### Del sujeto al grupo
Lo mismo que hicimos con {sujeto} se repite en los 9 sujetos. En cada marcador se calcula el cambio
de potencia en 3-back de cada música respecto de sin música ({sujeto} en rosa).

Ningún cambio es significativo (Wilcoxon con Holm, arriba de cada música), y tampoco el test de
las 4 sesiones juntas:

""" + "\n".join(f"- **{k}**: Friedman p = {fmt(v, 2)}" for k, v in friedman.items())
    return fig, texto


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EEG paso a paso", layout="wide")
st.title("Una señal, paso a paso")

if "paso" not in st.session_state:
    st.session_state.paso = PASOS[0]

c1, c2 = st.columns([1, 2])
sujeto = c1.selectbox("Sujeto", cfg.SUJETOS_ACTIVOS, index=cfg.SUJETOS_ACTIVOS.index("A3")
                      if "A3" in cfg.SUJETOS_ACTIVOS else 0)
canal = c2.radio("Canal", cfg.CANALES, horizontal=True)
st.radio("Paso", PASOS, key="paso", horizontal=True, label_visibility="collapsed")
i = PASOS.index(st.session_state.paso)
st.caption(f"Seguimos el canal {canal} del sujeto {sujeto} desde el electrodo hasta la estadística del grupo.")

izq, der = st.columns([2.2, 1], gap="large")
with izq:
    if i in (3, 4, 6):
        sesion = st.radio("Sesión", SES, format_func=NOM.get, horizontal=True)
    if i in (0, 1):
        segundos = st.radio("Ventana", [2, 10], format_func=lambda s: f"{s} s", horizontal=True)

    if i == 0:
        fig, texto = paso_crudo(sujeto, canal, segundos)
    elif i == 1:
        fig, texto = paso_filtrado(sujeto, canal, segundos)
    elif i == 2:
        fig, texto = paso_separacion(sujeto, canal)
    elif i == 3:
        fig, texto = paso_ventaneo(sujeto, canal, sesion)
    elif i == 4:
        fig, texto = paso_tareas(sujeto, canal, sesion)
    elif i == 5:
        fig, texto = paso_psd_sesion(sujeto, canal)
    elif i == 6:
        fig, texto = paso_psd_tarea(sujeto, canal, sesion)
    else:
        fig, texto = paso_grupo(sujeto)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with der:
    st.markdown(texto)


def mover(delta):
    st.session_state.paso = PASOS[max(0, min(len(PASOS) - 1, i + delta))]


b1, _, b2 = st.columns([1, 4, 1])
b1.button("← Anterior", on_click=mover, args=(-1,), disabled=i == 0, use_container_width=True)
b2.button("Siguiente →", on_click=mover, args=(1,), disabled=i == len(PASOS) - 1, use_container_width=True)