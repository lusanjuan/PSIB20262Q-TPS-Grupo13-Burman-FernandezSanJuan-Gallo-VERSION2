"""
Paso 9: gráficos de los resultados, en el orden del relato del informe.

Usa las tablas de los pasos 4, 5 y 8 (incluida la Parte 3b: 3b_musica_eeg_3back.csv). Solo beta (hipótesis del paper) y theta
(marcador de memoria de trabajo); alfa queda fuera del relato principal.

RELATO
  fig1_musica_aciertos.png  La música relajante mejora el desempeño en 3-back (significativo).
                            La excitante lo baja (n.s.) y la IA lo sube (tendencia). La caída
                            con la excitante no es compatible con una mejora por pura práctica.
  fig2_carga_aciertos.png   El 3-back es más difícil que el 1-back (9 de 9 sujetos), y la música
                            solo cambia el 3-back (el 1-back está cerca del techo).
  fig3_eeg_vs_conducta.png  Hipótesis del paper: si el desempeño cambia, beta y theta deberían
                            cambiar. El desempeño mejora, pero beta y theta no acompañan.
  fig3b_musica_eeg_3back.png  Control: aun sacando el 1-back (cerca del techo de aciertos) y los
                            tramos sin tarea, beta y theta en 3-back no cambian entre sesiones.
  fig4_paper_vs_grupo.png   El paper lo muestra en un sujeto (A9); en el grupo no se ve.

ANEXO (para la discusión)
  fig5_dos_analisis.png     Por qué el paper llegó a otra conclusión: con muestras de un solo
                            sujeto todo da "significativo", incluso en direcciones opuestas.
  fig6_correlacion.png      Los sujetos que más mejoran no son los que más cambian beta o theta.

Estilo de todas las figuras:
  - título (la conclusión, en negrita) + subtítulo gris (qué se compara)
  - nota al pie con las convenciones: gris = cada sujeto; color = media ± error estándar;
    p corregido por Holm (el mismo del paso 8); * p<0.05, ** p<0.01, *** p<0.001, n.s.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

DIR_PSD = cfg.DIR_RESULTADOS / "04_psd"
DIR_NBACK = cfg.DIR_RESULTADOS / "05_nback"
DIR_EST = cfg.DIR_RESULTADOS / "08_estadistica"
DIR_FIG = cfg.DIR_RESULTADOS / "09_graficos"
DIR_FIG.mkdir(parents=True, exist_ok=True)

MIN_VENTANAS = 20
REF = cfg.SESION_REFERENCIA
musicas = [s for s in cfg.SESIONES if s != REF]
ETIQUETAS = {"sin_musica": "Sin música", "musica_relajante": "Relajante",
             "musica_excitante": "Excitante", "musica_ia": "IA"}
NOMBRE = {"musica_relajante": "la música relajante", "musica_excitante": "la música excitante",
          "musica_ia": "la música IA"}
PAPER_A9 = {"musica_relajante": 29.9, "musica_excitante": 27.3, "musica_ia": 46.8}
# Marcadores EEG del relato: beta en los canales del paper, theta en los frontales
MARCADORES = [("beta", "TP9"), ("beta", "TP10"), ("theta", "AF7"), ("theta", "AF8")]

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
# Paleta validada para daltonismo
AZUL, NARANJA, VIOLETA = "#2a78d6", "#eb6834", "#6250d6"
GRIS, GRIS_CLARO, NEGRO, TINTA_2 = "#b5b4ae", "#dcdbd5", "#0b0b0b", "#52514e"
COLOR_BANDA = {"beta": AZUL, "theta": VIOLETA}
NOTA = ("Gris: cada sujeto · Color: media del grupo ± error estándar · "
        "p corregido por Holm · * p<0.05  ** p<0.01  *** p<0.001  n.s. no significativo")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": NEGRO, "axes.linewidth": 0.8,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.facecolor": "white",
})


def estrellas(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def a_pct(log_cambio):
    return 100 * (10 ** log_cambio - 1)


def encabezado(fig, titulo, subtitulo="", reserva=1.0):
    """
    Título en negrita (la conclusión) y subtítulo gris (qué se compara).
    'reserva' = pulgadas libres arriba de los gráficos para el encabezado
    (más grande si los paneles tienen títulos propios).
    """
    alto = fig.get_figheight()
    fig.subplots_adjust(top=1 - reserva / alto)
    fig.text(0.01, 1 - 0.15 / alto, titulo, ha="left", va="top", fontsize=14, fontweight="bold")
    if subtitulo:
        fig.text(0.01, 1 - 0.5 / alto, subtitulo, ha="left", va="top", fontsize=10.5, color=TINTA_2)


def nota(fig, texto=NOTA):
    fig.text(0.01, -0.06, texto, ha="left", va="top", fontsize=8.5, color=TINTA_2)


def marca_significancia(ax, x, p, flecha=""):
    """Asteriscos arriba del panel: en negro si es significativo, gris si no."""
    texto = estrellas(p) + flecha
    color = NEGRO if (not pd.isna(p) and p < 0.05) else GRIS
    ax.text(x, 1.0, texto, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=12 if color == NEGRO else 10, fontweight="bold", color=color)


def puntos_sujetos(ax, x, valores, semilla=0, ancho=0.09):
    """Un punto gris por sujeto, apenas separados en x para que no se tapen."""
    valores = pd.Series(valores).dropna()
    jitter = np.random.default_rng(semilla).uniform(-ancho, ancho, len(valores))
    ax.scatter(x + jitter, valores, color=GRIS, s=26, zorder=2, edgecolor="white", linewidths=0.6)


def media(ax, x, valores, color, marcador="o", tam=11):
    """Media ± error estándar, con borde blanco para que se separe de los puntos."""
    valores = pd.Series(valores).dropna()
    ax.errorbar(x, valores.mean(), yerr=valores.sem(), fmt=marcador, color=color, ms=tam,
                mec="white", mew=1.5, capsize=5, elinewidth=2, capthick=2, zorder=4)


def guardar(fig, nombre):
    fig.savefig(DIR_FIG / nombre, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {nombre}")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
bloques = pd.read_csv(DIR_NBACK / "bloques.csv")
P1a = pd.read_csv(DIR_EST / "1a_conducta_carga.csv")
P1b = pd.read_csv(DIR_EST / "1b_conducta_musica.csv")
P2 = pd.read_csv(DIR_EST / "2_replica_resumen.csv")
P3 = pd.read_csv(DIR_EST / "3_musica_eeg.csv")
P3b = pd.read_csv(DIR_EST / "3b_musica_eeg_3back.csv")
P5 = pd.read_csv(DIR_EST / "5_eeg_y_aciertos.csv")

potencia = pd.read_csv(DIR_PSD / "potencia_bandas.csv")
potencia = potencia[potencia["n_ventanas_aceptadas"] >= MIN_VENTANAS]
nback = pd.read_csv(DIR_NBACK / "potencia_nback.csv")
nback.loc[nback["n_ventanas_aceptadas"] < MIN_VENTANAS, "potencia_rel"] = np.nan

# Aciertos por sujeto, sesión y tarea
acc = bloques.groupby(["sujeto", "tarea", "sesion"])["aciertos_pct"].mean()
acc3 = acc.xs("3-back", level="tarea").unstack().reindex(columns=cfg.SESIONES)
acc1 = acc.xs("1-back", level="tarea").unstack().reindex(columns=cfg.SESIONES)
d_acc3 = acc3[musicas].sub(acc3[REF], axis=0)          # puntos porcentuales vs sin música


def cambio_eeg_3back(banda, canal):
    """% de cambio de la potencia en los bloques de 3-back, cada música vs sin música."""
    d = nback[(nback["tarea"] == "3-back") & (nback["banda"] == banda) & (nback["canal"] == canal)]
    m = d.pivot_table(index="sujeto", columns="sesion", values="potencia_rel").reindex(columns=cfg.SESIONES)
    return a_pct(np.log10(m[musicas].div(m[REF], axis=0)))


def fila_1b(tarea, sesion):
    return P1b[(P1b["tarea"] == tarea) & (P1b["sesion"] == sesion)].iloc[0]


marcadores = [(b, c) for b, c in MARCADORES if ((P5["banda"] == b) & (P5["canal"] == c)).any()]
x = np.arange(len(cfg.SESIONES))
xm = np.arange(len(musicas))
print("Figuras:")

# ===========================================================================
# Figura 1 – La música y el desempeño en 3-back
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 6))
for _, fila in acc3.iterrows():
    ax.plot(x, fila, color=GRIS_CLARO, lw=1.2, marker="o", ms=4, mfc=GRIS, mec=GRIS, zorder=2)
ax.plot(x, acc3.mean(), color=NARANJA, lw=3, zorder=3)
for k in x:
    media(ax, k, acc3.iloc[:, k], NARANJA)

# Corchetes: cada música vs sin música (etiqueta corta; el p va en la nota)
techo = np.nanmax(acc3.to_numpy())
paso = 0.09 * (techo - np.nanmin(acc3.to_numpy()))
for k, s in enumerate(musicas, start=1):
    f = fila_1b("3-back", s)
    y = techo + paso * (1 + 1.3 * (k - 1))
    color = NEGRO if f["p_holm"] < 0.05 else GRIS
    ax.plot([0, 0, k, k], [y - paso * 0.3, y, y, y - paso * 0.3], color=color, lw=1)
    ax.text(k, y, f"  {estrellas(f['p_holm'])}  {f['diferencia_pp']:+.1f} pts "
            f"({int(f['mejoran'])}/{int(f['n_sujetos'])} mejoran)",
            ha="left", va="center", fontsize=10, color=color,
            fontweight="bold" if f["p_holm"] < 0.05 else "normal")

# Nota sobre la práctica, junto a la sesión excitante
k_exc = cfg.SESIONES.index("musica_excitante")
ax.annotate("Si la mejora fuera solo práctica,\nacá debería seguir subiendo",
            xy=(k_exc, acc3.mean().iloc[k_exc]),
            xytext=(k_exc, np.nanmin(acc3.to_numpy()) - 2.2 * paso),
            fontsize=9.5, color=TINTA_2, ha="center", va="top",
            bbox={"boxstyle": "round,pad=0.4", "fc": "#f7f6f2", "ec": GRIS_CLARO},
            arrowprops={"arrowstyle": "-|>", "color": TINTA_2, "lw": 1})

ax.set_xticks(x)
ax.set_xticklabels([f"{ETIQUETAS[s]}\n{i + 1}.ª sesión" for i, s in enumerate(cfg.SESIONES)])
ax.set_xlim(-0.4, len(x) - 1 + 1.6)                       # lugar a la derecha para las etiquetas
ax.set_ylim(np.nanmin(acc3.to_numpy()) - 6 * paso, techo + paso * (1.3 * len(musicas) + 0.5))
ax.set_ylabel("Aciertos en 3-back [%]")

sig = P1b[(P1b["tarea"] == "3-back") & (P1b["p_holm"] < 0.05) & (P1b["diferencia_pp"] > 0)]
nombres = [NOMBRE[s] for s in sig["sesion"]]
if nombres:
    titulo = " y ".join(nombres) + (" mejora" if len(nombres) == 1 else " mejoran") + " el desempeño en 3-back"
    titulo = titulo[0].upper() + titulo[1:]
else:
    titulo = "Ninguna música cambia significativamente el desempeño en 3-back"
encabezado(fig, titulo, "Aciertos en la tarea difícil en cada sesión; cada música comparada con sin música",
           reserva=1.1)
nota(fig)
guardar(fig, "fig1_musica_aciertos.png")

# ===========================================================================
# Figura 2 – El 3-back es más difícil (y la música solo afecta al 3-back)
# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.35})

ax = axs[0]
por_tarea = bloques.groupby(["sujeto", "tarea"])["aciertos_pct"].mean().unstack()
for _, fila in por_tarea.iterrows():
    ax.plot([0, 1], [fila["1-back"], fila["3-back"]], color=GRIS_CLARO, lw=1.2, marker="o", ms=5,
            mfc=GRIS, mec=GRIS, zorder=2)
media(ax, 0, por_tarea["1-back"], AZUL)
media(ax, 1, por_tarea["3-back"], NARANJA)
f = P1a.iloc[0]
ax.set_xticks([0, 1])
ax.set_xticklabels(["1-back", "3-back"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Aciertos [%]")
ax.set_title(f"{int(f['sujetos_peor_en_3back'])} de {int(f['n_sujetos'])} sujetos aciertan menos\n"
             f"{f['diferencia_pp']:+.1f} pts · {estrellas(f['p_wilcoxon'])} p = {f['p_wilcoxon']:.3f}",
             fontsize=11, color=NEGRO)

ax = axs[1]
for m, tarea, color in [(acc1, "1-back", AZUL), (acc3, "3-back", NARANJA)]:
    ax.plot(x, m.mean(), color=color, lw=3, zorder=3)
    for k in x:
        media(ax, k, m.iloc[:, k], color, tam=10)
    p = fila_1b(tarea, musicas[0])["p_friedman_4_sesiones"]
    ax.text(x[-1] + 0.25, m.mean().iloc[-1], f"{tarea}\n{estrellas(p)} entre sesiones",
            va="center", fontsize=10, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([ETIQUETAS[s] for s in cfg.SESIONES])
ax.set_xlim(-0.3, len(x) + 0.6)
ax.set_ylabel("Aciertos [%]")
ax.set_title("La música solo cambia el 3-back\n(el 1-back está cerca del techo)", fontsize=11, color=NEGRO)

encabezado(fig, "El 3-back es claramente más difícil que el 1-back",
           "Izquierda: cada sujeto (Wilcoxon). Derecha: ambas tareas en las 4 sesiones (Friedman entre sesiones)",
           reserva=1.6)
nota(fig)
guardar(fig, "fig2_carga_aciertos.png")

# ===========================================================================
# Figura 3 – El desempeño mejora, pero beta y theta no acompañan (bloques de 3-back)
# ===========================================================================
n_eeg = len(marcadores)
fig = plt.figure(figsize=(3.1 * (n_eeg + 1) + 1, 5.2))
grid = fig.add_gridspec(1, n_eeg + 2, width_ratios=[1, 0.55] + [1] * n_eeg, wspace=0.12)
ax_c = fig.add_subplot(grid[0, 0])
axs_eeg = [fig.add_subplot(grid[0, 2])]
for i in range(1, n_eeg):
    axs_eeg.append(fig.add_subplot(grid[0, 2 + i], sharey=axs_eeg[0]))

# Panel de conducta
for k, s in enumerate(musicas):
    puntos_sujetos(ax_c, k, d_acc3[s], semilla=k)
    media(ax_c, k, d_acc3[s], NARANJA)
    marca_significancia(ax_c, k, fila_1b("3-back", s)["p_holm"])
ax_c.axhline(0, color=NEGRO, lw=0.8)
ax_c.set_title("Desempeño", fontsize=12, fontweight="bold", color=NARANJA, pad=22)
ax_c.set_ylabel("Cambio de aciertos vs sin música\n[puntos porcentuales]")

# Paneles EEG (mismo eje y para poder compararlos)
cambios = {m: cambio_eeg_3back(*m) for m in marcadores}
for i, (ax, (banda, canal)) in enumerate(zip(axs_eeg, marcadores)):
    cambio = cambios[(banda, canal)]
    for k, s in enumerate(musicas):
        puntos_sujetos(ax, k, cambio[s], semilla=10 + k)
        media(ax, k, cambio[s], COLOR_BANDA[banda], marcador="D", tam=10)
        p = P5[(P5["banda"] == banda) & (P5["canal"] == canal) & (P5["sesion"] == s)]["p_wilcoxon_holm"].iloc[0]
        marca_significancia(ax, k, p)
    ax.axhline(0, color=NEGRO, lw=0.8)
    ax.set_title(f"{banda} {canal}", fontsize=12, fontweight="bold", color=COLOR_BANDA[banda], pad=22)
    if i == 0:
        ax.set_ylabel("Cambio de potencia vs sin música [%]")
    else:
        ax.tick_params(labelleft=False)

for ax in [ax_c] + axs_eeg:
    ax.set_xticks(xm)
    ax.set_xticklabels([ETIQUETAS[s] for s in musicas], rotation=30, ha="right")
    ax.set_xlim(-0.5, len(musicas) - 0.5)
# Eje y simétrico en los paneles EEG, para que el 0 quede en el medio
lim = np.nanmax([np.nanmax(np.abs(c.to_numpy())) for c in cambios.values()]) * 1.1
axs_eeg[0].set_ylim(-lim, lim)

eeg_sig = (P5[P5.set_index(["banda", "canal"]).index.isin(marcadores)]["p_wilcoxon_holm"] < 0.05).any()
encabezado(fig, "El desempeño mejora con la música" +
           (", y el EEG cambia en algún marcador" if eeg_sig else ", pero beta y theta no acompañan"),
           "Bloques de 3-back: cambio de cada música respecto de sin música, en la conducta y en el EEG",
           reserva=1.55)
izq, der = ax_c.get_position(), axs_eeg[0].get_position()
fig.text(izq.x1 + 0.3 * (der.x0 - izq.x1), (der.y0 + der.y1) / 2, "≠", fontsize=34, color=TINTA_2,
         ha="center", va="center")
nota(fig)
guardar(fig, "fig3_eeg_vs_conducta.png")

# ===========================================================================
# Figura 3b – Solo 3-back: ¿la potencia cambia entre las 4 sesiones?
# ===========================================================================
def p_global(tabla, banda, canal, columna):
    t = tabla[(tabla["banda"] == banda) & (tabla["canal"] == canal)]
    return t[columna].iloc[0] if len(t) else np.nan


fig, axs = plt.subplots(1, len(marcadores), figsize=(3.6 * len(marcadores), 5.4), sharey=True)
axs = np.atleast_1d(axs)
todos = []
for i, (ax, (banda, canal)) in enumerate(zip(axs, marcadores)):
    cambio = cambio_eeg_3back(banda, canal)                       # % vs sin música, 3-back
    cambio.insert(0, REF, 0.0)                                    # sin música = 0 por definición
    cambio = cambio[cambio[musicas].notna().any(axis=1)]
    todos.append(cambio[musicas].to_numpy())
    for _, fila in cambio.iterrows():                             # una línea por sujeto
        ax.plot(x, fila, color=GRIS_CLARO, lw=1.1, marker="o", ms=3.5, mfc=GRIS, mec=GRIS, zorder=2)
    medias = cambio.mean()
    ax.plot(x, medias, color=COLOR_BANDA[banda], lw=2.5, zorder=3)
    for k, s in enumerate(cfg.SESIONES):
        if s == REF:
            ax.plot(k, 0, "o", color=COLOR_BANDA[banda], ms=10, mec="white", mew=1.5, zorder=4)
            continue
        media(ax, k, cambio[s], COLOR_BANDA[banda], marcador="D", tam=9)
        p = P3b[(P3b["banda"] == banda) & (P3b["canal"] == canal) & (P3b["sesion"] == s)]["p_holm"]
        marca_significancia(ax, k, p.iloc[0] if len(p) else np.nan)
    ax.axhline(0, color=NEGRO, lw=0.8)

    # Test global de las 4 sesiones: solo 3-back vs sesión completa (Parte 3)
    p_fr3 = p_global(P3b, banda, canal, "p_friedman_4_sesiones")
    p_rm3 = p_global(P3b, banda, canal, "p_anova_rm_4_sesiones")
    p_frc = p_global(P3, banda, canal, "p_friedman_4_sesiones")
    ax.text(0.5, -0.34,
            f"Solo 3-back: Friedman p = {p_fr3:.2f} {estrellas(p_fr3)}\n"
            f"ANOVA-MR p = {p_rm3:.2f} {estrellas(p_rm3)}\n"
            f"Sesión completa: Friedman p = {p_frc:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=TINTA_2)
    ax.set_title(f"{banda} {canal}", fontsize=12, fontweight="bold", color=COLOR_BANDA[banda], pad=22)
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[s] for s in cfg.SESIONES], rotation=30, ha="right")
    ax.set_xlim(-0.4, len(x) - 0.6)
    if i == 0:
        ax.set_ylabel("Cambio de potencia en 3-back\nvs sin música [%]")
    else:
        ax.tick_params(labelleft=False)
lim = np.nanmax(np.abs(np.concatenate([t.ravel() for t in todos]))) * 1.1
axs[0].set_ylim(-lim, lim)

marc = P3b.set_index(["banda", "canal"]).index.isin(marcadores)
hay_sig = ((P3b[marc]["p_holm"] < 0.05) | (P3b[marc]["p_friedman_4_sesiones"] < 0.05)
           | (P3b[marc]["p_anova_rm_4_sesiones"] < 0.05)).any()
encabezado(fig, "Solo con 3-back, algún marcador cambia entre sesiones" if hay_sig else
           "Aun usando solo el 3-back, beta y theta no cambian con la música",
           "Bloques de 3-back (sin el 1-back, cerca del techo de aciertos, ni los tramos sin tarea); "
           "cada música vs sin música", reserva=1.55)
fig.subplots_adjust(bottom=0.40)
fig.text(0.01, 0.03, NOTA + "\nAbajo de cada panel: test de las 4 sesiones juntas (sin corregir entre "
         "canales), solo 3-back y, como referencia, con la sesión completa (Parte 3)",
         ha="left", va="bottom", fontsize=8.5, color=TINTA_2)
guardar(fig, "fig3b_musica_eeg_3back.png")

# ===========================================================================
# Figura 4 – El paper (un sujeto) vs el grupo: beta en los canales del paper
# ===========================================================================
temporales = [c for c in ["TP9", "TP10"] if c in potencia["canal"].unique()]
d = potencia[(potencia["banda"] == "beta") & potencia["canal"].isin(temporales)]
mm = d.groupby(["sujeto", "sesion"])["potencia_rel"].mean().unstack().reindex(columns=cfg.SESIONES)
cambio = a_pct(np.log10(mm[musicas].div(mm[REF], axis=0)))
a9 = cambio.loc["A9"] if "A9" in cambio.index else None

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ancho = 0.25
barras_paper = ax.bar(xm - ancho, [PAPER_A9[s] for s in musicas], ancho, color=NEGRO,
                      label="Paper: sujeto A9")
ax.bar_label(barras_paper, fmt="+%.1f%%", padding=3, fontsize=9.5, color=NEGRO)
if a9 is not None:
    barras_a9 = ax.bar(xm, a9.to_numpy(), ancho, color=GRIS_CLARO, edgecolor=TINTA_2, linewidth=0.8,
                       label="A9 con nuestro procesamiento")
    ax.bar_label(barras_a9, fmt="%+.1f%%", padding=3, fontsize=9.5, color=TINTA_2)
ax.bar(xm + ancho, cambio.mean(), ancho, color=AZUL, alpha=0.85,
       label=f"Grupo: {cambio.notna().sum().max()} sujetos (media ± e.e.)")
for k, s in enumerate(musicas):
    puntos_sujetos(ax, k + ancho, cambio[s], semilla=k, ancho=0.05)
    ax.errorbar(k + ancho, cambio[s].mean(), yerr=cambio[s].sem(), fmt="none", ecolor=NEGRO,
                capsize=5, elinewidth=1.8, capthick=1.8, zorder=4)
    t = P3[(P3["banda"] == "beta") & P3["canal"].isin(temporales) & (P3["sesion"] == s)]
    mejor = t.loc[t["p_holm"].idxmin()]
    flecha = (" ↑" if mejor["cambio_pct"] > 0 else " ↓") if mejor["p_holm"] < 0.05 else ""
    marca_significancia(ax, k + ancho, mejor["p_holm"], flecha)
ax.axhline(0, color=NEGRO, lw=0.8)
ax.set_xticks(xm)
ax.set_xticklabels([ETIQUETAS[s] for s in musicas], fontsize=11.5)
ax.set_ylabel("Cambio de beta (TP9/TP10) vs sin música [%]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=10)
sube_sig = ((P3["banda"] == "beta") & P3["canal"].isin(temporales)
            & (P3["p_holm"] < 0.05) & (P3["cambio_pct"] > 0)).any()
encabezado(fig, "El aumento de beta del paper se ve en parte del grupo" if sube_sig else
           "El aumento de beta que el paper muestra en un sujeto no se ve en el grupo",
           "Beta relativa en los canales del paper (TP9/TP10), toda la sesión; "
           "arriba del grupo: mejor p Holm (↑ sube, ↓ baja)", reserva=1.15)
nota(fig, "Puntos grises: cada sujeto · p corregido por Holm · * p<0.05  ** p<0.01  n.s. no significativo")
guardar(fig, "fig4_paper_vs_grupo.png")

# ===========================================================================
# ANEXO – Figura 5: mismo dato, dos análisis (beta)
# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
ax = axs[0]
r = P2.groupby("canal")[["suben_signif", "bajan_signif", "no_signif"]].sum()
r = r.reindex([c for c in ["AF7", "AF8", "TP9", "TP10"] if c in r.index])
y = np.arange(len(r))
izq = np.zeros(len(r))
for col, color, etiqueta in [("suben_signif", AZUL, "sube (p < 0.05)"),
                             ("bajan_signif", NARANJA, "baja (p < 0.05)"),
                             ("no_signif", GRIS_CLARO, "no significativo")]:
    barras = ax.barh(y, r[col], left=izq, color=color, height=0.62, label=etiqueta,
                     edgecolor="white", linewidth=1.5)
    ax.bar_label(barras, labels=[str(int(v)) if v > 0 else "" for v in r[col]], label_type="center",
                 fontsize=10, color="white" if color != GRIS_CLARO else TINTA_2, fontweight="bold")
    izq += r[col].to_numpy()
ax.set_yticks(y)
ax.set_yticklabels(r.index, fontsize=11)
ax.invert_yaxis()
ax.grid(False)
ax.set_xlabel("Comparaciones sujeto × música")
pct = 100 * (r["suben_signif"].sum() + r["bajan_signif"].sum()) / r.to_numpy().sum()
opuestas = r["suben_signif"].sum() > 0 and r["bajan_signif"].sum() > 0
ax.set_title(f"Método del paper (ANOVA sobre ventanas):\n{pct:.0f}% \"significativo\""
             + (", en direcciones opuestas" if opuestas else ""), fontsize=11, color=NEGRO, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9.5)

ax = axs[1]
canales_beta = [c for c in ["AF7", "AF8", "TP9", "TP10"] if c in potencia["canal"].unique()]
for k, canal in enumerate(canales_beta):
    d = potencia[(potencia["banda"] == "beta") & (potencia["canal"] == canal)]
    mm = d.pivot_table(index="sujeto", columns="sesion", values="potencia_rel").reindex(columns=cfg.SESIONES)
    c = a_pct(np.log10(mm[musicas].div(mm[REF], axis=0)).mean(axis=1)).dropna()
    puntos_sujetos(ax, k, c, semilla=k, ancho=0.12)
    media(ax, k, c, NEGRO, marcador="D", tam=9)
    marca_significancia(ax, k, P3[(P3["banda"] == "beta") & (P3["canal"] == canal)]["p_holm"].min())
ax.axhline(0, color=NEGRO, lw=0.8)
ax.set_xticks(range(len(canales_beta)))
ax.set_xticklabels(canales_beta, fontsize=11)
ax.set_ylabel("Cambio de beta con música [%]")
sig3 = P3[(P3["banda"] == "beta") & (P3["p_holm"] < 0.05)]["canal"].unique().tolist()
ax.set_title("Sujeto como unidad:\n" + ("ningún canal cambia de forma consistente" if not sig3
             else f"cambio consistente solo en {', '.join(sig3)}"),
             fontsize=11, color=NEGRO, loc="left", pad=18)
encabezado(fig, "Anexo · Mismo dato, dos análisis",
           "Beta relativa: el método del paper (izquierda) frente al sujeto como unidad de análisis (derecha)",
           reserva=1.7)
nota(fig, "Derecha: gris = cada sujeto (promedio de las 3 músicas) · negro = media ± e.e. · "
          "arriba: mejor p Holm del canal")
guardar(fig, "fig5_dos_analisis.png")

# ===========================================================================
# ANEXO – Figura 6: correlación entre el cambio del EEG y el de los aciertos
# ===========================================================================
SESION_FIG6 = "musica_relajante"         # la música que mejoró significativamente
fig, axs = plt.subplots(1, len(marcadores), figsize=(3.7 * len(marcadores), 4.3), sharey=True)
for ax, (banda, canal) in zip(np.atleast_1d(axs), marcadores):
    juntos = pd.concat([cambio_eeg_3back(banda, canal)[SESION_FIG6].rename("eeg"),
                        d_acc3[SESION_FIG6].rename("acc")], axis=1).dropna()
    ax.axhline(0, color=GRIS, lw=0.8, zorder=1)
    ax.axvline(0, color=GRIS, lw=0.8, zorder=1)
    ax.scatter(juntos["eeg"], juntos["acc"], color=COLOR_BANDA[banda], s=60, edgecolor="white",
               linewidths=1, zorder=3)
    f = P5[(P5["banda"] == banda) & (P5["canal"] == canal) & (P5["sesion"] == SESION_FIG6)].iloc[0]
    ax.set_title(f"{banda} {canal}", fontsize=12, fontweight="bold", color=COLOR_BANDA[banda])
    ax.text(0.03, 0.97, f"rho = {f['rho_spearman']:+.2f}\n{estrellas(f['p_spearman_holm'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRIS_CLARO})
    ax.set_xlabel(f"Cambio de {banda} [%]")
    ax.grid(True, axis="both")
np.atleast_1d(axs)[0].set_ylabel("Cambio de aciertos [puntos %]")
t = P5[(P5["sesion"] == SESION_FIG6) & P5.set_index(["banda", "canal"]).index.isin(marcadores)]
encabezado(fig, "Anexo · " + ("En algún marcador, el cambio del EEG se relaciona con la mejora"
                              if (t["p_spearman_holm"] < 0.05).any() else
                              "Los sujetos que más mejoran no son los que más cambian el EEG"),
           f"Música {ETIQUETAS[SESION_FIG6].lower()} vs sin música, bloques de 3-back · un punto = un sujeto",
           reserva=1.25)
nota(fig, "rho: correlación de Spearman (−1 a +1; 0 = sin relación) · p corregido por Holm")
guardar(fig, "fig6_correlacion.png")