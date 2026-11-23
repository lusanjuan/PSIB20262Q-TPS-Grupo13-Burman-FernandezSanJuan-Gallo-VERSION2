"""
Paso 7: promedio de todos los sujetos de las figuras bandas_<sujeto>.png del paso 4.

Una sola figura, un panel por banda (theta, alfa, beta). En cada panel, para
cada estímulo (sesión de música) y canal:
  - caja (boxplot): distribución de la potencia relativa entre sujetos
    (línea = mediana, caja = 50% central, bigotes = rango sin valores atípicos)
  - puntos grises: cada sujeto
  - rombo blanco: media del grupo
  - número arriba: cambio medio del grupo respecto de sin_musica [%]

Datos faltantes: un sujeto sin ventanas aceptadas en una sesión/canal no tiene
fila en potencia_bandas.csv, así que no entra (no se cuenta como 0). Los valores
con menos de MIN_VENTANAS ventanas aceptadas también se dejan afuera.

Cambio respecto de sin_musica: se calcula por sujeto como log10(sesión / sin_musica),
se promedia entre sujetos y se pasa a %: 100 * (10**media - 1).

Guarda en resultados/07_promedios_psd/:
  - promedio_bandas.png
  - beta_<canal>.png: una figura por canal, solo beta relativa, con una línea por sujeto
  - promedio_bandas.csv: n_sujetos, media, mediana, error estándar y cambio medio
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

DIR_PSD = cfg.DIR_RESULTADOS / "04_psd"
DIR_SALIDA = cfg.DIR_RESULTADOS / "07_promedios_psd"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

MIN_VENTANAS = 20
COLORES_CANAL = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
canales = cfg.CANALES_ANALISIS
bandas = list(cfg.BANDAS)

# ---------------------------------------------------------------------------
# 1. Cargar y limpiar
# ---------------------------------------------------------------------------
tabla = pd.read_csv(DIR_PSD / "potencia_bandas.csv")
tabla = tabla[tabla["canal"].isin(canales)]
pocas = tabla["n_ventanas_aceptadas"] < MIN_VENTANAS
print(f"Valores con menos de {MIN_VENTANAS} ventanas aceptadas (se dejan afuera): {pocas.sum()}")
tabla = tabla[~pocas]
sujetos = sorted(tabla["sujeto"].unique(), key=lambda s: int(s[1:]))
print(f"Sujetos: {sujetos}")

# ---------------------------------------------------------------------------
# 2. Cambio de cada sujeto respecto de sin_musica (en log, después a %)
# ---------------------------------------------------------------------------
referencia = tabla[tabla["sesion"] == cfg.SESION_REFERENCIA]
referencia = referencia[["sujeto", "canal", "banda", "potencia_rel"]].rename(
    columns={"potencia_rel": "rel_referencia"})
tabla = tabla.merge(referencia, on=["sujeto", "canal", "banda"], how="left")
tabla["log_vs_ref"] = np.log10(tabla["potencia_rel"] / tabla["rel_referencia"])

# ---------------------------------------------------------------------------
# 3. Resumen del grupo
# ---------------------------------------------------------------------------
resumen = tabla.groupby(["banda", "canal", "sesion"]).agg(
    n_sujetos=("potencia_rel", "count"),
    media=("potencia_rel", "mean"),
    mediana=("potencia_rel", "median"),
    sem=("potencia_rel", "sem"),
    log_medio=("log_vs_ref", "mean"),
).reset_index()
resumen["cambio_medio_pct"] = 100 * (10 ** resumen["log_medio"] - 1)
resumen.loc[resumen["sesion"] == cfg.SESION_REFERENCIA, "cambio_medio_pct"] = np.nan
resumen = resumen.drop(columns="log_medio")
resumen.round(4).to_csv(DIR_SALIDA / "promedio_bandas.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Figura: un panel por banda, cajas por estímulo y canal
# ---------------------------------------------------------------------------
ancho = 0.8 / len(canales)
posiciones = np.arange(len(cfg.SESIONES))
rng = np.random.default_rng(0)               # para separar un poco los puntos

fig, axs = plt.subplots(1, len(bandas), figsize=(6 * len(bandas), 5.5))
for ax, banda in zip(axs, bandas):
    for c, canal in enumerate(canales):
        color = COLORES_CANAL[c % len(COLORES_CANAL)]
        x_canal = posiciones + (c - (len(canales) - 1) / 2) * ancho

        for s, sesion in enumerate(cfg.SESIONES):
            d = tabla[(tabla["banda"] == banda) & (tabla["canal"] == canal)
                      & (tabla["sesion"] == sesion)]
            valores = d["potencia_rel"].dropna().to_numpy()
            if len(valores) == 0:
                continue
            x = x_canal[s]

            # Caja
            ax.boxplot(valores, positions=[x], widths=ancho * 0.85, patch_artist=True,
                       showfliers=False,
                       boxprops={"facecolor": color, "alpha": 0.45, "edgecolor": color},
                       medianprops={"color": "k"},
                       whiskerprops={"color": color}, capprops={"color": color})
            # Un punto por sujeto
            ax.scatter(x + rng.uniform(-ancho * 0.2, ancho * 0.2, len(valores)), valores,
                       color="0.3", s=8, zorder=3)
            # Media del grupo
            ax.scatter(x, valores.mean(), marker="D", color="white", edgecolor="k",
                       s=28, zorder=4)
            # Cambio medio respecto de sin_musica
            r = resumen[(resumen["banda"] == banda) & (resumen["canal"] == canal)
                        & (resumen["sesion"] == sesion)]["cambio_medio_pct"]
            if len(r) and not np.isnan(r.iloc[0]):
                ax.text(x, 1.0, f"{r.iloc[0]:+.0f}%", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, rotation=90, color=color)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(cfg.SESIONES, rotation=20)
    ax.set_xlim(-0.6, len(cfg.SESIONES) - 0.4)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2)     # lugar arriba para los %
    ax.set_title(f"{banda} {cfg.BANDAS[banda][0]:g}–{cfg.BANDAS[banda][1]:g} Hz")
    ax.grid(alpha=0.3, axis="y")
axs[0].set_ylabel("Potencia relativa (banda / 1–45 Hz)")

# Leyenda: colores de canal + qué es cada símbolo
leyenda = [plt.Rectangle((0, 0), 1, 1, color=COLORES_CANAL[c % len(COLORES_CANAL)], alpha=0.45,
                         label=canal) for c, canal in enumerate(canales)]
leyenda += [plt.Line2D([], [], marker="D", color="white", markeredgecolor="k", ls="",
                       label="media"),
            plt.Line2D([], [], marker="o", color="0.3", ls="", markersize=3, label="sujeto")]
axs[-1].legend(handles=leyenda, loc="center left", bbox_to_anchor=(1, 0.5))

fig.suptitle(f"Potencia relativa por estímulo – {len(sujetos)} sujetos "
             f"(% arriba = cambio medio respecto de {cfg.SESION_REFERENCIA})")
fig.tight_layout()
fig.savefig(DIR_SALIDA / "promedio_bandas.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Una figura por canal, solo beta relativa
# ---------------------------------------------------------------------------
BANDA_CANAL = "beta"
for c, canal in enumerate(canales):
    color = COLORES_CANAL[c % len(COLORES_CANAL)]
    d = tabla[(tabla["banda"] == BANDA_CANAL) & (tabla["canal"] == canal)]
    # Matriz sujetos x sesiones (NaN donde el sujeto no tiene dato)
    m = d.pivot_table(index="sujeto", columns="sesion", values="potencia_rel")
    m = m.reindex(columns=cfg.SESIONES)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Una línea por sujeto, para ver si cada uno sube o baja
    for sujeto, valores in m.iterrows():
        ax.plot(posiciones, valores.to_numpy(), color="0.6", lw=0.8, marker="o", ms=3, zorder=2)

    for s, sesion in enumerate(cfg.SESIONES):
        valores = m[sesion].dropna().to_numpy()
        if len(valores) == 0:
            continue
        ax.boxplot(valores, positions=[s], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
                   medianprops={"color": "k"},
                   whiskerprops={"color": color}, capprops={"color": color})
        ax.scatter(s, valores.mean(), marker="D", color="white", edgecolor="k", s=40, zorder=4)
        r = resumen[(resumen["banda"] == BANDA_CANAL) & (resumen["canal"] == canal)
                    & (resumen["sesion"] == sesion)]
        texto = f"n={int(r['n_sujetos'].iloc[0])}"
        if not np.isnan(r["cambio_medio_pct"].iloc[0]):
            texto = f"{r['cambio_medio_pct'].iloc[0]:+.0f}%\n" + texto
        ax.text(s, 1.0, texto, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(cfg.SESIONES, rotation=15)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2)
    lo, hi = cfg.BANDAS[BANDA_CANAL]
    ax.set_ylabel(f"Potencia {BANDA_CANAL} relativa ({lo:g}–{hi:g} Hz / 1–45 Hz)")
    ax.set_title(f"{canal} – {BANDA_CANAL} relativa por estímulo\n"
                 f"(gris: cada sujeto; rombo: media; % = cambio medio vs {cfg.SESION_REFERENCIA})",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / f"{BANDA_CANAL}_{canal}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Resumen en pantalla
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200)
print("\nCambio medio respecto de sin_musica [%] (n = sujetos con dato):")
r = resumen[resumen["sesion"] != cfg.SESION_REFERENCIA]
print(r.pivot_table(index=["banda", "canal"], columns="sesion",
                    values="cambio_medio_pct").reindex(columns=cfg.SESIONES[1:]).round(1).to_string())