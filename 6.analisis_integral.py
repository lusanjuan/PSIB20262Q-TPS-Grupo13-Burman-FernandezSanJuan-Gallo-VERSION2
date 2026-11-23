"""
Paso 6: comparación de grupo (todos los sujetos). Solo descriptivo: promedios,
cambios y gráficos. Los tests estadísticos quedan para el paso 7.

Parte de las tablas del paso 5 (una fila por sujeto, sesión, canal, tarea y banda).

Cómo se tratan los datos faltantes (no se promedian ceros):
  - El paso 5 ya promedia solo las ventanas ACEPTADAS. Las descartadas no entran.
  - Si un sujeto no tiene ventanas aceptadas en una sesión/canal/tarea, esa fila
    no existe: queda como NaN, no como 0.
  - Si tiene menos de MIN_VENTANAS ventanas aceptadas, también se pasa a NaN.
  - Primero se calcula un valor por sujeto y después se promedian los sujetos:
    cada sujeto pesa lo mismo, y cada celda se promedia con los sujetos que
    tienen dato (n_sujetos indica cuántos).

Tres comparaciones (Opción B):
  1. CARGA:       3-back respecto de 1-back
  2. MÚSICA:      cada sesión respecto de sin_musica
  3. INTERACCIÓN: la diferencia 3-back vs 1-back en cada sesión

Medida de cambio: log10(condición / referencia), que es simétrica (subir al doble
y bajar a la mitad pesan igual). Para mostrarla se pasa a %: 100 * (10**x - 1).

Guarda en resultados/06_grupo/:
  - promedio_grupo.csv:     media y error estándar por canal, banda, sesión y tarea
  - carga_por_sujeto.csv:   cambio 3-back vs 1-back de cada sujeto (para el paso 7)
  - musica_por_sujeto.csv:  cambio de cada sesión vs sin_musica de cada sujeto (paso 7)
  - resumen_carga.csv, resumen_musica.csv, resumen_interaccion.csv: medias del grupo
  - conducta.csv:           aciertos por sujeto, sesión y tarea
  - grupo_interaccion.png, grupo_efecto_carga.png, grupo_efecto_musica.png,
    grupo_psd.png, grupo_conducta.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

DIR_NBACK = cfg.DIR_RESULTADOS / "05_nback"
DIR_GRUPO = cfg.DIR_RESULTADOS / "06_grupo"
DIR_GRUPO.mkdir(parents=True, exist_ok=True)

VARIABLE = "potencia_rel"      # o "potencia_abs_uV2"
MIN_VENTANAS = 20              # mínimo de ventanas aceptadas para usar un valor
TAREAS = ["1-back", "3-back"]
COLORES_TAREA = {"1-back": "tab:blue", "3-back": "tab:red"}
canales = cfg.CANALES_ANALISIS
bandas = list(cfg.BANDAS)
x = np.arange(len(cfg.SESIONES))


def a_pct(log_cambio):
    """Pasa log10(cociente) a % de cambio."""
    return 100 * (10 ** log_cambio - 1)


# ---------------------------------------------------------------------------
# 0. Cargar y limpiar
# ---------------------------------------------------------------------------
potencia = pd.read_csv(DIR_NBACK / "potencia_nback.csv")
potencia = potencia[potencia["canal"].isin(canales)]

pocas = potencia["n_ventanas_aceptadas"] < MIN_VENTANAS
print(f"Valores con menos de {MIN_VENTANAS} ventanas aceptadas (pasan a NaN): "
      f"{pocas.sum()} de {len(potencia)}")
potencia.loc[pocas, VARIABLE] = np.nan

sujetos = sorted(potencia["sujeto"].unique(), key=lambda s: int(s[1:]))
print(f"Sujetos: {sujetos}")

# Tabla ancha: una fila por sujeto/sesión/canal/banda, columnas 1-back y 3-back
ancha = potencia.pivot_table(index=["sujeto", "sesion", "canal", "banda"],
                             columns="tarea", values=VARIABLE, dropna=False).reset_index()

# ---------------------------------------------------------------------------
# 1. Promedio del grupo por sesión y tarea (media ± error estándar entre sujetos)
# ---------------------------------------------------------------------------
promedio = (potencia.groupby(["canal", "banda", "sesion", "tarea"])[VARIABLE]
            .agg(n_sujetos="count", media="mean", sem="sem").reset_index())
promedio.round(5).to_csv(DIR_GRUPO / "promedio_grupo.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Carga: log10(3-back / 1-back) por sujeto, promediado en las 4 sesiones
# ---------------------------------------------------------------------------
ancha["log_3vs1"] = np.log10(ancha["3-back"] / ancha["1-back"])
carga_sujeto = ancha.groupby(["sujeto", "canal", "banda"])["log_3vs1"].mean().reset_index()
carga_sujeto["cambio_pct"] = a_pct(carga_sujeto["log_3vs1"])
carga_sujeto.round(5).to_csv(DIR_GRUPO / "carga_por_sujeto.csv", index=False)

resumen_carga = carga_sujeto.groupby(["canal", "banda"]).agg(
    n_sujetos=("log_3vs1", "count"),
    log_medio=("log_3vs1", "mean"),
    sujetos_que_suben=("log_3vs1", lambda v: (v > 0).sum()),
    sujetos_que_bajan=("log_3vs1", lambda v: (v < 0).sum()),
).reset_index()
resumen_carga["cambio_medio_pct"] = a_pct(resumen_carga["log_medio"])
resumen_carga = resumen_carga.drop(columns="log_medio")
resumen_carga.round(2).to_csv(DIR_GRUPO / "resumen_carga.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Música: cada sesión vs sin_musica (promedio de 1-back y 3-back)
# ---------------------------------------------------------------------------
ancha["ambas"] = ancha[TAREAS].mean(axis=1, skipna=False)   # si falta una tarea, NaN
musica = ancha.pivot_table(index=["sujeto", "canal", "banda"], columns="sesion",
                           values="ambas", dropna=False).reindex(columns=cfg.SESIONES)
log_musica = np.log10(musica.div(musica[cfg.SESION_REFERENCIA], axis=0))

musica_sujeto = log_musica.reset_index().melt(id_vars=["sujeto", "canal", "banda"],
                                              var_name="sesion", value_name="log_vs_ref")
musica_sujeto = musica_sujeto[musica_sujeto["sesion"] != cfg.SESION_REFERENCIA]
musica_sujeto["cambio_pct"] = a_pct(musica_sujeto["log_vs_ref"])
musica_sujeto.round(5).to_csv(DIR_GRUPO / "musica_por_sujeto.csv", index=False)

resumen_musica = musica_sujeto.groupby(["canal", "banda", "sesion"], sort=False).agg(
    n_sujetos=("log_vs_ref", "count"),
    log_medio=("log_vs_ref", "mean"),
    sujetos_que_suben=("log_vs_ref", lambda v: (v > 0).sum()),
    sujetos_que_bajan=("log_vs_ref", lambda v: (v < 0).sum()),
).reset_index()
resumen_musica["cambio_medio_pct"] = a_pct(resumen_musica["log_medio"])
resumen_musica = resumen_musica.drop(columns="log_medio")
resumen_musica.round(2).to_csv(DIR_GRUPO / "resumen_musica.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Interacción: diferencia 3-back vs 1-back en cada sesión (media del grupo)
# ---------------------------------------------------------------------------
resumen_inter = ancha.groupby(["canal", "banda", "sesion"])["log_3vs1"].agg(
    n_sujetos="count", log_medio="mean").reset_index()
resumen_inter["carga_pct"] = a_pct(resumen_inter["log_medio"])
resumen_inter = resumen_inter.pivot_table(index=["canal", "banda"], columns="sesion",
                                          values="carga_pct").reindex(columns=cfg.SESIONES)
resumen_inter.round(2).to_csv(DIR_GRUPO / "resumen_interaccion.csv")

# ---------------------------------------------------------------------------
# 5. Conducta: aciertos por sujeto, sesión y tarea
# ---------------------------------------------------------------------------
bloques = pd.read_csv(DIR_NBACK / "bloques.csv")
conducta = bloques.groupby(["sujeto", "sesion", "tarea"])["aciertos_pct"].mean().reset_index()
conducta.round(2).to_csv(DIR_GRUPO / "conducta.csv", index=False)

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

# --- Figura 1: interacción. Filas = bandas, columnas = canales ---
fig, axs = plt.subplots(len(bandas), len(canales),
                        figsize=(4.2 * len(canales), 3 * len(bandas)), squeeze=False)
for fila, banda in enumerate(bandas):
    for col, canal in enumerate(canales):
        ax = axs[fila, col]
        for k, tarea in enumerate(TAREAS):
            d = promedio[(promedio["canal"] == canal) & (promedio["banda"] == banda)
                         & (promedio["tarea"] == tarea)].set_index("sesion").reindex(cfg.SESIONES)
            ax.errorbar(x + (k - 0.5) * 0.1, d["media"], yerr=d["sem"], marker="o",
                        capsize=3, color=COLORES_TAREA[tarea], label=tarea)
        ax.set_xticks(x)
        ax.set_xticklabels(cfg.SESIONES, rotation=20, fontsize=8)
        if fila == 0:
            ax.set_title(canal)
        if col == 0:
            ax.set_ylabel(f"{banda}\n{VARIABLE}")
        ax.grid(alpha=0.3)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(0.5, 0.965), ncol=2)
fig.suptitle(f"Promedio de {len(sujetos)} sujetos (media ± error estándar)")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(DIR_GRUPO / "grupo_interaccion.png", dpi=150)
plt.close(fig)

# --- Figura 2: carga, un punto por sujeto ---
fig, axs = plt.subplots(1, len(bandas), figsize=(4.5 * len(bandas), 4.5), sharey=True)
for ax, banda in zip(axs, bandas):
    for c, canal in enumerate(canales):
        v = carga_sujeto[(carga_sujeto["canal"] == canal)
                         & (carga_sujeto["banda"] == banda)]["cambio_pct"].dropna()
        ax.scatter(np.full(len(v), c) + np.random.uniform(-0.1, 0.1, len(v)), v,
                   color="0.5", s=20, zorder=2)
        ax.scatter(c, a_pct(np.log10(1 + v / 100).mean()), color="k", marker="_",
                   s=600, zorder=3)
        ax.text(c, 1.0, f"{(v < 0).sum()}↓ {(v > 0).sum()}↑",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.set_xticks(range(len(canales)))
    ax.set_xticklabels(canales)
    ax.set_title(banda)
    ax.grid(alpha=0.3, axis="y")
axs[0].set_ylabel("Cambio 3-back vs 1-back [%]")
fig.suptitle("Efecto de la carga: cada sujeto (gris) y media del grupo (negro); "
             "↓↑ = sujetos que bajan / suben")
fig.tight_layout()
fig.savefig(DIR_GRUPO / "grupo_efecto_carga.png", dpi=150)
plt.close(fig)

# --- Figura 3: música, una línea por sujeto ---
fig, axs = plt.subplots(len(bandas), len(canales),
                        figsize=(4.2 * len(canales), 3 * len(bandas)), squeeze=False, sharex=True)
for fila, banda in enumerate(bandas):
    for col, canal in enumerate(canales):
        ax = axs[fila, col]
        m = log_musica.xs((canal, banda), level=["canal", "banda"])
        for _, valores in m.iterrows():
            ax.plot(x, a_pct(valores), color="0.7", lw=0.8)
        ax.plot(x, a_pct(m.mean()), color="k", lw=2, marker="o")
        ax.axhline(0, color="r", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cfg.SESIONES, rotation=20, fontsize=8)
        if fila == 0:
            ax.set_title(canal)
        if col == 0:
            ax.set_ylabel(f"{banda}\ncambio vs {cfg.SESION_REFERENCIA} [%]")
        ax.grid(alpha=0.3)
fig.suptitle("Efecto de la música: cada sujeto (gris) y media del grupo (negro)")
fig.tight_layout()
fig.savefig(DIR_GRUPO / "grupo_efecto_musica.png", dpi=150)
plt.close(fig)

# --- Figura 4: PSD promedio del grupo, 1-back vs 3-back ---
psd = pd.read_csv(DIR_NBACK / "psd_nback.csv")
psd = psd[psd["canal"].isin(canales) & psd["frecuencia_Hz"].between(*cfg.BANDA_TOTAL)]
# Por sujeto: promedio de las 4 sesiones (en log, porque la PSD varía mucho entre sujetos)
psd["log_psd"] = np.log10(psd["psd_uV2_Hz"])
psd_sujeto = psd.groupby(["sujeto", "canal", "tarea", "frecuencia_Hz"])["log_psd"].mean().reset_index()

fig, axs = plt.subplots(1, len(canales), figsize=(4.5 * len(canales), 4), sharey=True,
                        squeeze=False)
for ax, canal in zip(axs[0], canales):
    for banda, (lo, hi) in cfg.BANDAS.items():
        ax.axvspan(lo, hi, color="0.9" if banda == "alfa" else "0.96")
    for tarea in TAREAS:
        d = psd_sujeto[(psd_sujeto["canal"] == canal) & (psd_sujeto["tarea"] == tarea)]
        g = d.groupby("frecuencia_Hz")["log_psd"].agg(["mean", "sem"])
        ax.plot(g.index, 10 ** g["mean"], color=COLORES_TAREA[tarea], label=tarea)
        ax.fill_between(g.index, 10 ** (g["mean"] - g["sem"]), 10 ** (g["mean"] + g["sem"]),
                        color=COLORES_TAREA[tarea], alpha=0.2)
    ax.set_yscale("log")
    ax.set_title(canal)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.grid(alpha=0.3, which="both")
axs[0, 0].set_ylabel("PSD [µV²/Hz] (media geométrica)")
axs[0, 0].legend()
fig.suptitle("PSD promedio del grupo (4 sesiones juntas), media ± error estándar")
fig.tight_layout()
fig.savefig(DIR_GRUPO / "grupo_psd.png", dpi=150)
plt.close(fig)

# --- Figura 5: conducta ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for tarea in TAREAS:
    d = conducta[conducta["tarea"] == tarea].pivot_table(
        index="sujeto", columns="sesion", values="aciertos_pct").reindex(columns=cfg.SESIONES)
    for _, valores in d.iterrows():
        ax.plot(x, valores, color=COLORES_TAREA[tarea], alpha=0.25, lw=0.8)
    ax.errorbar(x, d.mean(), yerr=d.sem(), color=COLORES_TAREA[tarea], lw=2,
                marker="o", capsize=3, label=tarea)
ax.set_xticks(x)
ax.set_xticklabels(cfg.SESIONES, rotation=15)
ax.set_ylabel("Aciertos [%]")
ax.set_title("Aciertos por sesión: cada sujeto (fino) y media ± error estándar (grueso)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(DIR_GRUPO / "grupo_conducta.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Resumen en pantalla
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200)
acc = conducta.groupby("tarea")["aciertos_pct"].mean()
print(f"\nCONDUCTA: aciertos medios 1-back {acc['1-back']:.1f}%, 3-back {acc['3-back']:.1f}%")
print("\n1. CARGA (3-back vs 1-back, promedio de las 4 sesiones):")
print(resumen_carga.to_string(index=False))
print("\n2. MÚSICA (cada sesión vs sin_musica):")
print(resumen_musica.to_string(index=False))
print("\n3. INTERACCIÓN (% de cambio 3-back vs 1-back en cada sesión):")
print(resumen_inter.round(1).to_string())