"""
Paso 5: PSD de 1-back y 3-back dentro de cada estímulo (sesión de música).

Para cada sujeto:
  1. Lee n_back_responses (E-Prime) y arma los bloques: grupos de estímulos
     seguidos de la misma tarea (1-back o 3-back) dentro de una sesión.
  2. Pasa el inicio y el fin de cada bloque a segundos desde el inicio de la
     sesión en el EEG (el mismo eje que 'inicio_sesion_s' de ventanas.csv).
  3. Asigna a cada bloque las ventanas de 1 s que caen completas dentro de él
     y que fueron aceptadas en el paso 3.
  4. Calcula la PSD de esas ventanas y la promedia por sesión y tarea.

Guarda en resultados/05_nback/:
  - bloques.csv:          un bloque por fila (sesión, tarea, inicio, fin, aciertos)
  - psd_nback.csv:        PSD promedio por sujeto, sesión, canal y tarea
  - potencia_nback.csv:   potencia por banda, y % de diferencia 3-back vs 1-back
  - psd_nback_<sujeto>.png:    PSD 1-back vs 3-back, un panel por canal y sesión
  - bandas_nback_<sujeto>.png: potencia relativa 1-back vs 3-back por sesión
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import config as cfg

DIR_ARTEFACTOS = cfg.DIR_RESULTADOS / "03_artefactos"
DIR_NBACK = cfg.DIR_RESULTADOS / "05_nback"
DIR_NBACK.mkdir(parents=True, exist_ok=True)

# Nombre de cada sesión en E-Prime (columna Running[Block]) -> nombre en el TP
SESIONES_EPRIME = {
    "NoMusic": "sin_musica",
    "RelaxSession": "musica_relajante",
    "StressSession": "musica_excitante",
    "NewMusicSession": "musica_ia",
}
TAREAS = ["1-back", "3-back"]
COLORES_TAREA = {"1-back": "tab:blue", "3-back": "tab:red"}
DURACION_ESTIMULO = 2.0     # s: letra + cruz de fijación (fin del último estímulo del bloque)
PAUSA_MINIMA = 5.0          # s: si entre dos estímulos pasa más que esto, empieza otro bloque

fs = cfg.FS_FINAL
muestras_por_ventana = int(cfg.DURACION_VENTANA * fs)


def potencia(f, psd, banda):
    """Integral de la PSD entre banda[0] y banda[1] [µV²]."""
    df = f[1] - f[0]
    dentro = (f >= banda[0]) & (f <= banda[1])
    return psd[..., dentro].sum(axis=-1) * df


def juntar(df, objeto, campo):
    """Junta Stimulus100.X ... Stimulus104.X (o Fixation) en una sola columna."""
    columnas = [f"{objeto}{n}.{campo}" for n in range(100, 105) if f"{objeto}{n}.{campo}" in df]
    return df[columnas].bfill(axis=1).iloc[:, 0]


ventanas_info = pd.read_csv(DIR_ARTEFACTOS / "ventanas.csv")
filas_bloques, filas_psd, filas_potencia = [], [], []

for sujeto in cfg.SUJETOS_ACTIVOS:
    print(f"\n{sujeto}:")

    # -----------------------------------------------------------------------
    # 1. Leer E-Prime y quedarse con los estímulos de las 4 sesiones
    # -----------------------------------------------------------------------
    archivos = list((cfg.DIR_DATOS / sujeto).glob("*n_back_responses*.csv"))
    if len(archivos) != 1:
        raise FileNotFoundError(f"{sujeto}: hay {len(archivos)} archivos n_back_responses, se esperaba 1")
    eprime = pd.read_csv(archivos[0])

    eprime["onset_ms"] = juntar(eprime, "Stimulus", "OnsetTime")
    eprime = eprime[eprime["onset_ms"].notna()]                       # solo filas de estímulos
    print("  Sesiones en E-Prime:", eprime["Running[Block]"].unique().tolist())
    print("  Tareas en E-Prime:  ", eprime["Summary"].unique().tolist())

    eprime = eprime[eprime["Running[Block]"].isin(SESIONES_EPRIME)].copy()   # sin la práctica
    eprime["sesion"] = eprime["Running[Block]"].map(SESIONES_EPRIME)
    eprime["tarea"] = eprime["Summary"]
    instruccion = eprime["Instruction"].astype("string").str.lower()
    eprime.loc[eprime["tarea"].isna() & instruccion.str.contains("one back", na=False), "tarea"] = "1-back"
    eprime.loc[eprime["tarea"].isna() & instruccion.str.contains("three back", na=False), "tarea"] = "3-back"
    eprime = eprime[eprime["tarea"].isin(TAREAS)].copy()

    # Acierto: la primera respuesta cuenta. Si no respondió durante la letra,
    # se toma lo que haya respondido durante la cruz de fijación.
    resp_estimulo = juntar(eprime, "Stimulus", "RESP")
    acc_estimulo = juntar(eprime, "Stimulus", "ACC")
    acc_fijacion = juntar(eprime, "Fixation", "ACC")
    eprime["acierto"] = np.where(resp_estimulo.notna(), acc_estimulo, acc_fijacion)

    # Tiempo unix de cada estímulo: inicio del experimento + OnsetTime
    eprime["t_unix"] = eprime["Start_time_unix"] + eprime["onset_ms"] / 1000
    eprime = eprime.sort_values("t_unix")

    tags = np.loadtxt(cfg.archivo_tags(sujeto), ndmin=1)

    for i, sesion in enumerate(cfg.SESIONES):
        inicio_tag = tags[2 * i]
        fin_tag = tags[2 * i + 1]
        est = eprime[eprime["sesion"] == sesion].copy()

        # Control de alineación: la sesión de E-Prime tiene que caer dentro de los tags
        desde_tag = est["t_unix"].iloc[0] - inicio_tag
        hasta_tag = fin_tag - (est["t_unix"].iloc[-1] + DURACION_ESTIMULO)
        print(f"  {sesion}: primer estímulo {desde_tag:+.1f} s después del tag de inicio, "
              f"último termina {hasta_tag:+.1f} s antes del tag de fin")
        if desde_tag < -2 or hasta_tag < -2:
            print("    AVISO: los estímulos caen fuera de los tags. Revisar la alineación.")

        # -------------------------------------------------------------------
        # 2. Armar los bloques: cambia la tarea o hay una pausa larga
        # -------------------------------------------------------------------
        tiempo = est["t_unix"].to_numpy()
        cambio_tarea = est["tarea"].ne(est["tarea"].shift()).to_numpy()
        pausa = np.r_[True, np.diff(tiempo) > PAUSA_MINIMA]
        est["bloque"] = np.cumsum(cambio_tarea | pausa)

        bloques = est.groupby("bloque").agg(
            tarea=("tarea", "first"),
            inicio_unix=("t_unix", "first"),
            ultimo_unix=("t_unix", "last"),
            n_estimulos=("t_unix", "size"),
            aciertos_pct=("acierto", "mean"),
        ).reset_index()
        bloques["inicio_sesion_s"] = bloques["inicio_unix"] - inicio_tag
        bloques["fin_sesion_s"] = bloques["ultimo_unix"] + DURACION_ESTIMULO - inicio_tag
        bloques["aciertos_pct"] *= 100

        conteo = bloques["tarea"].value_counts().to_dict()
        print(f"    bloques: {conteo} | duración media "
              f"{(bloques['fin_sesion_s'] - bloques['inicio_sesion_s']).mean():.1f} s")

        for _, b in bloques.iterrows():
            filas_bloques.append({
                "sujeto": sujeto, "sesion": sesion, "bloque": int(b["bloque"]),
                "tarea": b["tarea"], "inicio_sesion_s": b["inicio_sesion_s"],
                "fin_sesion_s": b["fin_sesion_s"], "n_estimulos": b["n_estimulos"],
                "aciertos_pct": b["aciertos_pct"],
            })

        # -------------------------------------------------------------------
        # 3. Cortar la sesión en ventanas y asignarlas a 1-back o 3-back
        # -------------------------------------------------------------------
        ruta = cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{sesion}.csv"
        x = pd.read_csv(ruta)[cfg.CANALES].to_numpy()
        n_ventanas = len(x) // muestras_por_ventana
        ventanas = x[:n_ventanas * muestras_por_ventana].reshape(
            n_ventanas, muestras_por_ventana, len(cfg.CANALES))

        inicio_v = np.arange(n_ventanas) * cfg.DURACION_VENTANA
        fin_v = inicio_v + cfg.DURACION_VENTANA
        tarea_v = np.full(n_ventanas, "", dtype=object)     # "" = fuera de los bloques
        for _, b in bloques.iterrows():
            dentro = (inicio_v >= b["inicio_sesion_s"]) & (fin_v <= b["fin_sesion_s"])
            tarea_v[dentro] = b["tarea"]

        # -------------------------------------------------------------------
        # 4. PSD promedio de 1-back y de 3-back, por canal
        # -------------------------------------------------------------------
        for canal in cfg.CANALES_ANALISIS:
            j = cfg.CANALES.index(canal)
            info = ventanas_info[(ventanas_info["sujeto"] == sujeto)
                                 & (ventanas_info["sesion"] == sesion)
                                 & (ventanas_info["canal"] == canal)]
            aceptadas = info["aceptada"].to_numpy()

            for tarea in TAREAS:
                elegidas = (tarea_v == tarea) & aceptadas
                n_total = int((tarea_v == tarea).sum())
                if elegidas.sum() == 0:
                    print(f"    {canal} {tarea}: sin ventanas aceptadas")
                    continue

                f, psd_v = signal.welch(ventanas[elegidas, :, j], fs=fs,
                                        nperseg=cfg.NPERSEG_WELCH, axis=-1)
                psd = psd_v.mean(axis=0)
                for frec, valor in zip(f, psd):
                    filas_psd.append({"sujeto": sujeto, "sesion": sesion, "canal": canal,
                                      "tarea": tarea, "frecuencia_Hz": frec, "psd_uV2_Hz": valor})

                total = potencia(f, psd, cfg.BANDA_TOTAL)
                for banda, limites in cfg.BANDAS.items():
                    absoluta = potencia(f, psd, limites)
                    filas_potencia.append({
                        "sujeto": sujeto, "sesion": sesion, "canal": canal,
                        "tarea": tarea, "banda": banda,
                        "n_ventanas_tarea": n_total,
                        "n_ventanas_aceptadas": int(elegidas.sum()),
                        "potencia_abs_uV2": absoluta,
                        "potencia_rel": absoluta / total,
                    })

# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
tabla_bloques = pd.DataFrame(filas_bloques)
tabla_psd = pd.DataFrame(filas_psd)
tabla_potencia = pd.DataFrame(filas_potencia)

# % de diferencia de 3-back respecto de 1-back (misma sesión, canal y banda)
uno = tabla_potencia[tabla_potencia["tarea"] == "1-back"]
uno = uno[["sujeto", "sesion", "canal", "banda", "potencia_rel"]].rename(
    columns={"potencia_rel": "potencia_rel_1back"})
tabla_potencia = tabla_potencia.merge(uno, on=["sujeto", "sesion", "canal", "banda"], how="left")
tabla_potencia["dif_3vs1_pct"] = 100 * (tabla_potencia["potencia_rel"]
                                        / tabla_potencia["potencia_rel_1back"] - 1)
tabla_potencia.loc[tabla_potencia["tarea"] == "1-back", "dif_3vs1_pct"] = np.nan
tabla_potencia = tabla_potencia.drop(columns="potencia_rel_1back")

tabla_bloques.round(3).to_csv(DIR_NBACK / "bloques.csv", index=False)
tabla_psd.round(6).to_csv(DIR_NBACK / "psd_nback.csv", index=False)
tabla_potencia.round(4).to_csv(DIR_NBACK / "potencia_nback.csv", index=False)

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
canales = cfg.CANALES_ANALISIS
for sujeto in cfg.SUJETOS_ACTIVOS:

    # --- Figura 1: PSD 1-back vs 3-back. Filas = canales, columnas = sesiones ---
    fig, axs = plt.subplots(len(canales), len(cfg.SESIONES),
                            figsize=(4.5 * len(cfg.SESIONES), 3.2 * len(canales)),
                            sharex=True, sharey=True, squeeze=False)
    for fila, canal in enumerate(canales):
        for col, sesion in enumerate(cfg.SESIONES):
            ax = axs[fila, col]
            for banda, (lo, hi) in cfg.BANDAS.items():
                ax.axvspan(lo, hi, color="0.9" if banda == "alfa" else "0.96")
            for tarea in TAREAS:
                d = tabla_psd[(tabla_psd["sujeto"] == sujeto) & (tabla_psd["sesion"] == sesion)
                              & (tabla_psd["canal"] == canal) & (tabla_psd["tarea"] == tarea)]
                d = d[(d["frecuencia_Hz"] >= cfg.BANDA_TOTAL[0])
                      & (d["frecuencia_Hz"] <= cfg.BANDA_TOTAL[1])]
                ax.semilogy(d["frecuencia_Hz"], d["psd_uV2_Hz"],
                            color=COLORES_TAREA[tarea], lw=1.3, label=tarea)
            if fila == 0:
                ax.set_title(sesion)
            if col == 0:
                ax.set_ylabel(f"{canal}\nPSD [µV²/Hz]")
            if fila == len(canales) - 1:
                ax.set_xlabel("Frecuencia [Hz]")
            ax.grid(alpha=0.3, which="both")
    axs[0, 0].legend()
    fig.suptitle(f"{sujeto} – PSD promedio de 1-back y 3-back en cada estímulo "
                 f"(bandas sombreadas: theta, alfa, beta)")
    fig.tight_layout()
    fig.savefig(DIR_NBACK / f"psd_nback_{sujeto}.png", dpi=150)
    plt.close(fig)

    # --- Figura 2: potencia relativa. Filas = bandas, columnas = canales ---
    t = tabla_potencia[tabla_potencia["sujeto"] == sujeto]
    posiciones = np.arange(len(cfg.SESIONES))
    fig, axs = plt.subplots(len(cfg.BANDAS), len(canales),
                            figsize=(4.5 * len(canales), 3 * len(cfg.BANDAS)), squeeze=False)
    for fila, banda in enumerate(cfg.BANDAS):
        for col, canal in enumerate(canales):
            ax = axs[fila, col]
            for k, tarea in enumerate(TAREAS):
                d = t[(t["banda"] == banda) & (t["canal"] == canal) & (t["tarea"] == tarea)]
                d = d.set_index("sesion").reindex(cfg.SESIONES)
                barras = ax.bar(posiciones + (k - 0.5) * 0.4, d["potencia_rel"], 0.4,
                                color=COLORES_TAREA[tarea], label=tarea)
                if tarea == "3-back":
                    for barra, dif in zip(barras, d["dif_3vs1_pct"]):
                        if not np.isnan(dif):
                            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                                    f"{dif:+.0f}%", ha="center", va="bottom", fontsize=7)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
            ax.set_xticks(posiciones)
            ax.set_xticklabels(cfg.SESIONES, rotation=20, fontsize=8)
            if fila == 0:
                ax.set_title(canal)
            if col == 0:
                ax.set_ylabel(f"{banda}\npotencia relativa")
            ax.grid(alpha=0.3, axis="y")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle(f"{sujeto} – Potencia relativa en 1-back y 3-back "
                 f"(% = diferencia de 3-back respecto de 1-back)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(DIR_NBACK / f"bandas_nback_{sujeto}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Resumen en pantalla
# ---------------------------------------------------------------------------
print("\nAciertos medios por sesión y tarea [%]:")
print(tabla_bloques.pivot_table(index=["sujeto", "sesion"], columns="tarea",
                                values="aciertos_pct").round(1).to_string())
print("\nDiferencia de 3-back respecto de 1-back en potencia relativa [%]:")
resumen = tabla_potencia[tabla_potencia["tarea"] == "3-back"]
print(resumen.pivot_table(index=["sujeto", "sesion"], columns=["banda", "canal"],
                          values="dif_3vs1_pct").round(1).to_string())