"""
Paso 8: análisis estadístico.

Orden del análisis (el mismo que conviene usar en el informe):

  PARTE 1 – Conducta
      1a. ¿El 3-back es más difícil que el 1-back?   -> control de que la tarea funcionó
      1b. ¿La música cambia los aciertos?            -> lo que el paper reporta como mejora
  PARTE 2 – Réplica del paper (un sujeto a la vez)
      ANOVA de un factor con las ventanas de 1 s como observaciones, en beta.
  PARTE 3 – Música y EEG en el grupo
      ¿La potencia de cada banda cambia con la música de forma consistente entre sujetos?
  PARTE 4 – Carga cognitiva y EEG en el grupo
      4a. ¿La potencia cambia en 3-back respecto de 1-back?
      4b. ¿La música modifica esa diferencia? (interacción)
  PARTE 5 – EEG y conducta
      En los bloques de 3-back: ¿la potencia cambia con la música, y los sujetos que
      más mejoran sus aciertos son los que más cambian la potencia?

Criterios comunes:
  - Unidad de análisis: el SUJETO (un valor por sujeto y condición), salvo en la
    Parte 2, que replica el método del paper.
  - Datos faltantes: los valores con menos de MIN_VENTANAS ventanas aceptadas se
    dejan afuera (NaN), nunca se cuentan como 0.
  - Cambio en el EEG: log10(condición / referencia), que es simétrico; se muestra
    en % como 100 * (10**x - 1). En la conducta, diferencia en puntos porcentuales.
  - Tests no paramétricos (pocos sujetos): Wilcoxon de rangos con signo,
    Friedman y Spearman. En la Parte 3 y en la conducta se agrega el ANOVA de
    medidas repetidas, que es la versión de grupo del ANOVA del paper.
  - Canales por banda (CANALES_POR_BANDA): theta solo en los frontales, por hipótesis.
  - Comparaciones múltiples: corrección de Holm dentro de cada familia de tests
    (se indica en cada parte sobre cuántos tests se corrige).

Guarda en resultados/08_estadistica/:
  1a_conducta_carga.csv, 1b_conducta_musica.csv,
  2_replica_por_sujeto.csv, 2_replica_resumen.csv,
  3_musica_eeg.csv, 4a_carga_eeg.csv, 4b_interaccion.csv,
  5_eeg_y_aciertos.csv, informe.txt (todo lo impreso en pantalla)
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg

# ===========================================================================
# Configuración
# ===========================================================================
DIR_PSD = cfg.DIR_RESULTADOS / "04_psd"
DIR_NBACK = cfg.DIR_RESULTADOS / "05_nback"
DIR_EST = cfg.DIR_RESULTADOS / "08_estadistica"
DIR_EST.mkdir(parents=True, exist_ok=True)

ALFA = 0.05                  # nivel de significación
MIN_VENTANAS = 20            # mínimo de ventanas aceptadas para usar un valor
BANDA_PAPER = "beta"         # banda que analiza el paper (Parte 2)

# Canales que se analizan para cada banda en las Partes 3, 4 y 5. Si una banda no
# está en el diccionario, se usan todos los de CANALES_ANALISIS.
# Theta solo en los frontales: la theta de memoria de trabajo es frontal
# (hipótesis definida a partir de la literatura, antes de mirar los resultados).
CANALES_POR_BANDA = {"theta": ["AF7", "AF8"]}

canales = cfg.CANALES_ANALISIS
bandas = list(cfg.BANDAS)
musicas = [s for s in cfg.SESIONES if s != cfg.SESION_REFERENCIA]
REF = cfg.SESION_REFERENCIA
pd.set_option("display.width", 200)
informe = []


# ===========================================================================
# Funciones auxiliares
# ===========================================================================
def decir(texto=""):
    """Imprime y guarda la línea para el informe."""
    print(texto)
    informe.append(str(texto))


def titulo(texto):
    decir("\n" + "=" * 80)
    decir(texto)
    decir("=" * 80)


def subtitulo(texto):
    decir("\n" + texto)
    decir("-" * len(texto))


def canales_de(banda):
    return [c for c in CANALES_POR_BANDA.get(banda, canales) if c in canales]


def a_pct(log_cambio):
    """log10(cociente) -> % de cambio."""
    return 100 * (10 ** log_cambio - 1)


def wilcoxon(valores):
    """Wilcoxon de rangos con signo contra 0. NaN si hay menos de 5 sujetos."""
    valores = pd.Series(valores).dropna()
    if len(valores) < 5 or (valores == 0).all():
        return np.nan
    return stats.wilcoxon(valores).pvalue


def friedman(matriz):
    """Friedman entre las columnas, con los sujetos completos. Devuelve (p, n)."""
    completa = matriz.dropna()
    if len(completa) < 5:
        return np.nan, len(completa)
    return stats.friedmanchisquare(*[completa[c] for c in completa]).pvalue, len(completa)


def anova_medidas_repetidas(matriz):
    """
    ANOVA de un factor con medidas repetidas (sujetos x condiciones, sin NaN).
        SS_total     = suma de (y - media_general)^2
        SS_condicion = n * suma de (media_condicion - media_general)^2
        SS_sujeto    = k * suma de (media_sujeto - media_general)^2
        SS_error     = SS_total - SS_condicion - SS_sujeto
        F = (SS_condicion / (k-1)) / (SS_error / ((n-1)(k-1)))
    Devuelve (F, p). NaN si hay menos de 3 sujetos completos.
    """
    y = matriz.dropna().to_numpy()
    if len(y) < 3:
        return np.nan, np.nan
    n, k = y.shape
    media = y.mean()
    ss_total = ((y - media) ** 2).sum()
    ss_cond = n * ((y.mean(axis=0) - media) ** 2).sum()
    ss_suj = k * ((y.mean(axis=1) - media) ** 2).sum()
    ss_error = ss_total - ss_cond - ss_suj
    gl_cond, gl_error = k - 1, (n - 1) * (k - 1)
    F = (ss_cond / gl_cond) / (ss_error / gl_error)
    return F, stats.f.sf(F, gl_cond, gl_error)


def holm(p):
    """
    Corrección de Holm: ordena los p de menor a mayor y multiplica el primero
    por m, el segundo por m-1, etc. (m = cantidad de tests). Un p corregido
    nunca queda menor que el anterior, y se topea en 1.
    """
    p = np.asarray(p, dtype=float)
    validos = ~np.isnan(p)
    m = validos.sum()
    corregido = np.full_like(p, np.nan)
    orden = np.argsort(np.where(validos, p, np.inf))[:m]
    maximo = 0.0
    for i, k in enumerate(orden):
        maximo = max(maximo, (m - i) * p[k])
        corregido[k] = min(1.0, maximo)
    return corregido


def contar(v):
    """Cuántos sujetos suben y cuántos bajan."""
    v = pd.Series(v).dropna()
    return int((v > 0).sum()), int((v < 0).sum())


# ===========================================================================
# Carga de datos
# ===========================================================================
bloques = pd.read_csv(DIR_NBACK / "bloques.csv")

ventanas = pd.read_csv(DIR_PSD / "potencia_ventanas.csv")
ventanas = ventanas[(ventanas["banda"] == BANDA_PAPER) & ventanas["canal"].isin(canales)]

potencia = pd.read_csv(DIR_PSD / "potencia_bandas.csv")
potencia = potencia[potencia["canal"].isin(canales)
                    & (potencia["n_ventanas_aceptadas"] >= MIN_VENTANAS)]

nback = pd.read_csv(DIR_NBACK / "potencia_nback.csv")
nback = nback[nback["canal"].isin(canales)].copy()
nback.loc[nback["n_ventanas_aceptadas"] < MIN_VENTANAS, "potencia_rel"] = np.nan
# Tabla ancha: columnas 1-back y 3-back, y el log del cociente
nback_ancha = nback.pivot_table(index=["sujeto", "sesion", "canal", "banda"], columns="tarea",
                                values="potencia_rel", dropna=False).reset_index()
nback_ancha["log_3vs1"] = np.log10(nback_ancha["3-back"] / nback_ancha["1-back"])

decir(f"Sujetos: {sorted(bloques['sujeto'].unique(), key=lambda s: int(s[1:]))}")
decir(f"Canales por banda: " + ", ".join(f"{b}: {canales_de(b)}" for b in bandas))
decir(f"Nivel de significación: {ALFA}. Mínimo de ventanas aceptadas: {MIN_VENTANAS}.")


# ===========================================================================
# PARTE 1 – CONDUCTA
# ===========================================================================
titulo("PARTE 1 – Conducta")

# --- 1a. ¿El 3-back es más difícil? --------------------------------------
subtitulo("1a. ¿El 3-back es más difícil que el 1-back? (Wilcoxon, aciertos por sujeto)")
acc_tarea = bloques.groupby(["sujeto", "tarea"])["aciertos_pct"].mean().unstack()
dif = acc_tarea["3-back"] - acc_tarea["1-back"]
p_1a = wilcoxon(dif)
mejoran, empeoran = contar(dif)
P1a = pd.DataFrame([{
    "aciertos_1back": acc_tarea["1-back"].mean(), "aciertos_3back": acc_tarea["3-back"].mean(),
    "diferencia_pp": dif.mean(), "sujetos_peor_en_3back": empeoran,
    "n_sujetos": dif.notna().sum(), "p_wilcoxon": p_1a,
}])
P1a.round(4).to_csv(DIR_EST / "1a_conducta_carga.csv", index=False)
decir(f"Aciertos: 1-back {acc_tarea['1-back'].mean():.1f}%, 3-back {acc_tarea['3-back'].mean():.1f}% "
      f"({dif.mean():+.1f} puntos). {empeoran} de {dif.notna().sum()} sujetos aciertan menos "
      f"en 3-back. p = {p_1a:.4f}")
decir("  -> El 3-back es más difícil: la manipulación de carga funcionó." if p_1a < ALFA else
      "  -> No hay diferencia clara: interpretar con cuidado la Parte 4.")

# --- 1b. ¿La música cambia los aciertos? -----------------------------------
subtitulo("1b. ¿La música cambia los aciertos? (por tarea; Holm sobre 3 tests por tarea)")
acc_ses = bloques.groupby(["sujeto", "tarea", "sesion"])["aciertos_pct"].mean().reset_index()
filas = []
dif_aciertos_3back = pd.DataFrame()        # se usa en la Parte 5
for tarea in ["1-back", "3-back"]:
    m = acc_ses[acc_ses["tarea"] == tarea].pivot_table(index="sujeto", columns="sesion",
                                                       values="aciertos_pct")
    m = m.reindex(columns=cfg.SESIONES)
    F, p_rm = anova_medidas_repetidas(m)
    p_fr, n_fr = friedman(m)
    dif = m[musicas].sub(m[REF], axis=0)             # puntos porcentuales vs sin_musica
    if tarea == "3-back":
        dif_aciertos_3back = dif
    p_wil = [wilcoxon(dif[s]) for s in musicas]
    for s, pw, ph in zip(musicas, p_wil, holm(p_wil)):
        sube, baja = contar(dif[s])
        filas.append({"tarea": tarea, "sesion": s, "n_sujetos": dif[s].notna().sum(),
                      "aciertos_ref": m[REF].mean(), "aciertos_sesion": m[s].mean(),
                      "diferencia_pp": dif[s].mean(), "mejoran": sube, "empeoran": baja,
                      "p_wilcoxon": pw, "p_holm": ph,
                      "p_anova_rm_4_sesiones": p_rm, "p_friedman_4_sesiones": p_fr})
    decir(f"\n{tarea}: " + ", ".join(f"{s} {m[s].mean():.1f}%" for s in cfg.SESIONES))
    decir(f"  ¿Difieren las 4 sesiones? ANOVA de medidas repetidas p = {p_rm:.4f}, "
          f"Friedman p = {p_fr:.4f} (n = {n_fr})")
P1b = pd.DataFrame(filas)
P1b.round(4).to_csv(DIR_EST / "1b_conducta_musica.csv", index=False)
decir("")
decir(P1b[["tarea", "sesion", "n_sujetos", "diferencia_pp", "mejoran", "empeoran", "p_holm"]]
      .round(3).to_string(index=False))
for _, f in P1b[P1b["p_holm"] < ALFA].iterrows():
    decir(f"  -> {f['tarea']}, {f['sesion']}: {f['diferencia_pp']:+.1f} puntos "
          f"({f['mejoran']} mejoran; p Holm = {f['p_holm']:.3f}).")
if not (P1b["p_holm"] < ALFA).any():
    decir("  -> Ningún cambio de aciertos con la música es significativo.")
decir("  Ojo: el orden de las sesiones fue siempre el mismo; una mejora puede deberse a práctica.")


# ===========================================================================
# PARTE 2 – RÉPLICA DEL PAPER
# ===========================================================================
titulo(f"PARTE 2 – Réplica del paper: ANOVA dentro de cada sujeto ({BANDA_PAPER}, ventanas de 1 s)")

filas = []
for (sujeto, canal), d in ventanas.groupby(["sujeto", "canal"]):
    grupos = {s: d.loc[d["sesion"] == s, "potencia_rel"].to_numpy() for s in cfg.SESIONES}
    if any(len(g) < MIN_VENTANAS for g in grupos.values()):
        continue
    for s in musicas:
        filas.append({"sujeto": sujeto, "canal": canal, "sesion": s,
                      "n_ventanas_ref": len(grupos[REF]), "n_ventanas_sesion": len(grupos[s]),
                      "cambio_pct": 100 * (grupos[s].mean() / grupos[REF].mean() - 1),
                      "p_anova": stats.f_oneway(grupos[REF], grupos[s]).pvalue})
P2 = pd.DataFrame(filas)
P2["significativo"] = P2["p_anova"] < ALFA
P2.round(5).to_csv(DIR_EST / "2_replica_por_sujeto.csv", index=False)

P2_res = P2.groupby(["canal", "sesion"]).apply(lambda d: pd.Series({
    "n_sujetos": len(d),
    "suben_signif": int(((d["cambio_pct"] > 0) & d["significativo"]).sum()),
    "bajan_signif": int(((d["cambio_pct"] < 0) & d["significativo"]).sum()),
    "no_signif": int((~d["significativo"]).sum()),
}), include_groups=False).reset_index()
P2_res.to_csv(DIR_EST / "2_replica_resumen.csv", index=False)

decir(f"\nComparaciones sesión vs {REF}: {len(P2)}. Significativas: {100 * P2['significativo'].mean():.0f}%.")
decir(P2_res.to_string(index=False))
decir("  Ojo: con cientos de ventanas casi cualquier diferencia da significativa, y las ventanas")
decir("  seguidas no son independientes (pseudorreplicación). Las conclusiones de grupo salen")
decir("  de las Partes 3 a 5.")


# ===========================================================================
# PARTE 3 – MÚSICA Y EEG EN EL GRUPO
# ===========================================================================
titulo(f"PARTE 3 – ¿La música cambia la potencia? (cada sesión vs {REF}, sesión completa)")
decir("Por canal y banda: ANOVA de medidas repetidas y Friedman entre las 4 sesiones;")
decir("Wilcoxon de cada música vs sin_musica (Holm sobre 3 tests por canal y banda).")

filas = []
for banda in bandas:
    for canal in canales_de(banda):
        d = potencia[(potencia["canal"] == canal) & (potencia["banda"] == banda)]
        m = d.pivot_table(index="sujeto", columns="sesion", values="potencia_rel")
        m = m.reindex(columns=cfg.SESIONES)
        F, p_rm = anova_medidas_repetidas(m)
        p_fr, n_fr = friedman(m)
        log_cambio = np.log10(m[musicas].div(m[REF], axis=0))
        p_wil = [wilcoxon(log_cambio[s]) for s in musicas]
        for s, pw, ph in zip(musicas, p_wil, holm(p_wil)):
            sube, baja = contar(log_cambio[s])
            filas.append({"banda": banda, "canal": canal, "sesion": s,
                          "n_sujetos": log_cambio[s].notna().sum(),
                          "cambio_pct": a_pct(log_cambio[s].mean()), "suben": sube, "bajan": baja,
                          "p_wilcoxon": pw, "p_holm": ph,
                          "p_anova_rm_4_sesiones": p_rm, "p_friedman_4_sesiones": p_fr,
                          "n_sujetos_completos": n_fr})
P3 = pd.DataFrame(filas)
P3.round(5).to_csv(DIR_EST / "3_musica_eeg.csv", index=False)

for banda in bandas:
    t = P3[P3["banda"] == banda]
    subtitulo(f"{banda} ({', '.join(canales_de(banda))})")
    decir(t[["canal", "sesion", "n_sujetos", "cambio_pct", "suben", "bajan", "p_holm",
             "p_anova_rm_4_sesiones", "p_friedman_4_sesiones"]].round(3).to_string(index=False))
    for _, f in t[t["p_holm"] < ALFA].iterrows():
        decir(f"  -> {f['canal']} {f['sesion']}: {f['cambio_pct']:+.1f}% "
              f"({f['suben']} suben, {f['bajan']} bajan; p Holm = {f['p_holm']:.3f}).")
    if not (t["p_holm"] < ALFA).any():
        decir(f"  -> Ningún cambio de {banda} con la música es significativo.")


# ===========================================================================
# PARTE 4 – CARGA COGNITIVA Y EEG EN EL GRUPO
# ===========================================================================
titulo("PARTE 4 – Carga cognitiva: 3-back vs 1-back")

# --- 4a. Efecto de la carga -------------------------------------------------
subtitulo("4a. ¿La potencia cambia en 3-back? (Wilcoxon contra 0; Holm sobre los canales de cada banda)")
carga = nback_ancha.groupby(["sujeto", "canal", "banda"])["log_3vs1"].mean().reset_index()
filas = []
for banda in bandas:
    p_banda = []
    for canal in canales_de(banda):
        v = carga[(carga["canal"] == canal) & (carga["banda"] == banda)]["log_3vs1"]
        sube, baja = contar(v)
        p = wilcoxon(v)
        p_banda.append(p)
        filas.append({"banda": banda, "canal": canal, "n_sujetos": v.notna().sum(),
                      "cambio_pct": a_pct(v.mean()), "suben": sube, "bajan": baja,
                      "p_wilcoxon": p})
    for fila, ph in zip(filas[-len(p_banda):], holm(p_banda)):
        fila["p_holm"] = ph
P4a = pd.DataFrame(filas)
P4a.round(5).to_csv(DIR_EST / "4a_carga_eeg.csv", index=False)
decir(P4a.round(3).to_string(index=False))
for _, f in P4a[P4a["p_holm"] < ALFA].iterrows():
    decir(f"  -> {f['canal']} {f['banda']}: {f['cambio_pct']:+.1f}% en 3-back "
          f"({f['suben']} suben, {f['bajan']} bajan; p Holm = {f['p_holm']:.3f}).")
if not (P4a["p_holm"] < ALFA).any():
    decir("  -> Ningún efecto de la carga es significativo.")

# --- 4b. Interacción ----------------------------------------------------------
subtitulo("4b. ¿La música modifica el efecto de la carga? (Friedman entre sesiones; Holm por banda)")
filas = []
for banda in bandas:
    p_banda = []
    for canal in canales_de(banda):
        d = nback_ancha[(nback_ancha["canal"] == canal) & (nback_ancha["banda"] == banda)]
        m = d.pivot_table(index="sujeto", columns="sesion", values="log_3vs1")
        m = m.reindex(columns=cfg.SESIONES)
        p, n = friedman(m)
        p_banda.append(p)
        fila = {"banda": banda, "canal": canal, "n_sujetos_completos": n, "p_friedman": p}
        for s in cfg.SESIONES:
            fila[f"carga_{s}_pct"] = a_pct(m[s].mean())
        filas.append(fila)
    for fila, ph in zip(filas[-len(p_banda):], holm(p_banda)):
        fila["p_holm"] = ph
P4b = pd.DataFrame(filas)
P4b.round(4).to_csv(DIR_EST / "4b_interaccion.csv", index=False)
decir(P4b[["banda", "canal", "n_sujetos_completos", "p_friedman", "p_holm"]].round(3).to_string(index=False))
for _, f in P4b[P4b["p_holm"] < ALFA].iterrows():
    decir(f"  -> {f['canal']} {f['banda']}: el efecto de la carga depende de la música.")
if not (P4b["p_holm"] < ALFA).any():
    decir("  -> No hay evidencia de que la música modifique el efecto de la carga.")


# ===========================================================================
# PARTE 5 – EEG Y CONDUCTA EN 3-BACK
# ===========================================================================
titulo("PARTE 5 – ¿El EEG acompaña la mejora de los aciertos en 3-back?")
decir("Solo bloques de 3-back. Para cada banda, canal y música:")
decir("  (1) ¿la potencia cambia respecto de sin_musica? (Wilcoxon)")
decir("  (2) ¿los sujetos que más mejoran sus aciertos son los que más cambian la potencia? (Spearman)")
decir("Holm sobre (canales de la banda) x 3 músicas, por separado para cada tipo de test.")

filas = []
for banda in bandas:
    inicio = len(filas)
    for canal in canales_de(banda):
        d = nback[(nback["tarea"] == "3-back") & (nback["banda"] == banda) & (nback["canal"] == canal)]
        m = d.pivot_table(index="sujeto", columns="sesion", values="potencia_rel")
        m = m.reindex(columns=cfg.SESIONES)
        log_cambio = np.log10(m[musicas].div(m[REF], axis=0))
        for s in musicas:
            juntos = pd.concat([log_cambio[s].rename("eeg"),
                                dif_aciertos_3back[s].rename("aciertos")], axis=1).dropna()
            rho, p_rho = (stats.spearmanr(juntos["eeg"], juntos["aciertos"])
                          if len(juntos) >= 5 else (np.nan, np.nan))
            sube, baja = contar(log_cambio[s])
            filas.append({"banda": banda, "canal": canal, "sesion": s,
                          "n_sujetos": log_cambio[s].notna().sum(),
                          "cambio_pct": a_pct(log_cambio[s].mean()), "suben": sube, "bajan": baja,
                          "p_wilcoxon": wilcoxon(log_cambio[s]),
                          "rho_spearman": rho, "p_spearman": p_rho})
    bloque = filas[inicio:]
    for fila, ph in zip(bloque, holm([f["p_wilcoxon"] for f in bloque])):
        fila["p_wilcoxon_holm"] = ph
    for fila, ph in zip(bloque, holm([f["p_spearman"] for f in bloque])):
        fila["p_spearman_holm"] = ph
P5 = pd.DataFrame(filas)
P5.round(5).to_csv(DIR_EST / "5_eeg_y_aciertos.csv", index=False)

for banda in bandas:
    t = P5[P5["banda"] == banda]
    subtitulo(f"{banda} en 3-back ({', '.join(canales_de(banda))}; Holm sobre {len(t)} tests)")
    decir(t[["canal", "sesion", "n_sujetos", "cambio_pct", "suben", "bajan", "p_wilcoxon_holm",
             "rho_spearman", "p_spearman_holm"]].round(3).to_string(index=False))
    hay = False
    for _, f in t.iterrows():
        if f["p_wilcoxon_holm"] < ALFA:
            hay = True
            decir(f"  -> {f['canal']} {f['sesion']}: {banda} cambia {f['cambio_pct']:+.1f}% "
                  f"(p Holm = {f['p_wilcoxon_holm']:.3f}).")
        if f["p_spearman_holm"] < ALFA:
            hay = True
            decir(f"  -> {f['canal']} {f['sesion']}: el cambio de {banda} se relaciona con el cambio "
                  f"de aciertos (rho = {f['rho_spearman']:+.2f}, p Holm = {f['p_spearman_holm']:.3f}).")
    if not hay:
        decir(f"  -> {banda}: ni cambios consistentes ni relación con los aciertos.")


(DIR_EST / "informe.txt").write_text("\n".join(informe), encoding="utf-8")
print(f"\nInforme guardado en {DIR_EST / 'informe.txt'}")