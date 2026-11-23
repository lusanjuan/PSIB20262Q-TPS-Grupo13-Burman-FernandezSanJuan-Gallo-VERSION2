"""
Interfaz para explorar los resultados del TP (dataset A).

Uso (desde la carpeta del TP, después de correr los pasos 1 a 5 con todos los sujetos):
    pip install streamlit
    streamlit run interfaz.py

Tiene tres pestañas:
  1. Sujeto:              señal de una sesión con las ventanas descartadas, PSD y aciertos.
  2. Pseudorreplicación:  cómo el p del método del paper depende de la cantidad de ventanas,
                          comparado con el test entre sujetos.
  3. Robustez:            repite el análisis de grupo (Parte 3 del paso 8) cambiando el
                          umbral de artefactos, la medida de potencia y los canales.

No calcula nada desde cero: lee los archivos de resultados/02_segmentacion,
03_artefactos, 04_psd y 05_nback.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

import config as cfg

DIR_ARTEFACTOS = cfg.DIR_RESULTADOS / "03_artefactos"
DIR_PSD = cfg.DIR_RESULTADOS / "04_psd"
DIR_NBACK = cfg.DIR_RESULTADOS / "05_nback"

REF = cfg.SESION_REFERENCIA
MUSICAS = [s for s in cfg.SESIONES if s != REF]
FS = cfg.FS_FINAL
MIN_VENTANAS = 20          # igual que en el paso 8
ALFA = 0.05


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
@st.cache_data
def leer(ruta):
    return pd.read_csv(ruta)


@st.cache_data
def leer_sesion(sujeto, sesion):
    i = cfg.SESIONES.index(sesion)
    ruta = cfg.DIR_SEPARACION / sujeto / f"{sujeto}_{i + 1}_{sesion}.csv"
    return pd.read_csv(ruta)[cfg.CANALES]


def wilcoxon(valores):
    """Wilcoxon de rangos con signo contra 0 (igual que en el paso 8)."""
    valores = pd.Series(valores).dropna()
    if len(valores) < 5 or (valores == 0).all():
        return np.nan
    return stats.wilcoxon(valores).pvalue


def holm(p):
    """Corrección de Holm (igual que en el paso 8)."""
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


def texto_p(p):
    if np.isnan(p):
        return "no se puede calcular"
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TP EEG - dataset A", layout="wide")
st.title("TP EEG - dataset A")

archivos = {
    "ventanas": DIR_ARTEFACTOS / "ventanas.csv",
    "potencia_ventanas": DIR_PSD / "potencia_ventanas.csv",
    "potencia_bandas": DIR_PSD / "potencia_bandas.csv",
    "psd": DIR_PSD / "psd_promedio.csv",
    "bloques": DIR_NBACK / "bloques.csv",
}
faltan = [str(r) for r in archivos.values() if not r.exists()]
if faltan:
    st.error("Faltan archivos. Correr antes los pasos 1 a 5:\n\n" + "\n\n".join(faltan))
    st.stop()

ventanas = leer(archivos["ventanas"])
pot_v = leer(archivos["potencia_ventanas"])
pot_b = leer(archivos["potencia_bandas"])
psd = leer(archivos["psd"])
bloques = leer(archivos["bloques"])

sujetos = [s for s in cfg.SUJETOS if s in set(ventanas["sujeto"])]
canales_psd = [c for c in cfg.CANALES if c in set(pot_v["canal"])]
bandas = list(cfg.BANDAS)

st.write(f"Sujetos con datos: {', '.join(sujetos)}. "
         f"Canales con PSD: {', '.join(canales_psd)}. "
         f"Sesión de referencia: {REF}.")

tab1, tab2, tab3 = st.tabs(["1. Sujeto", "2. Pseudorreplicación", "3. Robustez"])


# ===========================================================================
# 1. SUJETO
# ===========================================================================
with tab1:
    c1, c2, c3 = st.columns(3)
    sujeto = c1.selectbox("Sujeto", sujetos, key="t1_sujeto")
    sesion = c2.selectbox("Sesión", cfg.SESIONES, key="t1_sesion")
    canal = c3.selectbox("Canal", cfg.CANALES, key="t1_canal")

    # --- Señal con las ventanas descartadas ---
    x = leer_sesion(sujeto, sesion)[canal].to_numpy()
    t = np.arange(len(x)) / FS
    info = ventanas[(ventanas["sujeto"] == sujeto) & (ventanas["sesion"] == sesion)
                    & (ventanas["canal"] == canal)]
    descartadas = info[~info["aceptada"]]

    duracion = int(t[-1])
    tramo = st.slider("Tramo a mostrar [s]", 0, duracion, (0, min(60, duracion)), key="t1_tramo")
    limitar = st.checkbox("Limitar el eje vertical a ±200 µV", value=True, key="t1_limitar")

    en_tramo = (t >= tramo[0]) & (t <= tramo[1])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t[en_tramo], x[en_tramo], color="black", lw=0.5)
    for _, v in descartadas.iterrows():
        if v["fin_sesion_s"] >= tramo[0] and v["inicio_sesion_s"] <= tramo[1]:
            ax.axvspan(v["inicio_sesion_s"], v["fin_sesion_s"], color="red", alpha=0.2, lw=0)
    if limitar:
        ax.set_ylim(-200, 200)
    ax.set_xlim(tramo)
    ax.set_xlabel("Tiempo desde el inicio de la sesión [s]")
    ax.set_ylabel("µV")
    ax.set_title(f"{sujeto} - {sesion} - {canal}. En rojo: ventanas con pico a pico "
                 f"> {cfg.UMBRAL_PP:g} µV")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Ventanas descartadas en esta sesión y canal: {len(descartadas)} de {len(info)} "
             f"({100 * len(descartadas) / max(len(info), 1):.1f} %).")

    c1, c2 = st.columns(2)

    # --- PSD de la sesión y de la referencia ---
    with c1:
        st.subheader("PSD")
        if canal not in set(psd["canal"]):
            st.write(f"El paso 4 no calculó la PSD de {canal} (no está en CANALES_ANALISIS).")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            for s, color in [(REF, "gray"), (sesion, "black")]:
                d = psd[(psd["sujeto"] == sujeto) & (psd["sesion"] == s) & (psd["canal"] == canal)]
                d = d[(d["frecuencia_Hz"] >= cfg.BANDA_TOTAL[0])
                      & (d["frecuencia_Hz"] <= cfg.BANDA_TOTAL[1])]
                ax.semilogy(d["frecuencia_Hz"], d["psd_uV2_Hz"], color=color, label=s)
                if s == sesion:
                    break                       # si la sesión es la referencia, una sola curva
            for lo, hi in cfg.BANDAS.values():
                ax.axvline(lo, color="gray", ls=":", lw=0.8)
                ax.axvline(hi, color="gray", ls=":", lw=0.8)
            ax.set_xlabel("Frecuencia [Hz]")
            ax.set_ylabel("PSD [µV²/Hz]")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            tabla = pot_b[(pot_b["sujeto"] == sujeto) & (pot_b["canal"] == canal)]
            tabla = tabla.pivot_table(index="sesion", columns="banda", values="cambio_pct")
            tabla = tabla.reindex(index=cfg.SESIONES, columns=bandas)
            st.write(f"Cambio de la potencia relativa respecto de {REF} [%]:")
            st.dataframe(tabla.round(1))

    # --- Aciertos ---
    with c2:
        st.subheader("Aciertos [%]")
        d = bloques[bloques["sujeto"] == sujeto]
        tabla = d.pivot_table(index="sesion", columns="tarea", values="aciertos_pct")
        st.dataframe(tabla.reindex(cfg.SESIONES).round(1))
        st.write("Promedio de los bloques de cada sesión y tarea.")


# ===========================================================================
# 2. PSEUDORREPLICACIÓN
# ===========================================================================
with tab2:
    st.write(
        "El paper compara dos sesiones de un sujeto con un ANOVA donde cada ventana de 1 s "
        "es una observación. Como las ventanas seguidas no son independientes, el p se achica "
        "solo con agregar ventanas, aunque el cambio sea el mismo. A la derecha está el test "
        "que corresponde para hablar del grupo: un valor por sujeto.")

    c1, c2, c3, c4 = st.columns(4)
    sujeto = c1.selectbox("Sujeto", sujetos, key="t2_sujeto")
    canal = c2.selectbox("Canal", canales_psd, key="t2_canal")
    sesion = c3.selectbox("Sesión", MUSICAS, key="t2_sesion")
    banda = c4.selectbox("Banda", bandas, index=bandas.index("beta"), key="t2_banda")

    d = pot_v[(pot_v["canal"] == canal) & (pot_v["banda"] == banda)]

    izq, der = st.columns(2)

    # --- Un sujeto: p del ANOVA según la cantidad de ventanas ---
    with izq:
        st.subheader(f"Método del paper: {sujeto}")
        ds = d[d["sujeto"] == sujeto]
        a = ds.loc[ds["sesion"] == REF, "potencia_rel"].to_numpy()
        b = ds.loc[ds["sesion"] == sesion, "potencia_rel"].to_numpy()
        n_max = min(len(a), len(b))

        if n_max < 10:
            st.write("Este sujeto tiene menos de 10 ventanas aceptadas en alguna de las dos sesiones.")
        else:
            # Las ventanas se toman en un orden al azar (fijo), para que usar pocas
            # ventanas no signifique usar solo el principio de la sesión.
            rng = np.random.default_rng(0)
            a = rng.permutation(a)
            b = rng.permutation(b)

            n = st.slider("Ventanas por sesión", 10, n_max, n_max, key="t2_n")
            p = stats.f_oneway(a[:n], b[:n]).pvalue
            cambio = 100 * (b[:n].mean() / a[:n].mean() - 1)

            ns = np.unique(np.linspace(10, n_max, 40).astype(int))
            ps = [stats.f_oneway(a[:k], b[:k]).pvalue for k in ns]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(ns, ps, color="black", marker="o", ms=3)
            ax.axhline(ALFA, color="red", ls="--", lw=1, label="p = 0.05")
            ax.axvline(n, color="gray", lw=1)
            ax.set_xlabel("Ventanas por sesión")
            ax.set_ylabel("p del ANOVA (escala log)")
            ax.set_title(f"{banda} {canal}: {sesion} vs {REF}")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.write(f"Con {n} ventanas por sesión: cambio de {cambio:+.1f} %, {texto_p(p)}.")

    # --- Grupo: un valor por sujeto ---
    with der:
        st.subheader("Grupo: un valor por sujeto")
        g = pot_b[(pot_b["canal"] == canal) & (pot_b["banda"] == banda)
                  & (pot_b["n_ventanas_aceptadas"] >= MIN_VENTANAS)]
        m = g.pivot_table(index="sujeto", columns="sesion", values="potencia_rel")
        if REF not in m or sesion not in m:
            st.write("No hay datos suficientes.")
        else:
            log_cambio = np.log10(m[sesion] / m[REF]).dropna()
            pct = 100 * (10 ** log_cambio - 1)
            p_grupo = wilcoxon(log_cambio)

            orden = [s for s in sujetos if s in pct.index]
            fig, ax = plt.subplots(figsize=(6, 4))
            colores = ["red" if s == sujeto else "black" for s in orden]
            ax.scatter(range(len(orden)), pct[orden], color=colores, zorder=3)
            ax.axhline(0, color="gray", lw=1)
            ax.set_xticks(range(len(orden)), orden)
            ax.set_ylabel(f"Cambio de {banda} relativa [%]")
            ax.set_title(f"{sesion} vs {REF} (en rojo, {sujeto})")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.write(f"{(pct > 0).sum()} sujetos suben y {(pct < 0).sum()} bajan. "
                     f"Wilcoxon entre sujetos (n = {len(pct)}): {texto_p(p_grupo)}.")

            # El método del paper aplicado a cada sujeto, con todas sus ventanas
            filas = []
            for s in orden:
                ds = d[d["sujeto"] == s]
                a_s = ds.loc[ds["sesion"] == REF, "potencia_rel"].to_numpy()
                b_s = ds.loc[ds["sesion"] == sesion, "potencia_rel"].to_numpy()
                if min(len(a_s), len(b_s)) < MIN_VENTANAS:
                    continue
                filas.append({"sujeto": s,
                              "cambio_pct": 100 * (b_s.mean() / a_s.mean() - 1),
                              "p_anova": stats.f_oneway(a_s, b_s).pvalue})
            tabla = pd.DataFrame(filas)
            if len(tabla):
                sig = tabla["p_anova"] < ALFA
                sube = int((sig & (tabla["cambio_pct"] > 0)).sum())
                baja = int((sig & (tabla["cambio_pct"] < 0)).sum())
                st.write(f"Con el método del paper (todas las ventanas), {sig.sum()} de "
                         f"{len(tabla)} sujetos dan significativo: {sube} suben y {baja} bajan.")
                st.dataframe(tabla.set_index("sujeto").round(4))


# ===========================================================================
# 3. ROBUSTEZ
# ===========================================================================
with tab3:
    st.write(
        "Repite la Parte 3 del paso 8 (cada música contra la referencia, un valor por sujeto, "
        "Wilcoxon y Holm sobre las 3 músicas de cada canal) con otras opciones de análisis. "
        f"Con umbral {cfg.UMBRAL_PP:g} µV y potencia relativa da lo mismo que el paso 8. "
        f"El umbral no puede subir por encima de {cfg.UMBRAL_PP:g} µV porque el paso 4 "
        "solo calculó la PSD de las ventanas aceptadas con ese valor.")

    c1, c2, c3, c4 = st.columns(4)
    umbral = c1.slider("Umbral pico a pico [µV]", 50, int(cfg.UMBRAL_PP), int(cfg.UMBRAL_PP),
                       step=10, key="t3_umbral")
    medida = c2.radio("Potencia", ["relativa", "absoluta"], key="t3_medida")
    banda = c3.selectbox("Banda", bandas, index=bandas.index("beta"), key="t3_banda")
    elegidos = c4.multiselect("Canales", canales_psd, default=canales_psd, key="t3_canales")

    # Pico a pico de cada ventana, para volver a aplicar el umbral
    pp = ventanas[["sujeto", "sesion", "canal", "ventana", "pico_a_pico_uV"]]
    d = pot_v[(pot_v["banda"] == banda) & pot_v["canal"].isin(elegidos)]
    d = d.merge(pp, on=["sujeto", "sesion", "canal", "ventana"])
    d = d[d["pico_a_pico_uV"] <= umbral].copy()

    # Potencia de la sesión = promedio de las ventanas. La relativa se arma como
    # promedio de la banda / promedio del total, igual que en el paso 4
    # (total de cada ventana = absoluta / relativa).
    d["total"] = d["potencia_abs_uV2"] / d["potencia_rel"]
    ses = d.groupby(["sujeto", "sesion", "canal"]).agg(
        banda_abs=("potencia_abs_uV2", "mean"), total=("total", "mean"),
        n=("potencia_abs_uV2", "size")).reset_index()
    ses["valor"] = ses["banda_abs"] if medida == "absoluta" else ses["banda_abs"] / ses["total"]
    ses.loc[ses["n"] < MIN_VENTANAS, "valor"] = np.nan

    filas = []
    for canal in elegidos:
        m = ses[ses["canal"] == canal].pivot_table(index="sujeto", columns="sesion", values="valor")
        m = m.reindex(columns=cfg.SESIONES)
        log_cambio = np.log10(m[MUSICAS].div(m[REF], axis=0))
        p_wil = [wilcoxon(log_cambio[s]) for s in MUSICAS]
        for s, pw, ph in zip(MUSICAS, p_wil, holm(p_wil)):
            v = log_cambio[s].dropna()
            filas.append({"canal": canal, "sesion": s, "n_sujetos": len(v),
                          "cambio_pct": 100 * (10 ** v.mean() - 1),
                          "suben": int((v > 0).sum()), "bajan": int((v < 0).sum()),
                          "p_wilcoxon": pw, "p_holm": ph})
    resultado = pd.DataFrame(filas)

    descarte = 100 * (1 - (ventanas["pico_a_pico_uV"] <= umbral).mean())
    st.write(f"Con {umbral} µV se descarta el {descarte:.1f} % de todas las ventanas "
             f"(todos los sujetos, sesiones y canales).")

    if len(resultado) == 0:
        st.write("Elegir al menos un canal.")
    else:
        st.dataframe(resultado.round(4), hide_index=True)
        signif = resultado[resultado["p_holm"] < ALFA]
        if len(signif):
            st.write("Significativos después de Holm: " + "; ".join(
                f"{f['canal']} {f['sesion']} ({f['cambio_pct']:+.1f} %)"
                for _, f in signif.iterrows()) + ".")
        else:
            st.write("Ningún cambio es significativo después de Holm.")

        # Un punto por sujeto y la media, un panel por canal
        fig, axs = plt.subplots(1, len(elegidos), figsize=(5 * len(elegidos), 4),
                                sharey=True, squeeze=False)
        for ax, canal in zip(axs[0], elegidos):
            m = ses[ses["canal"] == canal].pivot_table(index="sujeto", columns="sesion",
                                                       values="valor")
            for i, s in enumerate(MUSICAS):
                if s not in m or REF not in m:
                    continue
                v = np.log10(m[s] / m[REF]).dropna()
                pct = 100 * (10 ** v - 1)
                ax.scatter(np.full(len(pct), i), pct, color="gray", s=15)
                ax.scatter(i, 100 * (10 ** v.mean() - 1), color="black", marker="_", s=400)
            ax.axhline(0, color="gray", lw=1)
            ax.set_xticks(range(len(MUSICAS)), MUSICAS, rotation=20)
            ax.set_title(canal)
        axs[0, 0].set_ylabel(f"Cambio de {banda} {medida} vs {REF} [%]")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write("Gris: cada sujeto. Raya negra: media del grupo.")