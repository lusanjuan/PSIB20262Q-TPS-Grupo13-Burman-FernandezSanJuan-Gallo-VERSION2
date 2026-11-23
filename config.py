"""
Configuración general del TP.
Dataset: Fekri Azgomi et al. (2023), experimento 1 (Muse, 4 canales).
"""
from pathlib import Path

# --- Rutas ---
RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "dataset A"            # carpetas A1 ... A10 (solo lectura)
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FILTRADO = DIR_RESULTADOS / "01_filtrado"
DIR_SEPARACION = DIR_RESULTADOS / "02_separacion"
DIR_BETA = DIR_RESULTADOS / "03_beta"

# --- Sujetos ---
# A8 queda excluido: su archivo de tags tiene 7 tiempos en vez de 8
# (falta el inicio de la sesión 2), así que no se puede segmentar bien.
EXCLUIDOS = ["A8"]
SUJETOS = [f"A{i}" for i in range(1, 11) if f"A{i}" not in EXCLUIDOS]
SUJETOS_ACTIVOS = SUJETOS
SUJETO_EJEMPLO = SUJETOS

# --- Canales (se descarta 'Right AUX') ---
COL_TIEMPO = "timestamps"
CANALES = ["TP9", "AF7", "AF8", "TP10"]

# --- Frecuencias ---
FS_ORIGINAL = 256      # Hz
FS_FINAL = 128         # Hz (después de submuestrear)
F_LINEA = 60.0         # Hz, ruido de la red eléctrica
F_BAJA = 1.0           # Hz, corte inferior del pasabanda
F_ALTA = 50.0          # Hz, corte superior del pasabanda
ORDEN_FILTRO = 4       # orden del Butterworth (se aplica ida y vuelta)

# --- Sesiones, en el orden en que se hicieron ---
SESIONES = ["sin_musica", "musica_relajante", "musica_excitante", "musica_ia"]
N_TAGS_ESPERADOS = 2 * len(SESIONES)   # inicio y fin de cada sesión
SESION_REFERENCIA = "sin_musica"       # contra la que se calcula el % de cambio

# --- Paso 3: ventanas, rechazo de artefactos y potencia beta ---
DURACION_VENTANA = 1.0     # s, ventanas sin solapamiento
UMBRAL_PP = 150.0          # µV pico a pico; ventanas por encima se descartan
# Solo se analizan los canales frontales: en TP9 y TP10 la amplitud típica
# (mediana ~290 µV pico a pico) supera el umbral y se perdería >80% de las
# ventanas (ver resultados/03_beta/histograma_pico_a_pico.png).
CANALES_ANALISIS = ["AF7", "AF8", "TP9", "TP10"]
CANAL_PSD = "AF7"          # canal de la figura de PSD de A1
NPERSEG_WELCH = 128        # muestras (1 s a 128 Hz) -> resolución de 1 Hz
BANDA_BETA = (13.0, 30.0)  # Hz
BANDA_TOTAL = (1.0, 45.0)  # Hz, denominador de la potencia relativa
BANDAS = {
    "theta": (4.0, 8.0),
    "alfa": (8.0, 13.0),
    "beta": BANDA_BETA,
}


def archivo_eeg(sujeto):
    """Ruta al CSV de EEG del sujeto (ej: dataset A/A1/A1.csv)."""
    return DIR_DATOS / sujeto / f"{sujeto}.csv"


def archivo_tags(sujeto):
    """
    Ruta al archivo de tags del sujeto. Los nombres varían por las descargas
    ('tags.csv', 'tags (1).csv', ...), así que se busca el que empiece con 'tags'.
    """
    candidatos = sorted((DIR_DATOS / sujeto).glob("tags*.csv"))
    if len(candidatos) != 1:
        raise FileNotFoundError(
            f"{sujeto}: se esperaba 1 archivo de tags y hay {len(candidatos)}")
    return candidatos[0]
