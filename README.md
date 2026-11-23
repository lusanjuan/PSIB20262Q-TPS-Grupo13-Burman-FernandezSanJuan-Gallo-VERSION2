# VERSION2_TPS_Grupo13# Música, memoria de trabajo y EEG portátil: reanálisis de Fekri Azgomi et al. (2023)

Trabajo práctico de **Procesamiento de Señales e Imágenes Biomédicas (PSIB)**.

Reanalizamos el experimento 1 de Fekri Azgomi et al. (2023). En ese experimento, 10 participantes hicieron una tarea n-back (1-back y 3-back) en cuatro sesiones: sin música, con música relajante, con música excitante y con música generada por IA. Durante la tarea se registró su EEG con una vincha **Muse** de 4 canales. El paper concluye que la música aumenta la potencia beta, pero lo muestra en **un solo sujeto (A9)** y con un ANOVA que trata cada ventana de 1 s como una observación independiente.

Este repositorio repite el análisis con **todos los sujetos** y toma al **sujeto como unidad de análisis**. Agrega la banda theta como marcador de memoria de trabajo, separa 1-back de 3-back y evalúa si los cambios en el desempeño se acompañan de cambios en el EEG.

---

## Contenido

- [Datos](#datos)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo correrlo](#cómo-correrlo)
- [Pipeline paso a paso](#pipeline-paso-a-paso)
- [Parámetros principales](#parámetros-principales)
- [Decisiones metodológicas](#decisiones-metodológicas)
- [Análisis complementarios](#análisis-complementarios)
- [Resultados principales](#resultados-principales)
- [Limitaciones](#limitaciones)
- [Referencias](#referencias)

---

## Datos

Los datos son públicos y están en PhysioNet:
**Regulation of Brain Cognitive States through Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring** (v1.0.0), https://doi.org/10.13026/jr6f-wc85

Para cada sujeto (`A1` … `A10`) se usan tres archivos:

| Archivo | Contenido |
|---|---|
| `A?.csv` | EEG del Muse: TP9, AF7, AF8 y TP10, a 256 Hz, con referencia en Fpz |
| `tags*.csv` | Tiempos unix de inicio y fin de cada sesión (8 valores) |
| `*n_back_responses*.csv` | Respuestas de E-Prime: estímulo, tarea, acierto y tiempo |



**A8 se excluye** porque su archivo de tags tiene 7 tiempos en lugar de 8, así que no se pueden separar sus sesiones. El análisis se hace con 9 sujetos.

---

## Requisitos e instalación

Python 3.10 o superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install numpy pandas scipy matplotlib
pip install streamlit      # solo para la interfaz (app.py)
```

Para exportar el video en mp4 (`video_sesion.py`) hace falta tener **ffmpeg** instalado. Si no está, el video se guarda como gif.

---

## Estructura del repositorio

```
.
├── config.py                    # rutas, sujetos, canales y parámetros (único lugar a editar)
├── 1_filtrado.py
├── 2_separacion.py            # separación por condiciones experimentales
├── 3_artefactos.py
├── 4_psd.py
├── 5_nback.py
├── 6_grupo.py
├── 7_promedios_psd.py
├── 8_estadistica.py
├── 10_graficos_estadistica.py
├── analisis_parpadeos.py        # análisis complementario
├── dataset A/                   # datos (no incluidos)
└── resultados/                  # se genera al correr los scripts
```

Todos los scripts leen sus parámetros de `config.py` y guardan sus salidas en `resultados/`, en una carpeta por paso. No modifican los datos originales.

---

## Cómo correrlo

**1. Probar con un sujeto.** En `config.py`:

```python
SUJETOS_ACTIVOS = ["A1"]
```

Después corran los pasos en orden:

```bash
python 1_filtrado.py
python 2_segmentacion.py
python 3_artefactos.py
python 4_psd.py
python 5_nback.py
```

**2. Correr todos los sujetos.** En `config.py`:

```python
SUJETOS_ACTIVOS = SUJETOS
```

y repetir los pasos 1 a 5. Los pasos 6 a 10 son de grupo, así que necesitan a todos los sujetos procesados:

```bash
python 6_grupo.py
python 7_promedios_psd.py
python 8_estadistica.py
python 9_validacion_A9.py
python 10_graficos_estadistica.py
```

Cada script imprime en pantalla lo que va haciendo y avisa si algo no coincide: tags faltantes, sesiones de distinta duración, estímulos que caen fuera de los tags, etc.

---

## Pipeline paso a paso

| Paso | Script | Qué hace | Salida principal |
|---|---|---|---|
| 1 | `1_filtrado.py` | Notch de 60 Hz, pasa-banda Butterworth de 1–50 Hz (orden 4, fase cero) y submuestreo de 256 a 128 Hz | `01_filtrado/` |
| 2 | `2_segmentacion.py` | **Separación por condiciones experimentales**: corta cada registro en las 4 sesiones usando los tags | `02_segmentacion/<sujeto>/` |
| 3 | `3_artefactos.py` | **Ventaneo** en ventanas de 1 s y rechazo de las que superan 150 µV de pico a pico | `03_artefactos/ventanas.csv` |
| 4 | `4_psd.py` | PSD de Welch por ventana aceptada, promedio por sesión y potencia absoluta y relativa en theta, alfa y beta | `04_psd/` |
| 5 | `5_nback.py` | Alinea E-Prime con el EEG, arma los bloques de 1-back y 3-back y calcula aciertos y PSD por tarea | `05_nback/` |
| 6 | `6_grupo.py` | Estadística descriptiva del grupo: efecto de la carga, de la música y su interacción | `06_grupo/` |
| 7 | `7_promedios_psd.py` | Figuras de potencia relativa por estímulo, con un punto por sujeto | `07_promedios_psd/` |
| 8 | `8_estadistica.py` | Tests estadísticos (ver abajo) e informe en texto | `08_estadistica/informe.txt` |
| 9 | `9_validacion_A9.py` | Compara nuestro procesamiento de A9 con los valores del paper | `09_validacion_A9/` |
| 10 | `10_graficos_estadistica.py` | Figuras finales, en el orden del relato del informe | `10_graficos/` |

### Estadística (paso 8)

| Parte | Pregunta | Test |
|---|---|---|
| 1a | ¿El 3-back es más difícil que el 1-back? | Wilcoxon de rangos con signo |
| 1b | ¿La música cambia los aciertos? | ANOVA de medidas repetidas, Friedman y Wilcoxon + Holm |
| 2 | Réplica del método del paper (un sujeto a la vez) | ANOVA de un factor sobre las ventanas |
| 3 | ¿La música cambia la potencia de cada banda? | ANOVA de medidas repetidas, Friedman y Wilcoxon + Holm |
| 4a | ¿La potencia cambia entre 3-back y 1-back? | Wilcoxon + Holm |
| 4b | ¿La música modifica esa diferencia? (interacción) | Wilcoxon + Holm |
| 5 | ¿Los sujetos que más cambian sus aciertos son los que más cambian el EEG? | Correlación de Spearman |

---

## Parámetros principales

Todos están en `config.py`:

| Parámetro | Valor | Motivo |
|---|---|---|
| `FS_ORIGINAL` / `FS_FINAL` | 256 / 128 Hz | Después del pasa-banda de 50 Hz, 128 Hz alcanza (Nyquist = 64 Hz) |
| `F_LINEA` | 60 Hz | Red eléctrica de EE. UU., donde se registraron los datos |
| `F_BAJA`, `F_ALTA` | 1–50 Hz | Saca la deriva lenta y el ruido de alta frecuencia |
| `CANALES` | TP9, AF7, AF8, TP10 | Los cuatro canales del Muse |
| `DURACION_VENTANA` | 1 s | El EEG se puede considerar aproximadamente estacionario en ventanas de 1 a 4 s |
| `UMBRAL_PP` | 150 µV | Límite superior de amplitud del EEG (5–150 µV) |
| `NPERSEG_WELCH` | 128 | Segmentos de 1 s, resolución de 1 Hz |
| `BANDA_TOTAL` | 1–45 Hz | Denominador de la potencia relativa |
| `BANDAS` | theta 4–8, alfa 8–13, beta 13–30 Hz | Bandas clásicas |

En `8_estadistica.py`:

| Parámetro | Valor | Motivo |
|---|---|---|
| `MIN_VENTANAS` | 20 | Un valor se usa solo si tiene al menos 20 s de señal limpia |
| `CANALES_POR_BANDA` | theta en AF7 y AF8 | La theta de memoria de trabajo es frontal (hipótesis definida antes de ver los resultados) |
| `ALFA` | 0,05 | Nivel de significación |

> Los decimales se escriben con **punto** (`0.5`, no `0,5`). Con coma, Python interpreta el valor como una tupla y el código falla.

---

## Decisiones metodológicas

**El sujeto como unidad de análisis.** Las ventanas de 1 s de un mismo registro no son independientes. Tratarlas como observaciones separadas (pseudorreplicación) hace que casi cualquier diferencia dé significativa. Por eso cada test de grupo usa un valor por sujeto y condición. El método del paper se replica igual en la Parte 2, para comparar.

**Cambio en escala logarítmica.** Los cambios se calculan como log10(condición / referencia), que es simétrico: duplicar y reducir a la mitad pesan lo mismo. Para mostrarlos se pasan a %: 100 · (10^x − 1).

**Tests no paramétricos.** Con 9 sujetos no se puede verificar la normalidad, así que se usan Wilcoxon, Friedman y Spearman. Se agrega el ANOVA de medidas repetidas porque es la versión de grupo del ANOVA del paper.

**Corrección por comparaciones múltiples.** Se aplica Holm dentro de cada familia de tests.

**Datos faltantes.** Una sesión, canal o tarea sin ventanas suficientes queda como dato faltante (NaN), nunca como 0. Cada promedio usa los sujetos que tienen dato.

**Theta en los canales frontales.** La theta asociada a memoria de trabajo se genera en la corteza cingulada anterior y se registra en la línea media frontal (Fz). AF7 y AF8 son los canales del Muse más cercanos. Beta se analiza también en TP9 y TP10 porque son los canales que usa el paper.

---

## Análisis complementarios

Ninguno de estos scripts modifica el pipeline principal.

### `analisis_parpadeos.py`

Detecta los parpadeos como picos negativos simultáneos en TP9 y TP10: con la referencia en Fpz, el parpadeo aparece invertido en los canales temporales. Calcula la tasa de parpadeo por sujeto, sesión y tarea. Además compara cuatro formas de elegir la señal limpia:

| Opción | Criterio |
|---|---|
| A | La actual: ventanas de 1 s y umbral de 150 µV |
| B | Ventanas solapadas al 50 % |
| C | B + descarte de toda ventana que toque un parpadeo detectado |
| D | Borrar los parpadeos y los picos de más de ±100 µV, y unir los tramos restantes |

Salidas en `resultados/parpadeos/`.

### `video_sesion.py`

Reproduce un tramo de una sesión "en tiempo real" para A9 y un sujeto de contraste, que se elige automáticamente como el que cambia beta en dirección opuesta a A9. Muestra la señal con los parpadeos y las ventanas descartadas, y la beta relativa ventana por ventana. Salida en `resultados/video/`.

### `app.py`

Interfaz en Streamlit para explorar los resultados. Tiene tres pestañas:
- **Sujeto:** señal, PSD y aciertos.
- **Pseudorreplicación:** cómo cambia el p del método del paper según la cantidad de ventanas.
- **Robustez:** la Parte 3 del paso 8 con otros umbrales y otras medidas de potencia.

```bash
streamlit run app.py
```

### Cambiar el largo de la ventana

Para repetir el análisis con ventanas de otro largo, sin pisar los resultados de 1 s, cambien en `config.py`:

```python
DURACION_VENTANA = 2.0          # o 0.5
NPERSEG_WELCH = 128             # con 0.5 s tiene que ser 64
DIR_RESULTADOS = RAIZ / "resultados_2s"
```

Además, ajusten `MIN_VENTANAS` para mantener 20 s de señal (10 ventanas con 2 s, 40 con 0,5 s). Con ventanas de 0,5 s la resolución es de 2 Hz, demasiado gruesa para separar theta de alfa: esa versión sirve solo para mostrar el efecto de la pseudorreplicación.

---

## Resultados principales

- **La tarea funciona.** En los 9 sujetos, el desempeño en 3-back es peor que en 1-back.
- **La música cambia el desempeño en 3-back.** La relajante lo mejora de forma significativa, la excitante lo baja (no significativo) y la de IA lo sube (tendencia). En 1-back casi no hay cambios porque los sujetos ya están cerca del máximo.
- **El EEG no acompaña esos cambios.** En el grupo, ni beta ni theta cambian de forma significativa con la música, y los sujetos que más mejoran no son los que más cambian el EEG.
- **El resultado del paper no se generaliza.** Con el método del paper, la mayoría de los sujetos dan "significativo", pero en direcciones opuestas. El aumento de beta de A9 no se ve en el grupo, y su signo depende de decisiones del análisis como el largo de la ventana.
- **Las conclusiones son robustas al rechazo de artefactos.** Los parpadeos son el artefacto principal (entre 7 y 51 por minuto según el sujeto). Con las cuatro formas de limpieza probadas, la conclusión sobre la música no cambia.

El detalle está en `resultados/08_estadistica/informe.txt` y las figuras en `resultados/10_graficos/`.

---

## Limitaciones

- **Orden fijo de las sesiones.** Las sesiones se hicieron siempre en el mismo orden, así que la práctica y el cansancio se confunden con el tipo de música.
- **Muestra chica.** Con 9 sujetos, solo se pueden detectar efectos grandes.
- **Cuatro canales con referencia en Fpz.** No permiten ICA ni volver a referenciar de forma útil, y pueden atenuar la theta frontal de línea media.
- **Artefactos en las bandas de interés.** Los parpadeos afectan delta y theta en los canales frontales, y el músculo temporal afecta beta en TP9 y TP10.
- **Canales sin contacto.** El umbral actual no descarta ventanas planas (pico a pico cercano a 0 µV). En A9 se observaron ventanas así en TP10. Se recomienda agregar un umbral inferior de 5 µV.

---

## Referencias

- Fekri Azgomi, H., Branco, L. R. F., Amin, M. R., Khazaei, S., & Faghih, R. T. (2023). Regulation of brain cognitive states through auditory, gustatory, and olfactory stimulation with wearable monitoring. *Scientific Reports*, 13, 12399.
- Gevins, A., Smith, M. E., McEvoy, L., & Yu, D. (1997). High-resolution EEG mapping of cortical activation related to working memory. *Cerebral Cortex*, 7(4), 374–385.
- Goncharova, I. I., McFarland, D. J., Vaughan, T. M., & Wolpaw, J. R. (2003). EMG contamination of EEG: spectral and topographical characteristics. *Clinical Neurophysiology*, 114(9), 1580–1593.
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187–211.
- Krigolson, O. E., et al. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience*, 11, 109.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.