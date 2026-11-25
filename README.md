# Música, memoria de trabajo y EEG portátil: reanálisis de Fekri Azgomi et al. (2023)

Trabajo práctico de Procesamiento de Señales Biomédicas, 2.º cuatrimestre 2026.
Grupo 13: Burman, Fernández San Juan, Gallo.

Reanalizamos el experimento 1 de Fekri Azgomi et al. (2023). En ese experimento, 10 participantes hicieron una tarea n-back (1-back y 3-back) en cuatro sesiones: sin música, con música relajante, con música excitante y con música generada por IA. Durante la tarea se registró su EEG con una vincha Muse de 4 canales. El paper concluye que la música aumenta la potencia beta, pero lo muestra en **un solo sujeto (A9)** y con un ANOVA que trata cada ventana de 1 s como una observación independiente.

Este repositorio repite el análisis con **todos los sujetos** y se centra en dos bandas: **beta**, la hipótesis del paper, y **theta**, un marcador de memoria de trabajo. Además separa 1-back de 3-back y evalúa si los cambios en el desempeño se acompañan de cambios en el EEG.

---

## Contenido

- [Datos](#datos)
- [Requisitos](#requisitos)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo correrlo](#cómo-correrlo)
- [Pipeline paso a paso](#pipeline-paso-a-paso)
- [Parámetros principales](#parámetros-principales)
- [Decisiones metodológicas](#decisiones-metodológicas)
- [Interfaz y video](#interfaz-y-video)
- [Resultados principales](#resultados-principales)
- [Limitaciones](#limitaciones)
- [Referencias](#referencias)

---

## Datos

Los datos son públicos y están en PhysioNet:
**Regulation of Brain Cognitive States through Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring** (v1.0.0), https://doi.org/10.13026/jr6f-wc85

Se usa el experimento 1, que está incluido en este repositorio en la carpeta `dataset A/`. Para cada sujeto (`A1` … `A10`) hay tres archivos:

| Archivo | Contenido |
|---|---|
| `A?.csv` | EEG del Muse: TP9, AF7, AF8 y TP10, a 256 Hz, con referencia en Fpz |
| `tags*.csv` | Tiempos unix de inicio y fin de cada sesión (8 valores) |
| `n_back_responses*.csv` | Respuestas de E-Prime: estímulo, tarea, acierto y tiempo |

Los nombres de los archivos de tags y de respuestas varían (`tags.csv`, `tags (2).csv`, …) por cómo se descargaron. `config.py` los busca por el comienzo del nombre.

**A8 se excluye** porque su archivo de tags tiene 7 tiempos en lugar de 8: falta el inicio de la sesión 2 y no se pueden separar sus sesiones. El análisis se hace con **9 sujetos**.

---

## Requisitos

Python 3.10 o superior, y las librerías de `requirements.txt`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` incluye numpy, pandas, scipy, matplotlib y streamlit. Streamlit solo hace falta para la interfaz. Para exportar el video en mp4 (`video.py`) además hace falta **ffmpeg**; si no está instalado, el video se guarda como gif.

---

## Estructura del repositorio

```
.
├── config.py                    # rutas, sujetos, canales y parámetros (único lugar a editar)
├── 1.filtrado.py                # paso 1: pasabanda y submuestreo
├── 2.separacion.py              # paso 2: separación en las 4 sesiones
├── 3.artefactos.py              # paso 3: ventaneo y rechazo por amplitud
├── 4.psd.py                     # paso 4: PSD y potencia por banda en cada sesión
├── 5.comparacion_nback.py       # paso 5: bloques de 1-back y 3-back, PSD por tarea
├── 6.analisis_integral.py       # paso 6: descriptivo del grupo
├── 7.promedios_psd.py           # paso 7: potencia relativa del grupo por sesión
├── 8.estadistica.py             # paso 8: tests estadísticos e informe
├── 9.graficos_estadisticos.py   # paso 9: figuras finales, en el orden del informe
├── interfaz.py                  # interfaz en Streamlit: el procesamiento paso a paso
├── video.py                     # video de una sesión "en vivo" (A9 vs. un sujeto de contraste)
├── requirements.txt
├── dataset A/                   # datos de PhysioNet (experimento 1)
└── resultados/                  # salidas de cada paso, en una carpeta por paso
```

Todos los scripts leen sus parámetros de `config.py` y guardan sus salidas en `resultados/`. No modifican los datos originales. El repositorio ya incluye `resultados/` completo, así que las figuras y tablas se pueden ver sin correr nada.

---

## Cómo correrlo

Desde la carpeta del repositorio, correr los pasos en orden. Los nombres llevan punto, así que conviene escribirlos completos:

```bash
python 1.filtrado.py
python 2.separacion.py
python 3.artefactos.py
python 4.psd.py
python 5.comparacion_nback.py
python 6.analisis_integral.py
python 7.promedios_psd.py
python 8.estadistica.py
python 9.graficos_estadisticos.py
```

El pipeline completo tarda unos 3 minutos con los 9 sujetos. Los pasos 1 a 5 procesan cada sujeto por separado; los pasos 6 a 9 son de grupo y necesitan a todos los sujetos procesados.

Para probar con un solo sujeto, en `config.py`:

```python
SUJETOS_ACTIVOS = ["A1"]       # para volver a todos: SUJETOS_ACTIVOS = SUJETOS
```

y correr los pasos 1 a 5.

Cada script imprime en pantalla lo que va haciendo y avisa si algo no coincide: tags faltantes o desordenados, estímulos que caen fuera de los tags, cantidad de ventanas distinta entre pasos, etc.

---

## Pipeline paso a paso

| Paso | Script | Qué hace | Salida principal |
|---|---|---|---|
| 1 | `1.filtrado.py` | Pasabanda Butterworth de 1–50 Hz (orden 4, ida y vuelta: fase cero) y submuestreo de 256 a 128 Hz. Figuras de control en el tiempo y en frecuencia, para los 4 canales. | `01_filtrado/` |
| 2 | `2.separacion.py` | **Separación** de cada registro en las 4 sesiones, con los tags. Verifica que haya 8 tags, en orden y dentro del registro. | `02_separacion/<sujeto>/` |
| 3 | `3.artefactos.py` | **Ventaneo** en ventanas de 1 s y rechazo de las que tienen un pico a pico mayor que 150 µV (artefacto) o menor que 5 µV (canal plano). Figuras de cada sesión con las ventanas rechazadas coloreadas. | `03_artefactos/ventanas.csv` |
| 4 | `4.psd.py` | PSD de Welch de cada ventana aceptada, promedio por sesión y potencia absoluta y relativa de cada banda. | `04_psd/` |
| 5 | `5.comparacion_nback.py` | Alinea E-Prime con el EEG, arma los bloques de 1-back y 3-back, les asigna las ventanas aceptadas y calcula aciertos y PSD por tarea. | `05_nback/` |
| 6 | `6.analisis_integral.py` | Descriptivo del grupo: efecto de la carga, de la música y su interacción. A9 resaltado en naranja. | `06_grupo/` |
| 7 | `7.promedios_psd.py` | Potencia relativa del grupo por sesión y canal, con un punto por sujeto. | `07_promedios_psd/` |
| 8 | `8.estadistica.py` | Tests estadísticos (ver abajo) e informe en texto. | `08_estadistica/informe.txt` |
| 9 | `9.graficos_estadisticos.py` | Figuras finales, en el orden del relato del informe. | `09_graficos/` |

### Estadística (paso 8)

| Parte | Pregunta | Test |
|---|---|---|
| 1a | ¿El 3-back es más difícil que el 1-back? | Wilcoxon de rangos con signo |
| 1b | ¿La música cambia los aciertos? | ANOVA de medidas repetidas, Friedman y Wilcoxon + Holm |
| 2 | Réplica del método del paper (un sujeto a la vez) | ANOVA de un factor sobre las ventanas de 1 s |
| 3 | ¿La música cambia la potencia de cada banda? (sesión completa) | ANOVA de medidas repetidas, Friedman y Wilcoxon + Holm |
| 3b | Lo mismo, pero solo con los bloques de 3-back | ANOVA de medidas repetidas, Friedman y Wilcoxon + Holm |
| 4a | ¿La potencia cambia entre 3-back y 1-back? | Wilcoxon + Holm |
| 4b | ¿La música modifica esa diferencia? (interacción) | Friedman + Holm |
| 5 | En 3-back: ¿la potencia cambia con la música, y los que más mejoran sus aciertos son los que más cambian el EEG? | Wilcoxon + Holm y correlación de Spearman + Holm |

La Parte 3b existe porque en 1-back los aciertos están cerca del techo: el 3-back es la única condición donde la música cambió el rendimiento y, por lo tanto, donde el EEG debería reflejarlo.

### Figuras finales (paso 9)

| Figura | Qué muestra |
|---|---|
| `fig1_musica_aciertos.png` | La música relajante mejora el desempeño en 3-back |
| `fig2_carga_aciertos.png` | El 3-back es más difícil que el 1-back, y la música solo cambia el 3-back |
| `fig3_eeg_vs_conducta.png` | El desempeño mejora, pero beta y theta no acompañan |
| `fig3b_musica_eeg_3back.png` | Aun usando solo 3-back, beta y theta no cambian con la música |
| `fig4_paper_vs_grupo.png` | El aumento de beta que el paper muestra en A9 no se ve en el grupo |
| `fig5_dos_analisis.png` | Anexo: el mismo dato con el método del paper y con el sujeto como unidad |
| `fig6_correlacion.png` | Anexo: los que más mejoran no son los que más cambian el EEG |

---

## Parámetros principales

Todos están en `config.py`:

| Parámetro | Valor | Motivo |
|---|---|---|
| `FS_ORIGINAL` / `FS_FINAL` | 256 / 128 Hz | Después del pasabanda de 50 Hz, 128 Hz alcanza (Nyquist = 64 Hz) |
| `F_BAJA`, `F_ALTA` | 1–50 Hz | Saca el offset y la deriva lenta, y el ruido de alta frecuencia (incluida la red de 60 Hz) |
| `ORDEN_FILTRO` | 4 | Butterworth: respuesta plana en la banda de paso |
| `CANALES` | TP9, AF7, AF8, TP10 | Los cuatro canales del Muse |
| `DURACION_VENTANA` | 1 s | El EEG se puede considerar aproximadamente estacionario en ventanas de 1 a 4 s |
| `UMBRAL_PP` | 150 µV | Límite superior de amplitud del EEG: por encima hay un parpadeo o un movimiento |
| `UMBRAL_PP_MIN` | 5 µV | Límite inferior: por debajo, el canal está plano (sin contacto) |
| `NPERSEG_WELCH` | 128 | Segmentos de 1 s: resolución de 1 Hz |
| `BANDA_TOTAL` | 1–45 Hz | Denominador de la potencia relativa (ver abajo por qué no hasta 50 Hz) |
| `BANDAS` | theta 4–8, alfa 8–13, beta 13–30 Hz | Bandas clásicas; alfa se calcula pero queda fuera del relato principal |

En los pasos 6 a 9:

| Parámetro | Valor | Motivo |
|---|---|---|
| `MIN_VENTANAS` | 20 | Un valor se usa solo si tiene al menos 20 s de señal limpia |
| `CANALES_POR_BANDA` | theta en AF7 y AF8 | La theta de memoria de trabajo es frontal (hipótesis definida antes de ver los resultados) |
| `ALFA` | 0,05 | Nivel de significación |

> Los decimales se escriben con **punto** (`0.5`, no `0,5`). Con coma, Python interpreta el valor como una tupla y el código falla.

---

## Decisiones metodológicas

- Las ventanas de 1 s de un mismo registro no son independientes. Tratarlas como observaciones separadas (pseudorreplicación) hace que casi cualquier diferencia dé significativa. Por eso cada test de grupo usa un valor por sujeto y condición. El método del paper se replica igual en la Parte 2, para comparar.

-  Al dividir la potencia de cada banda por la potencia total se compara cómo se reparte la potencia entre las bandas, y no cuánta señal llega en total.

- El pasabanda aplicado ida y vuelta ya atenúa cerca del corte: en 45 Hz pasa el 57 % de la potencia y en 50 Hz, el 25 %. Cortar el total en 45 Hz deja afuera la zona más deformada por el filtro. Theta y beta (4–30 Hz) quedan en la zona donde el filtro no modifica la señal.

- Los bloques de 1-back y 3-back se ubican con los tiempos de E-Prime, pasados al reloj del EEG con la hora de inicio del archivo de respuestas. Cada bloque va desde su primer estímulo hasta 2 s después del último (lo que duran la letra y la cruz de fijación). Una ventana pertenece a un bloque solo si cae completa dentro de él; las de instrucciones y descansos no se usan.

- Los cambios se calculan como log10(condición / referencia), que es simétrico: duplicar y reducir a la mitad pesan lo mismo. Para mostrarlos se pasan a %: 100 · (10^x − 1).

- Con 9 sujetos no se puede verificar la normalidad, así que se usan Wilcoxon, Friedman y Spearman. Se agrega el ANOVA de medidas repetidas porque es la versión de grupo del ANOVA del paper.

- Se aplica Holm dentro de cada familia de tests; cada parte del informe indica sobre cuántos tests se corrige.

- Una sesión, canal o tarea sin ventanas suficientes queda como dato faltante (NaN), nunca como 0. Cada promedio usa los sujetos que tienen dato.

-  La theta asociada a memoria de trabajo se genera en la corteza cingulada anterior y se registra en la línea media frontal; AF7 y AF8 son los canales del Muse más cercanos. Beta se analiza en TP9 y TP10 porque son los canales que usa el paper.

---

## Interfaz y video

### `interfaz.py`

Interfaz en Streamlit que sigue un canal de un sujeto por todo el procesamiento, en 8 pasos: señal cruda, filtrado, separación, ventaneo y rechazo, ventanas por tarea, PSD por sesión, PSD por tarea y comparación con el grupo. Se puede elegir el sujeto, el canal y la sesión. No recalcula el pipeline: lee las tablas de `resultados/`.

```bash
streamlit run interfaz.py
```

### `video.py`

Reproduce un tramo de una sesión "en vivo" para A9 y para un sujeto de contraste, que se elige automáticamente como el que cambia beta en dirección opuesta a A9 (con los resultados actuales, A2). Muestra la señal con los parpadeos y las ventanas descartadas, y la beta relativa ventana por ventana. Salida en `resultados/video/`, incluido el último cuadro como imagen por si el video no se puede reproducir.

---

## Resultados principales

El detalle está en `resultados/08_estadistica/informe.txt` y las figuras en `resultados/09_graficos/`.

- **La tarea funciona.** Los aciertos bajan de 93,0 % en 1-back a 81,4 % en 3-back, en los 9 sujetos (Wilcoxon, p = 0,0039).
- **La música cambia el desempeño en 3-back.** Las 4 sesiones difieren (ANOVA de medidas repetidas, p = 0,0003). La música relajante mejora los aciertos en 3,7 puntos (8 de 9 sujetos, p Holm = 0,023). En 1-back no hay cambios porque los sujetos ya están cerca del máximo.
- **El EEG no acompaña esos cambios.** Ni beta ni theta cambian de forma significativa con la música, ni en la sesión completa ni usando solo los bloques de 3-back. Tampoco hay relación entre cuánto mejora cada sujeto y cuánto cambia su EEG.
- **La carga no cambia el EEG de forma significativa.** La potencia no difiere entre 3-back y 1-back en ninguna banda ni canal, y la música no modifica esa diferencia.
- **El resultado del paper no se generaliza.** Con el método del paper (ANOVA sobre ventanas), el 85 % de las comparaciones sujeto × canal × música da "significativo", pero en direcciones opuestas según el sujeto (32 suben y 52 bajan, en beta). El aumento de beta de A9 no se ve en el grupo.

---

## Limitaciones

- **Orden fijo de las sesiones.** Las sesiones se hicieron siempre en el mismo orden, así que la práctica y el cansancio se confunden con el tipo de música.
- **Muestra chica.** Con 9 sujetos, solo se pueden detectar efectos grandes.
- **Canales descartados por completo.** En A4 se descarta el 100 % de las ventanas de AF7, y en A9 el 100 % de AF8 y TP10: esos canales tienen picos a pico de cientos de µV durante todo el registro (mal contacto). Por eso algunos análisis tienen n = 8. En particular, de A9 (el sujeto del paper) solo se analizan TP9 y AF7.
- **Muchas ventanas descartadas en los canales temporales.** En TP9 y TP10 se descarta entre el 11 % y el 73 % de las ventanas según el sujeto, sobre todo por parpadeos: con la referencia en Fpz, el parpadeo aparece entero (e invertido) en los canales temporales.
- **Cuatro canales con referencia en Fpz.** No permiten separar fuentes con ICA ni volver a referenciar de forma útil, y pueden atenuar la theta frontal de línea media.
- **Artefactos en las bandas de interés.** Los parpadeos afectan delta y theta, y la actividad del músculo temporal afecta beta en TP9 y TP10.


---

## Referencias

- Fekri Azgomi, H., Branco, L. R. F., Amin, M. R., Khazaei, S., & Faghih, R. T. (2023). Regulation of brain cognitive states through auditory, gustatory, and olfactory stimulation with wearable monitoring. *Scientific Reports*, 13, 12399.
- Gevins, A., Smith, M. E., McEvoy, L., & Yu, D. (1997). High-resolution EEG mapping of cortical activation related to working memory. *Cerebral Cortex*, 7(4), 374–385.
- Goncharova, I. I., McFarland, D. J., Vaughan, T. M., & Wolpaw, J. R. (2003). EMG contamination of EEG: spectral and topographical characteristics. *Clinical Neurophysiology*, 114(9), 1580–1593.
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187–211.
- Krigolson, O. E., et al. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience*, 11, 109.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.