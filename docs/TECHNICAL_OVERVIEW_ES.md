# Chess Alpha — Memoria técnica integral

**Estado documental:** reconstrucción histórica a partir de los hitos y registros disponibles hasta septiembre de 2026. Este texto explica decisiones, implementaciones, resultados, incidencias y líneas de trabajo. **No reemplaza una auditoría del repositorio local** ni certifica que todas las ramas experimentales se hayan fusionado.

## 1. Visión y evolución del proyecto

Chess Alpha se planteó como un entorno de investigación para desarrollar un motor de ajedrez combinando dos enfoques: **modelos aprendidos de los datos** (policy/value y predicción de secuencias) y **búsqueda explícita de decisiones** (árbol legal con poda, transposiciones, extensiones tácticas y controles de finalización). No se trataba únicamente de predecir el siguiente movimiento humano. El proyecto buscaba explorar si una política neuronal podía orientar decisiones sin sacrificar corrección táctica, y construir la infraestructura necesaria para medirlo.

La evolución registrada separó varias líneas: (1) Chess Alpha v1.1 y elección de checkpoints policy/value; (2) `sequence_policy_v1` con transformador causal; (3) `search_tree_v1` y sucesivas fases de búsqueda; (4) `tokenization_v1` para codificar millones de partidas; (5) Golden Dataset y enriquecimiento de posiciones V3; (6) diseños de variables search-aware e integración futura. También se discutieron enfoques tipo AlphaZero, MCTS y self-play, **sin evidencia de un bucle completo de entrenamiento self-play aprobado**.

## 2. Dataset PGN de origen

Se inspeccionó el archivo `AJ-OTB-PGN-000.pgn`, en codificación UTF-8-SIG, con **4.471.784 partidas**. Incluía encabezados como `Event`, `Site`, `Date`, `Round`, `White`, `Black`, `Result`, `ECO`, `WhiteElo`, `BlackElo` y `PlyCount`. Algunas partidas carecían de Elo; otras eran no utilizables por resultado `*`, ausencia de jugadas, variante no admitida, partida vacía o repetición.

Se optó por el PGN anterior frente a un volcado de Lichess de unos 201 GB, mucho más costoso de manejar en el entorno disponible. La entrada original se mantuvo en **solo lectura**, sin hacer una pasada previa de deduplicación sobre todos los millones de registros. La muestra se seleccionaba de forma determinista y se reemplazaban candidatos inválidos hasta completar el cupo de partidas válidas y únicas.

La identidad de una partida se definió mediante hash de **FEN inicial normalizado + variante + secuencia UCI íntegra**. Esto evita tratar como idénticas dos secuencias que tengan un significado distinto por comenzar en posiciones diferentes. La semilla establecida fue **20260803**.

### Resultados del piloto

| Indicador | Resultado |
|---|---:|
| Corrida | `test_10000` |
| Partidas válidas y únicas | 10.000 |
| Candidatos examinados | 10.042 |
| Rechazos | 42 |
| Partidas vacías | 26 |
| Duplicadas | 14 |
| Sin resultado | 1 |
| Variante no soportada | 1 |
| Plies tokenizados | 808.611 |
| Media de plies/partida | 80,9 |
| Mediana | 78 |
| Percentil 90 | 123 |
| Máximo | 331 |
| Rendimiento registrado | 92,85 partidas/s, con dos workers |
| Memoria agregada máxima | ~551 MB |
| Suite de tokenizador inicial | 25/25 PASS |

Se ejecutó un **test real de interrupción alrededor de las 2.000 partidas**, comprobando persistencia del checkpoint, reanudación por `--resume`, continuidad de `draw_rank` y `sample_id` (0–9999), hashes únicos y ausencia de reprocesado de partidas consolidadas. Un primer mecanismo de vigilancia de la interrupción fue desechado por reaccionar tarde; la repetición con un vigilante persistente validó el comportamiento correcto.

La documentación posterior registra un dataset principal de **1.000.000 de partidas** y aproximadamente **81.244.568 plies**. Para una publicación definitiva, el `manifest.json` de esa corrida debe utilizarse como prueba primaria de las cifras finales.

## 3. Arquitectura de tokenización y seguridad operativa

La implementación se organizó de forma aislada en `tokenization_v1/` con scripts de inspección, indexado de partidas, muestreo, tokenización, ejecución y generación de reportes. Las carpetas de salida separaban `outputs`, `shards`, `checkpoints`, `logs`, `reports`, `config` y `tests`. No era necesario copiar el PGN original a la carpeta del proyecto.

La ejecución del piloto limitó `n_workers = 2` y usó `multiprocessing` con `spawn` en Windows. Los workers reconstruían tableros, verificaban jugadas, generaban tokens y calculaban hashes; el proceso principal coordinaba offsets, ordenaba resultados por `draw_rank`, consolidaba shards y escribía checkpoints.

Se utilizó escritura atómica (**temporal → flush/cierre → `os.replace`**) para no marcar como completada una partida que todavía no estuviera consolidada en disco. El sistema contemplaba cierre seguro por `Ctrl+C` o señal `STOP_REQUEST` y reanudación por bandera `--resume`. En la configuración previa a la expansión se establecieron umbrales de seguridad como RAM agregada de 6 GB, mínimo de 20 GB disponibles y máximo de dos workers; estos umbrales son **históricos de esa tarea**, no parámetros universales del motor.

El proyecto compartía máquina con una fase activa de Stockfish/`search_tree_v1`. Por eso, la tokenización fue **aislada** para no interferir ni modificar archivos, procesos o checkpoints de esa otra ejecución. Se documentó además un warning de compatibilidad entre `numexpr`, NumPy 2.x, pandas y PyArrow, que no presentó corrupción observada y se dejó sin cambios para proteger el entorno.

### Codificación de movimientos

La fórmula compacta fue:

```text
action_id = (promo_code << 12) | (from_square << 6) | to_square
```

- `from_square` y `to_square`: valores 0–63.
- `promo_code`: 0 sin promoción, 1 caballo, 2 alfil, 3 torre, 4 dama.
- Espacio de IDs: `5 × 64 × 64 = 20.480`.
- Tipo binario: `uint16`.

El esquema cubre movimientos ordinarios, capturas, enroques, en passant y todas las promociones, con o sin captura. **El ID no es suficiente por sí solo para determinar la legalidad o semántica especial**: para ello se utiliza el estado del tablero. Las pruebas incluyeron ida/vuelta `move → id → move`, FEN, enroques de ambos bandos, en passant y partidas reconstruidas íntegramente. La línea de Sequence Policy registra **20.483 tokens de vocabulario** al incluir símbolos especiales de secuencia.

## 4. Golden Dataset: muestra de referencia congelada

Se creó y congeló la muestra de **10.000 partidas** denominada `chess_alpha_tokenization_golden_test10000_v1` (versión documental `v1.0.0`). Sus **808.611 plies** dan lugar a **818.611 filas** en el recuento de posiciones enriquecidas, al sumarse **10.000 filas adicionales** asociadas a las partidas. El dataset fue particionado en **9 shards**.

Se reportaron validaciones `quick/full`, pruebas de integridad y verificaciones de hashes SHA-256; la muestra quedó protegida como **inmutable**. Los registros mencionan comprobaciones de hashes sobre base y motor, y un paquete de **74 archivos** verificados en una revisión posterior. Los recuentos exactos por componente deben comprobarse contra el manifiesto específico antes de presentarlos como un único total homogéneo.

El propósito de congelar una muestra no era mejorar artificialmente el entrenamiento: era garantizar un punto de comparación estable para transformaciones, variables, joins y futuras evaluaciones.

## 5. Ingeniería de variables: Tokenization Enrichment V3

El enriquecimiento de posiciones amplió la representación de los tokens de movimiento con señales útiles para evaluación y búsqueda. La ejecución registrada alcanzó:

| Control | Resultado |
|---|---:|
| Familias construidas | 28/28 |
| Filas/posiciones | 818.611 |
| Shards consolidados | 9/9 |
| Integridad de joins `row_id` | 100% |
| Pruebas focalizadas | 30/30 PASS |
| Salidas | 81 archivos aproximadamente |
| Tamaño agregado | ~298,2 MB |
| Golden de origen | Preservado byte a byte |

El full-run permitió detectar un bug de **incompatibilidad de tipos de datos en bitboards** durante la consolidación de un shard. Se corrigió y se añadió un test de regresión, sin reescribir el Golden. El diseño distinguía deliberadamente tres conceptos: **identidad del movimiento tokenizado**, **variables descriptivas de posición** y **etiquetas/señales observadas por el algoritmo de búsqueda**.

No se dispone en esta reconstrucción de un diccionario exhaustivo confiable con los nombres y definiciones de las 28 familias; sería incorrecto inventarlo. Para publicar ese nivel de detalle deben extraerse automáticamente de los scripts, reportes y `data dictionary` reales.

## 6. Sequence Policy v1: Transformer de movimientos

En paralelo al evaluador de tablero, se implementó una línea de entrenamiento autoregresivo: predecir el siguiente token de movimiento dada una ventana previa de la partida.

### Componentes de ingeniería

- Almacenamiento compacto de tokens en `moves_uint16.bin`.
- Ventanas de contexto y targets con índices de inicio/fin y protección frente a duplicación de targets.
- `windows_train.parquet`, `windows_dev.parquet` y ventanas TEST congeladas.
- `SequenceWindowDataset`, `ExtendedBatchCollator`, pérdidas enmascaradas mediante `ignore_index` y evaluación Top-K/MRR.
- Checkpoints recuperables, hashes para confirmar integridad, scripts de entrenamiento y bundles de Colab.
- Corrección y pruebas de un fallo de acumulación de gradientes; ronda registrada de **113 PASS, 1 skipped**.

### Configuración de red registrada

| Parámetro | Valor |
|---|---:|
| Tipo | Causal Move Transformer |
| Capas | 8 |
| `d_model` | 384 |
| Attention heads | 6 |
| FFN hidden | 1.536 |
| Longitud de contexto | 256 |
| Vocabulario | 20.483 |
| Positional embedding | RoPE |

Estos parámetros describen el **experimento Sequence Policy**, no todos los modelos de Chess Alpha. El entrenamiento FULL recuperado se cerró en `global_step = 2286`, `tokens_seen = 74.536.785`.

Se verificó un checkpoint denominado `last.ckpt` con SHA-256:

```text
29f96d5a5343931e4cab71c6affa7628ab97240459a6792d7314f51901644804
```

Durante una etapa intermedia, una ejecución alcanzó el step 1600 y se observaron métricas de DEV `top1=0,29794`, `top10=0,69095` y `MRR=0,42985`; **ese checkpoint step 1600 no estaba inicialmente recuperable en Drive**. La recuperación posterior del checkpoint FULL step 2286 no debe confundirse con esos resultados intermedios.

La evaluación de TEST congelado usó **50.016 ventanas** y **4.118.551 targets** y verificó los hashes del checkpoint y del test al comenzar. El reporte final `sequence_policy_full_test_2286.json` (`evaluation_type = FROZEN_TEST_FINAL`, lectura única, checkpoint fijo y sin tuning iterativo) se localizó y se revisó en la auditoría de publicación (2026-11-20). Sus valores: **top-1 0,30098**, top-3 0,49442, top-5 0,58261, **top-10 0,70065**, top-20 0,81434, **MRR 0,43436**, perplejidad 19,65, ECE 0,0139 y rango mediano 4. En DEV (50.444 ventanas) se obtuvo top-1 0,3007, top-10 0,69986 y MRR 0,43383. Que ambos resultados sean prácticamente idénticos no sugiere sobreajuste. Por tramo de partida, el modelo es mucho más preciso en la apertura (plies 1–10: top-1 0,623) que en el medio juego y el final (plies 31–80: 0,191; 81–150: 0,140).

## 7. Línea policy/value basada en tablero y Chess Alpha v1.1

Otra rama investigó representación explícita del tablero y salida de policy/value. Entre las arquitecturas propuestas/registradas aparecen **23 canales de entrada 8×8**, bloques convolucionales residuales (configuración de seis bloques y ancho 256), una fase Transformer (cuatro bloques, ancho 256, ocho cabezas) y cabezales para probabilidad de movimiento/evaluación. Para confirmar qué configuración entrenó cada checkpoint se requiere abrir los `config` y el código del modelo; no debe hacerse equivaler con Sequence Policy.

### Selección de checkpoint

En julio de 2026 se compararon `control_epoch02`, `balanced_epoch02` y `epoch5`. El enfrentamiento directo entre `control_epoch02` y `balanced_epoch02` se consideró prácticamente empatado (una referencia contempla 500 partidas), pero se escogió **`control_epoch02`** por indicadores de calidad de jugada y correlación de valor; se conservó `balanced_epoch02` sin promocionarlo como ganador.

En un benchmark de 100 partidas frente a `epoch5` bajo configuración controlada, `control_epoch02` obtuvo **44 victorias, 39 tablas y 17 derrotas**, score **63,5 %**, estimación relativa **+96,2 Elo** con intervalo aproximado **[+44, +153]** y **cero jugadas ilegales**. Se usó una evaluación híbrida de material 0,25 + value 0,75, profundidad 2 y top-k 16. La estimación **no mide Elo absoluto ni implica superioridad frente a motores externos**.

### Epoch4: modelo operativo seleccionado

Después de v1.1 se construyó y congeló un dataset nuevo (Dataset Epoch 4), con auditorías de leakage, de fases y de control de tiempo, y se entrenó **Epoch4** con la arquitectura `PolicyValueNetHybrid`: 23 planos de entrada, 6 bloques residuales convolucionales de 256 canales y 4 bloques Transformer de 8 cabezas (`src/chess_alpha/model_hybrid.py`). En una Arena confirmatoria de **500 partidas** (250 pares con colores invertidos, aperturas nuevas generadas con semilla fija) contra `control_epoch02`, Epoch4 obtuvo un **score de 60,8 %**, con intervalo de Elo íntegramente por encima de 0, 0 errores y 0 jugadas ilegales. Por eso fue promovido a modelo operativo. Su checkpoint no se publica.

También se revisaron propuestas de positional encoding **Relative2D** A/B sobre una base común de checkpoint, datos y semilla; quedaron por comprobar métricas comparables de coverage, Spearman, centipawn loss y latencia. Para selección de partidas de una fase posterior se examinó un pool de **81.592 partidas** y se corrigió la regla de diversificación de aperturas para limitar **códigos ECO completos** (5% por código), no letras agregadas A–E.

## 8. Motor de búsqueda: Search Tree v1/v1.1

El motor fue desarrollado como sistema de búsqueda legal, no como un árbol de decisión estadístico tradicional. La base utiliza generación de jugadas válidas, aplicación/deshacer o reconstrucción segura de estados, evaluación de hojas y exploración recursiva.

### Componentes registrados

1. **Alpha–Beta** para descartar ramas que no pueden mejorar una decisión ya conocida.
2. **PVS** para explotar que el primer movimiento ordenado suele ser el candidato dominante.
3. **Transposition Table** para memorizar posiciones evaluadas, cotas y mejores movimientos.
4. **Iterative Deepening** para producir una mejor línea a profundidades crecientes.
5. **Quiescence Search** para no detener búsquedas en estados tácticamente inestables; experimentos con `max_qnodes=20.000`.
6. **LMR** para reducir temporalmente la profundidad de exploración de movimientos menos prometedores y reabrirlos cuando sea necesario.
7. **Null Move** como mecanismo selectivo de poda, sometido a salvaguardas.
8. **DFPN/PNS** para exploración de prueba de mate o problemas tácticos especializados.
9. **Legalidad/terminalidad**: jaques, mates, ahogados, repeticiones/ciclos y consistencia de la variación principal.

Hubo trabajo específico de corrección de bugs, benchmarks y cobertura de regresiones. El cierre de las fases I–VI registró **387 tests PASS y 1 skipped**. El cierre de la Fase VIII (integración de Epoch4, Sequence Policy y Fusion) registró **458 PASS, 1 skipped y 0 failed**. Son snapshots sucesivos de la misma suite y no deben sumarse. Nota de nombres: «V1.1» corresponde al *dataset* (`SEARCH TREE DATASET V1.1`); el buscador es **Search Tree V1**, construido en fases I–VIII. En la auditoría de publicación (2026-11-20) se volvió a ejecutar la suite exportada sin datasets ni checkpoints. Los números de esa corrida figuran en el README, e incluyen los tests del Observatory agregados después del cierre de la Fase VIII.

**Integración neuronal (Fases VII y VIII).** Epoch4 y Sequence Policy actúan como proveedores de ordenamiento que solo pueden **reordenar**: devuelven siempre una permutación exacta de las jugadas que PVS ya generó. La autoridad táctica sigue siendo de la búsqueda de ancho completo y de DFPN. En el benchmark A/B/C/D de N=2000 posiciones con historial real (profundidad 2) se registraron 0 jugadas ilegales y 0 errores en 11.988 búsquedas. Epoch4 en `root_only` resolvió el 48,0 % de los casos tácticos y la fusión RRF con Sequence Policy, el 46,0 %. En defensa única, los valores fueron 75,1 % y 73,5 %. Por eso Sequence Policy y Fusion quedan integrados pero desactivados por defecto.

Se incorporó observabilidad opcional: inspección de nodos, eventos y estadísticas de búsqueda sin alterar decisiones. En un benchmark posterior **90 posiciones × 5 configuraciones = 450 ejecuciones** produjeron 450 registros, con **cero mutaciones de estado, cero movimientos ilegales y cero errores de ejecución observados**. La existencia de esta instrumentación sirve para estudiar costo y comportamiento; no demuestra por sí sola superioridad competitiva.

## 9. Calibración de policy prior: éxito parcial y rechazo táctico

Se investigó usar información de la política para mejorar el orden de movimientos del árbol. Una **Fase 6 con `λ=0,15`** sobre **703 posiciones** informó:

| Indicador | Antes | Después |
|---|---:|---:|
| Agreement@1 | 0,228 | 0,464 |
| MRR | 0,348 | 0,629 |
| Policy-correct / search-wrong | 0,306 | 0,041 |

En ese benchmark hubo **197 mejoras y 31 regresiones**. Era una señal positiva sobre ranking/acuerdo, pero no suficiente para aprobar cambios en el motor.

La **Fase 7 fue rechazada** después de detectar dos defensas obligatorias contra mate que no se atendían y una caída de rendimiento en ataques descubiertos del **42,5 % al 27,5 %** sobre **40 casos**. El gate no encontró ilegales ni mates consumados omitidos, pero sí errores tácticos relevantes. **No se hizo promoción del candidato, no se recalibró λ y no se ejecutaron las fases posteriores de Arena vinculadas a ese candidato.**

Este episodio es uno de los resultados más interesantes: **una mejora en métricas globales de ranking puede producir pérdidas locales graves en seguridad táctica**. Por eso se dio prioridad a las pruebas de regresión basadas en situaciones concretas.

## 10. Variables search-aware y futuros modelos basados en features

Después de completar V3 se propuso un enriquecimiento adicional ligado al comportamiento del buscador. Se pasó de una idea inicial de **10 señales** a un diseño de **40**, seguido por una ampliación conceptual a unas **60 variables candidatas** para luego seleccionar las que aportaran valor.

Categorías estudiadas: ranking de orden de movimientos, pertenencia a la principal variation, cutoffs, nodos por rama, `qnodes`, estabilidad del mejor movimiento o de la evaluación entre profundidades, profundidad necesaria para decidir, criterios de seguridad de LMR y discordancias entre policy y search. Las tablas de candidatos se pensaron como outputs separados, unidos mediante claves `row_id + move_id`, sin modificar Golden ni V3.

**Estado verificado en la auditoría de publicación:** la implementación existe (`search_enrichment_v1`, no publicada en esta versión) y define una matriz de 60 familias. Se completaron pilotos de 1.000 y 10.000 posiciones y un gate de 100.000 posiciones (2.000/2.000 chunks), pero ese gate cerró con **PASS operacional = FALSE** por throughput y episodios de suspensión de Windows. La producción completa sobre las 818.611 filas **no se completó**. **No se debe declarar que las 60 variables fueron generadas y validadas completamente.** Tampoco debe afirmarse que ya existe un árbol de decisión entrenado que consuma ese bloque: el objetivo era reunir información y después comparar la aportación predictiva de cada grupo mediante experimentos o ablaciones.

## 11. Dataset de Search Tree, Stockfish y evaluaciones auxiliares

La línea `search_tree_v1` incluía trabajo de construcción de datasets usando posiciones y, en momentos específicos, procesos de Stockfish. También se construyó `SEARCH TREE DATASET V1.1`, con **39.961 posiciones** de partidas humanas de Elo verificado en el rango 2200–2300. Su auditoría final (Fase 19) cerró como PASS_WITH_WARNINGS, con 7 advertencias no bloqueantes. De ese dataset se derivó **Nivel C**: **12.963 posiciones** analizadas exhaustivamente con Stockfish (profundidad 17, MultiPV sobre todas las jugadas legales) en 4 shards consolidados y congelados con SHA-256. Nivel C es terreno de validación posición por posición del buscador, no un dataset de entrenamiento. Ninguno de los dos se publica.

En una fase de validación basada en Python y `python-chess` se ejecutaba CPU-only la cadena de legalidad → Alpha–Beta → TT → quiescence → validación. No es correcto afirmar que esa fase usaba GPU, Stockfish en cada posición o el checkpoint de Sequence Policy.

## 12. Incidencias resueltas y disciplina de calidad

- **Errores de rutas relativas:** corrección antes de dar por válido un trabajo paralelo del tokenizador.
- **Interrupción tardía del vigilante:** se descartó una prueba y se repitió con vigilancia persistente.
- **Incompatibilidad dtype de bitboards:** corregida durante V3 y cubierta con regresión.
- **Acumulación de gradientes:** se corrigió y testeó en Sequence Policy FULL.
- **Riesgos de concurrencia:** se observó modificación externa de archivos de `search_v1` por otra sesión; se trató como conflicto de edición y no como autorización para sobrescribir cambios.
- **Contaminación de experimentos:** preservación de splits/golden y checkpoints originales; rechazo de candidatos con regresiones aunque mejoraran métricas medias.
- **Limitación de recursos y movilidad:** checkpoints seguros y reanudación tras detener sesiones o trasladar equipo.

## 13. Qué está realmente terminado y qué no

| Componente | Juicio documental |
|---|---|
| Muestreo PGN y piloto 10k | Implementado y probado |
| Esquema de action ID uint16 | Implementado y probado |
| Dataset principal de 1M | Completo: manifest con 1.000.000 partidas, 81.244.568 tokens, semilla 20260803 y hashes SHA-256 |
| Golden Dataset | Construido, congelado y comprobado |
| V3 de 28 familias | Full-run y validaciones completadas; código publicado |
| Sequence Policy FULL step 2286 | Entrenamiento final registrado y checkpoint recuperado; código publicado, checkpoint no |
| Sequence Policy TEST definitivo | Completado: `FROZEN_TEST_FINAL`, top-1 0,301 y MRR 0,434 |
| v1.1 control vs epoch5 | Evaluación arena contextualizada y documentada |
| Epoch4 vs control_epoch02 | Arena confirmatoria de 500 partidas con score de 60,8 %; Epoch4 seleccionado |
| Search Tree V1 (fases I–VIII) | Implementado, testeado y congelado; código publicado |
| Observabilidad de búsqueda | Benchmark de 450 ejecuciones validado |
| Priors `λ=0,15` | Fase 6 favorable, Fase 7 rechazada |
| 40–60 señales search-aware | Implementado y en piloto; gate de 100k con PASS operacional = FALSE; producción completa no realizada |
| Tuning de parámetros de búsqueda (Optuna) | Experimental, no publicado |
| Arena de `search_v1` | Piloto de 40 partidas (score 50 %), no concluyente, no publicado |
| Self-play con RL + MCTS | Investigación/propuesta, no confirmado como implementado |
| Modelo final integrado / motor de torneo | No certificado por esta evidencia |

## 14. Recomendaciones para reproducibilidad y publicación

Para que GitHub sea una documentación técnica fiable, conviene vincular los números anteriores a los artefactos reales: `manifest.json`, reportes JSON, hashes SHA-256, los tests de cada rama y checkpoints con licencia clara. Las rutas personales de Windows pueden conservarse en notas privadas, pero no son necesarias en la publicación pública. Nunca subir PGNs ajenos, credenciales, archivos corporativos, bundles pesados ni binarios de terceros sin verificar permisos.

La división entre **policy/value por tablero**, **Sequence Policy por tokens**, **Search Tree**, **enriquecimiento V3** y **search-aware** debe mantenerse también en la estructura del repositorio: cada rama tiene datasets, métricas y criterios de éxito diferentes. El resultado más relevante de Chess Alpha no es una única cifra de Elo, sino el recorrido completo de aprendizaje automático, infraestructura de datos, diseño algorítmico y evaluación controlada de decisiones.

---

*Documento elaborado como memoria de ingeniería para acompañar el código. Revisar nombres de módulos, rutas relativas y resultados finales contra el repositorio antes de publicar una versión definitiva.*