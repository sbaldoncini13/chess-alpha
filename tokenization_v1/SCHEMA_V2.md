# tokenization_schema_v2 — esquema del dataset enriquecido de posiciones

Documento de referencia para `tokenization_v1/scripts/06_enrich_positions.py`
y el resto de los módulos `v2`. No reemplaza ni modifica el schema `tokv1.0`
(secuencias de `action_id` por partida en `outputs/<test_name>/`) — lo
**extiende** leyéndolo en modo solo lectura y produciendo un dataset nuevo,
posicion por posicion, en `outputs_v2/<test_name>/`.

## Relación con v1 (no negociable)

v2 nunca reabre el PGN crudo ni reinvoca `game_parser.py`. Parte de los
artefactos ya validados de v1 (`moves_uint16.bin`, `game_offsets.npy`,
`game_lengths.npy`, `results.npy`, `metadata.parquet`) y reconstruye cada
posición reproduciendo la secuencia de `action_id` ya validada de esa
partida con `python-chess`. El vocabulario de movimientos (`action_id`,
5·64·64 = 20480, `tokenizer.py`) se reutiliza **sin cambios** — sigue siendo
`tokv1.0`.

## Filas por partida

Para una partida con `ply_count` movimientos se generan **`ply_count + 1`**
filas (posiciones), no `ply_count`: una por cada posición *antes* de cada
movimiento jugado (filas `0..ply_count-1`, con `human_move_id` = movimiento
jugado desde ahí) más **una fila final** para la posición resultante después
del último movimiento (sin movimiento humano asociado — `human_move_id`
nulo). Esta fila final es la única en la que `is_checkmate` / `is_stalemate`
pueden ser verdaderos (si la posición ya está en jaque mate o ahogada la
partida terminó ahí; en las demás filas, por construcción, siempre existe un
movimiento legal jugado a continuación). Ver más abajo por qué esto hace que
el conteo total de posiciones del piloto (~818k) no coincida exactamente con
el conteo de plies de referencia de v1 (~808.611) — es la razón documentada,
no una discrepancia sin explicar.

## Tabla `positions` (parquet, por shard)

| columna | tipo | PASO | descripción |
|---|---|---|---|
| `row_id` | int64 | 5 | id global secuencial 0..N-1, único por fila |
| `game_id` | int64 | 5 | = `sample_id` de v1 (índice en `metadata.parquet` de v1) |
| `ply` | int32 | 5 | posición antes del `ply`-ésimo movimiento (0-based); en la fila final = `ply_count` de la partida |
| `fen` | string | 5,6,7,8,9,10 | FEN completo de la posición — snapshot suficiente para reconstrucción inequívoca sin reproducir la partida desde el inicio |
| `side_to_move` | bool | 7 | `true` = blancas mueven |
| `castling_white_kingside/queenside`, `castling_black_kingside/queenside` | bool | 8 | derechos reales de enroque en esta posición (no inferidos de la ubicación de rey/torres) |
| `en_passant_square` | int8 | 9 | 0-63, o -1 si no hay casilla de captura al paso disponible |
| `halfmove_clock` | int16 | 10 | contador de la regla de 50 movimientos |
| `fullmove_number` | int16 | 10 | número de jugada completa |
| `position_id` | uint64 | 4 | `chess.polyglot.zobrist_hash(board)` — identidad de posición independiente de la partida |
| `position_occurrence_count` | int32 | 4 | cuántas veces aparece este `position_id` en TODO el dataset (piloto), calculado en un pase previo; no se elimina ninguna repetición automáticamente |
| `repetition_count_in_game` | int8 | 20 | 1/2/3+, cuántas veces se vio esta posición (mismo `position_id`) dentro de ESTA partida hasta esta fila inclusive |
| `legal_move_count` | int16 | 12,28 | cantidad de movimientos legales desde esta posición |
| `human_move_id` | int32 | 13 | `action_id` del movimiento jugado; -1 en la fila final (no hay movimiento) |
| `human_move_uci` | string | 13 | UCI del movimiento jugado, solo para debugging/auditoría; vacío en la fila final |
| `value_target_stm` | int8 | 14 | resultado de la partida desde la perspectiva del jugador al turno EN ESTA POSICIÓN: +1 gana el que mueve, 0 tablas, -1 pierde el que mueve |
| `game_result_white_pov` | int8 | 14 | resultado crudo desde blancas (+1/0/-1), igual a `result_code` de v1, para auditoría de la convención de signo |
| `white_pawns..white_queens`, `black_pawns..black_queens` | int8 | 15 | conteo de piezas por tipo y color (sin rey) |
| `material_white`, `material_black`, `material_diff` | int16 | 15 | escala P=1,N=3,B=3,R=5,Q=9 (misma que `src/chess_alpha/board_features.material_score`, solo lectura) |
| `game_phase` | string | 16 | `"opening"`/`"middlegame"`/`"endgame"` — heurística de `src/chess_alpha/board_features.game_phase` (solo lectura, reimplementada sin importar el módulo): material total ≤20 o sin damas → endgame; ≤45 → middlegame; si no, opening |
| `is_check`, `is_checkmate`, `is_stalemate` | bool | 17 | estado de ESTA posición (antes del movimiento humano, si lo hay) |
| `move_is_capture/check/checkmate/promotion/castling/en_passant/quiet` | bool | 18 | flags del movimiento humano jugado desde esta fila (todo en `false` en la fila final); `quiet` = ninguno de capture/promotion/castling/en_passant (puede coexistir con `check`) |
| `white_elo`, `black_elo`, `eco` | int32/string | metadata | copiados de `metadata.parquet` de v1 para esta partida |
| `difficulty_score`, `difficulty_bucket` | float32/string | 28 | ver fórmula abajo |
| `tactical_or_critical` | bool | 27 | ver fórmula abajo |
| `criticality_score` | float32 | 27 | ver fórmula abajo |
| `sample_weight` | float32 | 29 | 1.0 por defecto en toda la tabla |

## Tabla `legal_moves` (formato ragged, análogo a v1)

Igual patrón que `moves_uint16.bin` + `game_offsets.npy` de v1, para no
duplicar 20480 columnas por fila:

- `legal_moves_uint16.bin`: todos los `action_id` legales de todas las
  posiciones, concatenados en orden de `row_id`.
- `legal_moves_offsets.npy` (int64): offset de inicio por `row_id`.
- (el conteo por fila ya está en `positions.legal_move_count`)

`tokenization_v1/scripts/legal_moves.py::expand_legal_mask(action_ids,
vocab_size=20480)` reconstruye el vector denso 0/1 de tamaño 20480 bajo
demanda. Se valida que `human_move_id` (cuando existe) esté siempre
contenido en su propia `legal_moves` list.

## Historial de movimientos (PASO 19)

No se duplica: `game_id` + `ply` ya son una referencia O(1) al array
`moves_uint16.bin`/`game_offsets.npy` de **v1** (solo lectura). La función
`tokenization_v1/scripts/history.py::get_move_history(game_id, ply, k)`
devuelve los últimos `k` `action_id` anteriores a `ply` (o menos si la
partida no tiene tantos), para `k ∈ {4,8,16,32}` configurable. No se
modifica el Transformer para consumirlos — solo se deja la función probada.

## Tensor de tablero (PASO 6)

`tokenization_v1/scripts/position_state.py::board_to_tensor_v2(board)`
reproduce **exactamente** la semántica canónica de
`src/chess_alpha/encoding.py::board_to_tensor` (solo lectura, no
modificada, no importada — reimplementada localmente para mantener
`tokenization_v1/` como módulo independiente): 23 planos 8×8 float32 —
6 planos piezas blancas + 6 negras (orden P,N,B,R,Q,K) + 1 turno + 4
enroques + 2 casillas atacadas (blancas/negras) + material_balance/39 +
movilidad(min(legal,60)/60) + jaque + fase numérica (opening=0, middlegame=0.5,
endgame=1). Orientación absoluta (no se voltea el tablero para negras — plano
0 son siempre peones blancos en su casilla real). Validado con test de
round-trip byte-a-byte contra una reimplementación de referencia y con
`board.turn`/`board.pieces()` de python-chess.

**No se persiste este tensor denso para las ~818k posiciones del piloto**
(ver justificación en `tokenization_v2_config.json ->
materialize_dense_tensors.reason`): se guarda el FEN + columnas
estructuradas (arriba), que reconstruyen el tensor en O(1) por posición sin
volver a parsear PGN ni revalidar legalidad. Se materializa una muestra
(`materialize_dense_tensors.demo_sample_size` posiciones) como
`outputs_v2/<test_name>/dense_tensor_demo.npz` para validar shape/dtype
sobre datos reales.

## Fórmulas base (sin Stockfish) — configurables, no hardcodeadas en silencio

**`difficulty_score`** (0..1, mayor = más difícil), pesos en
`tokenization_v2_config.json -> difficulty_weights`:

```
score = w_inverse_legal_moves * (1 - min(legal_move_count,60)/60)
      + w_is_check              * is_check
      + w_forced_move           * (legal_move_count == 1)
      + w_phase_endgame_bonus   * (game_phase == "endgame")
```//clamped a [0,1]. `difficulty_bucket` = "low"(<0.33) / "medium"(<0.66) / "high".

**`tactical_or_critical`** / **`criticality_score`** (sin motor — señales
disponibles en la tokenización base únicamente), umbrales en
`criticality_thresholds`:

```
criticality_score = 0.5 * (legal_move_count <= low_branching_max_legal_moves)
                   + 0.3 * is_check
                   + 0.2 * (legal_move_count <= forced_move_max_legal_moves)
```
`tactical_or_critical = criticality_score >= 0.5`.

Cuando el enriquecimiento Stockfish está habilitado (PASO 21-27), el shard
`engine_*` separado añade una **segunda** columna
`engine_criticality_score` (basada en swing de evaluación / gap primer vs.
segundo candidato / mate disponible) — nunca sobrescribe la base, para no
mezclar metadata determinística con engine labels (regla explícita del
encargo: "no mezclar inputs, targets y metadata").

## Deduplicación de posiciones (PASO 4) — dos pasadas

Pasada A (liviana): recorre todas las partidas, calcula `position_id`
(zobrist) por posición y acumula un `Counter` global en memoria (~800k
enteros de 64 bits ≈ decenas de MB, no cientos). Pasada B: escribe las filas
con `position_occurrence_count` ya conocido (el conteo final, no un conteo
parcial creciente). No se elimina ninguna posición duplicada — queda a
criterio de una decisión posterior (p. ej. vía `sample_weight`).

## Tablas de enriquecimiento Stockfish (PASO 21-27) — opcionales, separadas

Solo se generan si `enable_stockfish=true` y `STOCKFISH_PATH` está
configurado. Nunca se mezclan en la tabla `positions` base:

- `engine_eval.parquet`: `row_id`, `engine_depth`, `engine_eval_cp`,
  `engine_is_mate`, `engine_mate_in`, `engine_best_move_id`,
  `engine_best_move_uci`, `centipawn_loss`, `move_quality`,
  `engine_tactical_or_critical`, `engine_criticality_score`.
- `engine_topk.parquet` (formato largo, una fila por candidato):
  `row_id`, `rank`, `move_id`, `move_uci`, `score_cp`, `is_mate`, `mate_in`.
- `engine_policy_target.parquet` (formato largo): `row_id`, `move_id`,
  `weight` — softmax con temperatura configurable
  (`stockfish_soft_policy_temperature_cp`) sobre `engine_topk` scores en cp
  (mates mapeados a un cp equivalente grande con signo, misma convención que
  `src/chess_alpha/engine_eval.py:CP_MATE_SCORE=100000`, solo lectura).

**Nota (auditoría 2026-08-11)**: esta sección describía el diseño conceptual
antes de que existiera una implementación que persistiera realmente el
detalle de `engine_topk`/`engine_policy_target` (solo se guardaban tamaños
agregados en el demo de 60 posiciones). Ver **"Extensión v2.1"** más abajo
para el formato REALMENTE implementado y ejecutado (prueba controlada de
1000 posiciones) — los nombres de columna difieren levemente de este boceto
original (se separa `eval_type` cp/mate explícitamente en vez de dos
columnas `is_mate`/`mate_in` implícitas) y se agrega `engine_criticality.parquet`
como tabla propia, nunca fusionada con `engine_eval.parquet`.

`centipawn_loss = max(0, eval_before_stm - eval_after_stm)`, ambos ya
convertidos a la perspectiva de quien mueve en esa posición (nunca se resta
un cp con un mate score sin pasar antes por la conversión de
`_score_to_cp`). `move_quality` usa `quality_thresholds_cp` (6 categorías:
best/excellent/good/inaccuracy/mistake/blunder), configurable.

## Manifest y versionado

`schema_version = "tokenization_schema_v2"`, `pipeline_version = "tokv2.0"`.
El manifest general (`outputs_v2/<test_name>/manifest.json`) documenta
config usada, hashes de los outputs de v1 leídos, conteos, Stockfish
habilitado/deshabilitado, y errores encontrados — ver
`tokenization_v1/reports/<test_name>_v2_report.json` para el reporte final
del piloto.

## Extensión v2.1 — persistencia real engine_topk / engine_policy_target / engine_criticality

Cierre de los 5 huecos técnicos detectados por la auditoría 30/30
(2026-08-11). **No modifica ni regenera** `shards_v2/<test_name>/` — es
puramente aditiva, versionada como `engine_schema_extension_version =
"tokenization_schema_v2.1-engine"` (`tokenization_v2_config.json`). Escrita
por `scripts/12_stockfish_validation.py`, que reutiliza
`StockfishEnricher`/`softmax_policy_from_topk` de `stockfish_enrichment.py`
**sin modificarlos**.

Representación de evaluación (nunca se mezcla mate con cp sin distinguir):
cada score se descompone en `eval_type` (`"cp"` o `"mate"`) + `eval_cp`
(float, solo poblado si `eval_type=="cp"`) + `mate_in` (int, solo poblado si
`eval_type=="mate"`, con signo: positivo = mate a favor de quien mueve) +
`eval_unified_cp_equivalent` (int, SIEMPRE poblado — la conversión ya
existente vía `CP_MATE_SCORE=100000`, usada solo internamente para
`centipawn_loss`/`gap`; nunca debe leerse como centipawns literales sin
antes chequear `eval_type`).

**`engine_eval.parquet`** (1 fila por posición analizada, clave `row_id`):
`row_id, depth, multipv_configured, multipv_returned, best_move_id,
best_move_uci, best_eval_type, best_eval_cp, best_eval_mate_in,
best_eval_unified_cp_equivalent, human_move_uci, human_eval_type,
human_eval_cp, human_eval_mate_in, human_eval_unified_cp_equivalent,
centipawn_loss, move_quality`.

**`engine_topk.parquet`** (formato largo, 1 fila por posición×candidato,
clave `(row_id, rank)`): `row_id, rank, move_id, move_uci, eval_type,
eval_cp, mate_in, eval_unified_cp_equivalent, depth`. Invariantes
garantizados por construcción y verificados por test: `rank` arranca en 1 y
es único por `row_id`; `rank=1` coincide siempre con
`engine_eval.best_move_uci` de esa posición; todos los `move_id` son
legales en esa posición (provienen directamente del PV de Stockfish, que
solo devuelve jugadas legales); cantidad de filas por `row_id` ≤
`multipv_configured`.

**`engine_policy_target.parquet`** (formato largo, clave `(row_id,
move_id)`): `row_id, move_id, probability, rank`. `probability` es
EXACTAMENTE `softmax_policy_from_topk(topk, stockfish_soft_policy_temperature_cp)`
ya existente (fórmula sin cambios, solo ahora persistida completa en vez de
solo su tamaño) — softmax sobre `eval_unified_cp_equivalent` con la
temperatura configurada. `sum(probability)` por `row_id` ≈ 1.0 (tolerancia
1e-6), sin negativos/NaN/inf por construcción de softmax.

**`engine_criticality.parquet`** (1 fila por posición, clave `row_id`):
`row_id, engine_criticality_score, engine_tactical_or_critical,
any_mate_candidate, gap_cp, policy_top1_weight`. Fórmula completa en
`scripts/engine_criticality.py`, resumida:

```
si CUALQUIER candidato en engine_topk[row_id] es mate (a favor o en contra):
    engine_criticality_score = 1.0   (any_mate_candidate = true)
si no:
    gap_cp = topk[0].eval_unified_cp_equivalent - topk[1].eval_unified_cp_equivalent
             (null si hay menos de 2 candidatos)
    gap_term = clamp(gap_cp / gap_saturation_cp, 0, 1)   (0 si gap_cp es null)
    policy_top1_weight = probability del candidato rank=1 (1.0 si solo hay 1 candidato)
    engine_criticality_score = clamp(
        w_gap * gap_term + w_policy_concentration * policy_top1_weight, 0, 1)

engine_tactical_or_critical = engine_criticality_score >= threshold
```

Constantes (`tokenization_v2_config.json -> engine_criticality`):
`gap_saturation_cp=200`, `w_gap=0.6`, `w_policy_concentration=0.4`,
`threshold=0.5` — mismo umbral que `is_tactical_or_critical` base, para que
ambas banderas sean comparables aunque midan cosas distintas.
**`criticality_score`/`tactical_or_critical` (base) NUNCA se sobrescriben**
— siguen existiendo, sin cambios, en `positions.parquet`.

### Prueba controlada `stockfish_validation_1000/`

`scripts/12_stockfish_validation.py` ejecuta esta extensión sobre una
muestra determinista de `stockfish_validation.sample_size` posiciones
(1000 por defecto, seed `stockfish_validation.seed`), seleccionadas por
`scripts/sample_selection.py::select_stratified_sample` (cuotas fijas de
jaque/táctica + resto repartido por fase, ver código — no fuerza
distribuciones imposibles, solo prioriza cobertura de casos). Nunca
selecciona filas terminales (`human_move_id == -1`). Escribe en
`outputs_v2/<test_name>/<stockfish_validation.output_subdir>/` — **nunca**
en `stockfish_demo.parquet` (que queda intacto como artefacto histórico del
demo de 60 posiciones). Reutiliza `stockfish_depth`/`stockfish_multipv` ya
definidos (10/3) — la extensión valida PERSISTENCIA, no busca mejorar
fuerza de análisis.
