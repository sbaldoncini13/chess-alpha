"""Schemas parquet compartidos por 12_stockfish_validation.py y
16_stockfish_full_enrichment.py -- una sola definicion, sin duplicar."""
import pyarrow as pa

EVAL_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("depth", pa.int16()),
    ("multipv_configured", pa.int16()), ("multipv_returned", pa.int16()),
    ("best_move_id", pa.int32()), ("best_move_uci", pa.string()),
    ("best_eval_type", pa.string()), ("best_eval_cp", pa.float32()), ("best_eval_mate_in", pa.int16()),
    ("best_eval_unified_cp_equivalent", pa.int32()),
    ("human_move_uci", pa.string()), ("human_eval_type", pa.string()),
    ("human_eval_cp", pa.float32()), ("human_eval_mate_in", pa.int16()),
    ("human_eval_unified_cp_equivalent", pa.int32()),
    ("centipawn_loss", pa.int32()), ("move_quality", pa.string()),
])

TOPK_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("rank", pa.int16()), ("move_id", pa.int32()), ("move_uci", pa.string()),
    ("eval_type", pa.string()), ("eval_cp", pa.float32()), ("mate_in", pa.int16()),
    ("eval_unified_cp_equivalent", pa.int32()), ("depth", pa.int16()),
])

POLICY_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("move_id", pa.int32()), ("probability", pa.float32()), ("rank", pa.int16()),
])

CRITICALITY_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("engine_criticality_score", pa.float32()),
    ("engine_tactical_or_critical", pa.bool_()), ("any_mate_candidate", pa.bool_()),
    ("gap_cp", pa.float32()), ("policy_top1_weight", pa.float32()),
])


def table_from_rows(rows: list, schema: pa.Schema) -> pa.Table:
    cols = {name: [row[name] for row in rows] for name in schema.names}
    return pa.table(cols, schema=schema)
