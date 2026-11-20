"""search_v1 - Fase I de SEARCH TREE V1 (posicion, movimientos, terminales,
repeticion, evaluacion estatica minima). Paquete nuevo, aislado de
src/search/ (motor operativo actual, sin tocar). Ver
reports/search_tree_v1/SEARCH_TREE_V1_TECHNICAL_DESIGN.md para el diseno
completo y reports/search_tree_v1/SEARCH_TREE_V1_IMPLEMENTATION_ROADMAP.md
para las fases siguientes (NO implementadas todavia: Negamax, Alpha-Beta,
PVS, quiescence, TT operativa, podas, DFPN, integracion Epoch4/Sequence
Policy, personalidad, Arena)."""

from search_v1.alpha_beta import alpha_beta
from search_v1.aspiration import AspirationConfig, search_with_aspiration
from search_v1.candidate_fusion import FusionConfig, reorder_with_fusion
from search_v1.dfpn import search_dfpn
from search_v1.dfpn_gate import DFPNGateConfig
from search_v1.dfpn_limits import DFPNLimits
from search_v1.dfpn_node import DFPNNode, NodeType
from search_v1.dfpn_result import DFPNResult
from search_v1.dfpn_table import DFPNEntry, DFPNTable
from search_v1.epoch4_provider import Epoch4OrderingCache, Epoch4OrderingConfig, evaluate_epoch4, reorder_with_epoch4
from search_v1.exceptions import IllegalMoveError, InvalidFENError, SearchV1Error, UndoStackEmptyError
from search_v1.forced_defense import ForcedDefenseResult, MoveDefenseClassification, analyze_forced_defense
from search_v1.iterative_deepening import iterative_deepening_search
from search_v1.mate_scores import (
    INFINITY, MATE_SCORE, is_mate_score, mate_distance_plies_from_score,
    score_from_tt, score_to_tt, terminal_score,
)
from search_v1.mate_search import find_forced_mate
from search_v1.move import Move
from search_v1.move_generator import LegalMove, generate_legal_moves, is_legal_uci, legal_move_count
from search_v1.move_ordering import OrderingState, order_moves, order_tactical_moves
from search_v1.negamax import negamax
from search_v1.observer import NodeEvent, SearchObserver
from search_v1.position import Position
from search_v1.principal_variation import validate_principal_variation
from search_v1.prior_mode import PRIOR_MODES, validate_prior_mode_configuration
from search_v1.proof_status import ProofStatus, is_proven, is_unknown
from search_v1.proof_verifier import exhaustive_mate_oracle, verify_mate_proof
from search_v1.pruning_config import PruningConfig
from search_v1.pvs import pvs
from search_v1.quiescence import quiescence
from search_v1.quiescence_limits import QuiescenceLimits
from search_v1.quiescence_result import quiescence_telemetry
from search_v1.repetition import RepetitionTracker, zobrist_key
from search_v1.result import PositionAnalysis, analyze_position
from search_v1.search_context import SearchAborted, SearchContext
from search_v1.search_limits import SearchLimits
from search_v1.search_result import ScoreType, SearchResult, StopReason
from search_v1.sequence_policy_provider import (
    SequencePolicyCache, SequencePolicyConfig, evaluate_sequence_policy, reorder_with_sequence_policy,
)
from search_v1.static_evaluation import evaluate_static_relative, evaluate_static_white_pov
from search_v1.tactical_moves import generate_tactical_moves
from search_v1.terminal import TerminalResult, TerminalStatus, detect_terminal, is_fifty_move_rule
from search_v1.transposition_table import Bound, TTEntry, TranspositionTable

__all__ = [
    "SearchV1Error", "InvalidFENError", "IllegalMoveError", "UndoStackEmptyError",
    "Move", "Position",
    "LegalMove", "generate_legal_moves", "legal_move_count", "is_legal_uci",
    "TerminalStatus", "TerminalResult", "detect_terminal", "is_fifty_move_rule",
    "RepetitionTracker", "zobrist_key",
    "evaluate_static_white_pov", "evaluate_static_relative",
    "PositionAnalysis", "analyze_position",
    # Fase II
    "negamax", "alpha_beta", "iterative_deepening_search",
    "SearchLimits", "SearchResult", "ScoreType", "StopReason",
    "SearchContext", "SearchAborted",
    "MATE_SCORE", "INFINITY", "is_mate_score", "mate_distance_plies_from_score", "terminal_score",
    "validate_principal_variation",
    # Fase III
    "TranspositionTable", "TTEntry", "Bound", "score_to_tt", "score_from_tt",
    "OrderingState", "order_moves",
    # Fase IV
    "quiescence", "QuiescenceLimits", "quiescence_telemetry",
    "generate_tactical_moves", "order_tactical_moves",
    # Fase V
    "pvs", "PruningConfig", "AspirationConfig", "search_with_aspiration",
    # Fase VI
    "ProofStatus", "is_proven", "is_unknown",
    "DFPNLimits", "DFPNNode", "NodeType", "DFPNEntry", "DFPNTable", "DFPNResult",
    "search_dfpn", "find_forced_mate",
    "ForcedDefenseResult", "MoveDefenseClassification", "analyze_forced_defense",
    "verify_mate_proof", "exhaustive_mate_oracle",
    "DFPNGateConfig",
    # Fase VII
    "Epoch4OrderingConfig", "Epoch4OrderingCache", "evaluate_epoch4", "reorder_with_epoch4",
    # Fase VIII
    "SequencePolicyConfig", "SequencePolicyCache", "evaluate_sequence_policy", "reorder_with_sequence_policy",
    "FusionConfig", "reorder_with_fusion",
    "PRIOR_MODES", "validate_prior_mode_configuration",
    # PASO 0 (Search Tree Observatory)
    "SearchObserver", "NodeEvent",
]
