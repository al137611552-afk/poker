"""德扑训练台 · 牌局引擎（纯逻辑层）。"""

from .actions import Action, ActionKind, LegalActions, bet, call, check, fold, raise_to
from .batch import BatchResult, MatchConfig, SeatStats, merge, run_batch, shard
from .cards import card_from_str, card_to_str, cards_from_str, cards_to_str
from .deck import deck_from_seed, shuffled_deck, stacked_deck
from .equity import exact_equity, monte_carlo_equity
from .equity_table import (
    equity_vs_range,
    preflop_equity,
    range_vs_range_equity,
)
from .evaluator import describe, evaluate
from .history import ActionRecord, action_records
from .metrics import bb_per_100, bb_per_100_interval
from .phh import parse_phh, phh_player_order, to_phh
from .positions import position_names, position_of
from .pots import Pot, award, build_pots, refund_uncalled
from .preflop_chain import TableConfig, TableSolution, solve_table
from .preflop_solver import PreflopSolution, solve_preflop
from .preflop_tree import PreflopConfig, PreflopTree, SubgameConfig, build_tree
from .pushfold import PushFoldSolution, solve_push_fold
from .realization import RealizationModel
from .ranges import (
    NUM_HAND_CLASSES,
    TOTAL_COMBOS,
    Range,
    class_combos,
    class_from_name,
    class_name,
    class_of,
    grid_position,
)
from .state import COMPLETE, FLOP, PREFLOP, RIVER, TURN, HandConfig, HandResult, HandState
from .store import HandStore, PlayerSummary

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionKind",
    "ActionRecord",
    "LegalActions",
    "HandConfig",
    "HandResult",
    "HandState",
    "HandStore",
    "PlayerSummary",
    "Pot",
    "Range",
    "NUM_HAND_CLASSES",
    "TOTAL_COMBOS",
    "class_combos",
    "class_from_name",
    "class_name",
    "class_of",
    "PreflopConfig",
    "PreflopSolution",
    "PreflopTree",
    "SubgameConfig",
    "TableConfig",
    "TableSolution",
    "PushFoldSolution",
    "RealizationModel",
    "MatchConfig",
    "BatchResult",
    "SeatStats",
    "run_batch",
    "shard",
    "merge",
    "bb_per_100",
    "bb_per_100_interval",
    "build_tree",
    "solve_preflop",
    "solve_table",
    "equity_vs_range",
    "exact_equity",
    "preflop_equity",
    "range_vs_range_equity",
    "solve_push_fold",
    "grid_position",
    "monte_carlo_equity",
    "action_records",
    "parse_phh",
    "phh_player_order",
    "position_names",
    "position_of",
    "to_phh",
    "PREFLOP",
    "FLOP",
    "TURN",
    "RIVER",
    "COMPLETE",
    "award",
    "bet",
    "build_pots",
    "call",
    "card_from_str",
    "card_to_str",
    "cards_from_str",
    "cards_to_str",
    "check",
    "deck_from_seed",
    "describe",
    "evaluate",
    "fold",
    "raise_to",
    "refund_uncalled",
    "shuffled_deck",
    "stacked_deck",
    "__version__",
]
