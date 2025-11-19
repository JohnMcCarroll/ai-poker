"""
This module defines all the constants required for initializing DEAP for evolving poker strategies and parameters specific to the algorithm.

Author: John McCarroll
"""
import numpy as np


# DEAP set up constants
RANK_ORDER = {
    None: 0., '2': 2., '3': 3., '4': 4., '5': 5., '6': 6., '7': 7., '8': 8., '9': 9., 'T': 10., 'J': 11., 'Q': 12., 'K': 13., 'A': 14.
}
OUTPUT_TYPE = float
# The arguments for our evolved function. This defines the inputs for the agents.
ARGUMENT_TYPES = [
    float,      # pot_size
    bool,       # button (True for small blind, False for big blind)
    float,      # stack_size
    float,      # opponent_stack_size
    float,      # amount_to_call
    float,      # hand_strength (a value from 0.0 to 1.0)
    str,        # hand_class (e.g., 'PAIR', 'FLUSH')
    str,        # street (e.g., 'PREFLOP', 'FLOP')

    float,      # 'board_highest_card',
    float,      # 'board_num_pairs',
    float,      # 'board_num_trips', 
    float,      # 'board_num_quads',
    float,      # 'board_num_same_suits',
    float,      # 'board_smallest_3_card_span',

    bool,       # 'hole_suited',
    float,      # 'hole_high_card',
    float,      # 'hole_low_card',
    bool,       # 'hole_paired',
    bool,       # 'hole_flush_draw',
    bool,       # 'hole_open_straight_draw',
    bool,       # 'hole_gutshot_straight_draw'

    str,        # 'preflop_opponent_line'
    str,        # 'flop_opponent_line'
    str,        # 'turn_opponent_line'
    str,        # 'river_opponent_line'

    float,      # num_hands
    float,      # VPIP
    float,      # PFR
    float,      # 3BET
    float,      # WTSD
    float,      # W$SD
    float,      # WWSF
    float,      # AF
    float,      # CBET%
    float,      # DONK%
    float,      # CHECKRAISE%
]
FLOAT_CONSTANTS = [
    -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.25, 
    -1.1, -1.0, -0.9, -0.75, -0.667, -0.5, -0.333, -0.25, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 
    6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
]
HAND_CLASSES = [
    'HIGH_CARD', 'PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT',
    'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH'
]
STREETS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
BETTING_LINES = [
    "DONK_BET", "BET", "RAISE", "CHECK", "CALL", "THREE_BET", "FOUR_BET", 
    "DONK_BET-CALL", "BET-CALL", "CHECK-RAISE", "RAISE-CALL", "CALL-CALL"
]
HAND_STRENGTH_MAP = {i: strength for i, strength in enumerate(np.linspace(0, 1, len(HAND_CLASSES)))}
STREET_MAP = {0: 'PREFLOP', 1: 'FLOP', 2: 'TURN', 3: 'RIVER'}

# algorithm parameters constants
INITIAL_MAX_TREE_HEIGHT = 9
INITIAL_MIN_TREE_HEIGHT = 2
MAX_NODE_COUNT = 10000
ELITE_PCT = 0.05
IMMIGRATION_PCT = 0.05
# mutation, prune, and crossover probabilities should add up to 1.0
PROB_CROSSOVER = 0.80
PROB_MUTATION = 0.10
PROB_PRUNE = 0.10
POP_SIZE = 500
N_GEN = 10000
MAX_HANDS = 200
MIN_HANDS = 100
HAND_NUM_STEP_SIZE = 10
GEN_CURRICULUM_STEP_SIZE = 200
EVALUATION_BENCH_SIZE = 50
WIN_RATE_FITNESS_WEIGHT = 1.0
NODE_COUNT_FITNESS_WEIGHT = -0.1
SEED = 55
SAVE_EVERY_X_GEN = 10
VERSION_NUM = 1.4
LOG = True
VERBOSE = True
EVALUATION_TIMEOUT = 600
RESTART_FROM_CKPT = None
# RESTART_FROM_CKPT = "C:\\Users\\John\Desktop\\poker_bot\\ai-poker\\ai_poker\\genetic\\fossils\\evo_ckpt_v1.3_gen130_2025-11-13_09-41-10.pkl"
