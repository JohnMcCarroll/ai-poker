import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import matplotlib.pyplot as plt
from ai_poker.mvp.poker_env import PokerEnv
from ai_poker.mvp.agents import RandomAgent, RandomAgent2, BaseAgent
from ai_poker.genetic.simple_agents import ManiacAgent, StationAgent, SimpleValueAgent
from clubs_gym.agent.base import BaseAgent
from typing import Any
import clubs
import random
import os
import pickle
import datetime
import inspect
from collections import Counter, namedtuple
from itertools import combinations
import copy
# ... (near your other imports)
import multiprocessing
import inspect
from deap import gp

# --- This is your new parallel evaluation function ---
# It MUST be at the top level of the script to be pickled.
def run_evaluation(task):
    """
    Wrapper function to run a single evaluation match in a worker process.
    """
    # 1. Unpack the task
    # (j will be None for a bench match)
    agent1_tree, agent2_tree, max_hands, i, j = task

    def compile_agent(agent_tree_or_class):
        """Helper to compile a DEAP tree or return a class."""
        if inspect.isclass(agent_tree_or_class):
            # It's a benchmark agent like RandomAgent
            return agent_tree_or_class
        # elif isinstance(agent_tree_or_class, gp.PrimitiveTree):
        elif isinstance(agent_tree_or_class, list):
            # It's a DEAP individual (from pop or fossil_record)
            deap_ind = creator.Individual(agent_tree_or_class)
            return toolbox.compile(expr=deap_ind)
        else:
            # Fallback for unexpected types
            print(f"Warning: Unknown agent type in eval: {type(agent_tree_or_class)}")
            return agent_tree_or_class

    try:
        # 2. Compile logic *inside the worker*
        agent1_logic = compile_agent(agent1_tree)
        agent2_logic = compile_agent(agent2_tree)
        
        # 3. Run the evaluation
        w1, w2, n_hands = toolbox.evaluate(agent1_logic, agent2_logic, max_hands=max_hands)
        
        # 4. Return results with indices to map back
        return (i, j, w1, w2, n_hands)
        
    except Exception as e:
        # Catch errors from bad individuals
        print(f"Error evaluating task ({i} vs {j}): {e}")
        return (i, j, 0, 0, 1) # Return 0 winnings, 1 hand (to avoid divide-by-zero)

# --- 1. Define Primitives and Terminals ---

INITIAL_MAX_TREE_HEIGHT = 15
INITIAL_MIN_TREE_HEIGHT = 5
MAX_TREE_HEIGHT = 90
VISUALIZE = False
RANK_ORDER = {
    None: 0., '2': 2., '3': 3., '4': 4., '5': 5., '6': 6., '7': 7., '8': 8., '9': 9., 'T': 10., 'J': 11., 'Q': 12., 'K': 13., 'A': 14.
}

# Define a protected division function to avoid ZeroDivisionError
def protected_div(left, right):
    """A protected division operator that returns 1.0 if the denominator is zero."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

# Define a conditional operator
def if_then_else(condition, out1, out2):
    """A conditional if-then-else operator."""
    return out1 if condition else out2

# Poker actions are mapped to a float:
# < 0.33 -> FOLD
# 0.33 to 0.66 -> CALL
# > 0.66 -> RAISE (the value can be used to scale the raise amount)
OUTPUT_TYPE = float

# The arguments for our evolved function. This defines the "senses" of our agent.
# We will map the gym's observation dictionary to these arguments.
ARGUMENT_TYPES = [
    float, # pot_size
    bool,  # button (True for small blind, False for big blind)
    float, # stack_size
    float, # opponent_stack_size
    float, # amount_to_call
    float, # hand_strength (a value from 0.0 to 1.0)
    str,   # hand_class (e.g., 'PAIR', 'FLUSH')
    str,   # street (e.g., 'PREFLOP', 'FLOP')

    float,        # 'board_highest_card',
    float,        # 'board_num_pairs',
    float,        # 'board_num_trips', 
    float,        # 'board_num_quads',
    float,        # 'board_num_same_suits',
    float,        # 'board_smallest_3_card_span',

    bool,        # 'hole_suited',
    float,        # 'hole_highest_card',
    bool,        # 'hole_paired',
    bool,        # 'hole_flush_draw',
    bool,        # 'hole_open_straight_draw',
    bool,        # 'hole_gutshot_straight_draw'

    str,        # 'preflop_opponent_line'
    str,         # 'flop_opponent_line'
    str,         # 'turn_opponent_line'
    str,         # 'river_opponent_line'

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

# Create the Primitive Set
# This defines the building blocks (functions and terminals) for our program trees.
pset = gp.PrimitiveSetTyped("main", ARGUMENT_TYPES, OUTPUT_TYPE)

# --- Add Primitives (Operators) ---
# Logical operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
# pset.addPrimitive(operator.xor_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# Comparison operators for floats
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(operator.ne, [float, float], bool)

# Comparison operators for strings (hand_class, street)
pset.addPrimitive(operator.eq, [str, str], bool)
pset.addPrimitive(operator.ne, [str, str], bool)

# Arithmetic operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)

# Conditional operators
pset.addPrimitive(if_then_else, [bool, OUTPUT_TYPE, OUTPUT_TYPE], OUTPUT_TYPE)
pset.addPrimitive(if_then_else, [bool, str, str], str, name="if_then_else_str")


# --- Add Terminals (Constants and Inputs) ---
# Rename arguments for clarity
pset.renameArguments(
    ARG0='pot_size', 
    ARG1='button', 
    ARG2='stack_size',
    ARG3='opponent_stack_size', 
    ARG4='amount_to_call',
    ARG5='hand_strength', 
    ARG6='hand_class', 
    ARG7='street',

    ARG8='board_highest_card',
    ARG9='board_num_pairs',
    ARG10='board_num_trips', 
    ARG11='board_num_quads',
    ARG12='board_num_same_suits',
    ARG13='board_smallest_3_card_span',

    ARG14='hole_suited',
    ARG15='hole_highest_card',
    ARG16='hole_paired',
    ARG17='hole_flush_draw',
    ARG18='hole_open_straight_draw',
    ARG19='hole_gutshot_straight_draw',

    ARG20='preflop_opponent_line',
    ARG21='flop_opponent_line',
    ARG22='turn_opponent_line',
    ARG23='river_opponent_line',

    ARG24='num_hands',
    ARG25='VPIP',
    ARG26='PFR_PCT',
    ARG27='THREEBET_PCT',
    ARG28='WTSD',
    ARG29='WSD',
    ARG30='WWSF',
    ARG31='AF',
    ARG32='CBET_PCT',
    ARG33='DONK_PCT',
    ARG34='CHECKRAISE_PCT',
)

# Add float constants
FLOAT_CONSTANTS = [-1.0, 0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
for val in FLOAT_CONSTANTS:
    pset.addTerminal(val, float)

# Add string constants for hand classifications
HAND_CLASSES = ['HIGH_CARD', 'PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT',
                'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH']
for hand_class in HAND_CLASSES:
    pset.addTerminal(hand_class, str)

# Add string constants for streets
STREETS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
for street in STREETS:
    pset.addTerminal(street, str)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# betting line strings
    # This seems a bit much, let's simplify
# LINES = ['FOLD', "DONK_BET", "BET", "RAISE", "CHECK", "CALL", "NONE", "THREE_BET", "FOUR_BET"]
# for line in LINES:
#     pset.addTerminal(line, str)
# # Add all joined combinations of 2 and three actions
# # some might be nonsense, #TODO: prune nonsense lines
# # some weird deep raise lines might not be captured
# TWO_ACTION_LINES = list(combinations(LINES, 2))
# for line in TWO_ACTION_LINES:
#     if 'FOLD' == line[0] or 'CALL' == line[0] or 'NONE' in line:
#         continue
#     pset.addTerminal('-'.join(line), str)

# THREE_ACTION_LINES = list(combinations(LINES, 3))
# for line in THREE_ACTION_LINES:
#     if 'FOLD' == line[0] or 'FOLD' == line[1] or 'CALL' == line[0] or 'CALL' == line[1] or 'NONE' in line:
#         continue
#     pset.addTerminal('-'.join(line), str)
LINES = ["DONK_BET", "BET", "RAISE", "CHECK", "CALL", "THREE_BET", "FOUR_BET", 
         "DONK_BET-CALL", "BET-CALL", "CHECK-RAISE", "RAISE-CALL", "CALL-CALL"]
for line in LINES:
    pset.addTerminal(line, str)

# --- 2. DEAP Toolbox Setup ---

# Define the fitness criteria: maximize winnings
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define the individual: a program tree with the defined fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator:
# We use genHalfAndHalf to create a mix of full trees and growing trees,
# which promotes diversity in the initial population.
toolbox.register("expr", gp.genFull, pset=pset, min_=INITIAL_MIN_TREE_HEIGHT, max_=INITIAL_MAX_TREE_HEIGHT)

# Structure initializers:
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# --- 3. Fitness Evaluation ---

# Helper function to map gym observations to our function's arguments
HAND_STRENGTH_MAP = {i: strength for i, strength in enumerate(np.linspace(0, 1, len(HAND_CLASSES)))}
STREET_MAP = {0: 'PREFLOP', 1: 'FLOP', 2: 'TURN', 3: 'RIVER'}


# Define a simple Card namedtuple for clarity and ease of use.
# Assumes ranks are integers: 2-10, J=11, Q=12, K=13, A=14.
# Assumes suits are strings: 's' (spades), 'h' (hearts), 'd' (diamonds), 'c' (clubs).
# Card = namedtuple('Card', ['rank', 'suit'])

def has_flush_draw(hole_cards, community_cards):
    """
    Checks if there are exactly 4 cards of the same suit among the combined
    hole and community cards.

    Args:
        hole_cards (list[Card]): A list of 2 Card objects for the player.
        community_cards (list[Card]): A list of 3 to 5 Card objects.

    Returns:
        bool: True if a flush draw exists, False otherwise.
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) < 4:
        return False
        
    suit_counts = Counter(card.suit for card in all_cards)
    
    # A flush draw is defined as having 4 cards of the same suit.
    # We check for a count of 4. A count of 5 or more is a made flush.
    return 4 in suit_counts.values()

def has_straight_draw(hole_cards, community_cards):
    """
    Checks for an open-ended straight draw (four contiguous cards).
    For example, holding 5,6 with a flop of 7,8,K.

    Args:
        hole_cards (list[Card]): A list of 2 Card objects for the player.
        community_cards (list[Card]): A list of 3 to 5 Card objects.

    Returns:
        bool: True if an open-ended straight draw exists, False otherwise.
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) < 4:
        return False

    # Get unique ranks to handle pairs correctly.
    ranks = sorted(list(set(RANK_ORDER[card.rank] for card in all_cards)))

    # Add Ace as a low card (rank 1) for A-2-3-4-5 straights
    if 14 in ranks: # 14 is the rank for Ace
        ranks.insert(0, 1)

    if len(ranks) < 4:
        return False

    # Check all 4-card combinations for a contiguous sequence.
    # A sequence of 4 cards is contiguous if the difference between the
    # highest and lowest card is exactly 3.
    for combo in combinations(ranks, 4):
        if (max(combo) - min(combo)) == 3:
            return True
            
    return False

#TODO: fix count logic
def count_straight_draws(hole_cards, community_cards):
    """
    Counts the number of open-ended (contiguous) and gapped (gutshot)
    straight draws.

    - Contiguous (Open-ended): 4 cards in a sequence, e.g., 5-6-7-8.
    - Gapped (Gutshot): 4 cards that span 5 ranks, e.g., 5-6-8-9.

    Args:
        hole_cards (list[Card]): A list of 2 Card objects for the player.
        community_cards (list[Card]): A list of 3 to 5 Card objects.

    Returns:
        tuple[int, int]: A tuple containing (contiguous_draw_count, gapped_draw_count).
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) < 4:
        return (0, 0)

    ranks = sorted(list(set(RANK_ORDER[card.rank] for card in all_cards)))

    if 14 in ranks:
        ranks.insert(0, 1)

    if len(ranks) < 4:
        return (0, 0)

    contiguous_count = 0
    gapped_count = 0

    # Check every possible combination of 4 unique ranks.
    for combo in combinations(ranks, 4):
        rank_span = max(combo) - min(combo)
        
        # 4 ranks spanning 4 values is a contiguous draw (e.g., 8,7,6,5 -> 8-5=3)
        if rank_span == 3:
            contiguous_count += 1
        # 4 ranks spanning 5 values is a gapped draw (e.g., 9,8,6,5 -> 9-5=4)
        elif rank_span == 4:
            gapped_count += 1
            
    return contiguous_count, gapped_count


def find_highest_card(community_cards):
    highest_rank = 0
    for card in community_cards:
        if RANK_ORDER[card.rank] > highest_rank:
            highest_rank = RANK_ORDER[card.rank]
    return highest_rank

def num_suited(community_cards):

    all_cards = community_cards
    if len(all_cards) < 3:
        return 0.0
    
    suit_counts = Counter(card.suit for card in all_cards)
    
    # A flush draw is defined as having 4 cards of the same suit.
    # We check for a count of 4. A count of 5 or more is a made flush.
    return max(suit_counts.values())

def num_pairs_trips_quads(community_cards):
    rank_counts = Counter(card.rank for card in community_cards)
    num_pairs = sum(1 for rank_count in rank_counts.values() if rank_count >= 2)
    num_trips = sum(1 for rank_count in rank_counts.values() if rank_count >= 3)
    num_quads = sum(1 for rank_count in rank_counts.values() if rank_count >= 4)
    return num_pairs, num_trips, num_quads


def find_smallest_3_card_span(community_cards):
    all_cards = community_cards
    if len(all_cards) < 3:
        return 0.0
    
    ranks = sorted(list(set(RANK_ORDER[card.rank] for card in all_cards)))

    if 14 in ranks:
        ranks.insert(0, 1)

    min_span = 20.0

    # Check every possible combination of 4 unique ranks.
    for combo in combinations(ranks, 3):
        rank_span = max(combo) - min(combo)
        if rank_span < min_span:
            min_span = rank_span
    return min_span


class ASTAgent(BaseAgent):
    def __init__(self, dealer: Any, seat_id: int, ast: Any, **kwargs):
        self.dealer = dealer
        self.ast = ast

        self.player_id = seat_id
        self.opponent_id = 1 - self.player_id
        self.button = None
        self.final_observation = {}
        
        # This will store the history indices for each street
        # e.g., {'PREFLOP': {'start': 0, 'end': 4}, 'FLOP': {'start': 4, 'end': 6}}
        self.street_boundaries = {}
        
        # This will store the final parsed lines for both players
        # e.g., {0: {'FLOP': 'CHECK-RAISE'}, 1: {'FLOP': 'BET-CALL'}}
        self.lines_by_street = {0: {}, 1: {}}
        
        # Who was the last aggressor in the preflop action?
        self.preflop_aggressor = None

        # --- NEW: Opponent Stats Storage ---
        # This dictionary holds the raw counts for all opponent stats.
        self.opponent_stats = {
            'num_hands': 0.0,
            
            # VPIP (Voluntarily Put In Pot)
            'vpip_hands': 0,       # Numerator
            # 'num_hands' is denominator
            
            # PFR (Preflop Raise)
            'pfr_hands': 0,        # Numerator
            # 'num_hands' is denominator
            
            # 3BET
            '3bet_hands': 0,       # Numerator
            '3bet_opportunities': 0, # Denominator (Faced a 2-bet)
            
            # WTSD (Went To Showdown)
            'wtsd_hands': 0,       # Numerator
            # 'num_hands' is denominator
            
            # W$SD (Won at Showdown)
            'wtsd_win_hands': 0,   # Numerator
            # 'wtsd_hands' is denominator
            
            # WWSF (Won When Saw Flop)
            'wwsf_hands': 0,       # Numerator
            'saw_flop_hands': 0,     # Denominator
            
            # AF (Aggression Factor)
            'agg_bets': 0,
            'agg_raises': 0,
            'agg_calls': 0,
            # AF = (bets + raises) / calls
            
            # CBET% (Continuation Bet)
            'cbet_hands': 0,
            'cbet_opportunities': 0, # Was PFA & saw flop
            
            # DONK%
            'donk_hands': 0,
            'donk_opportunities': 0, # Was OOP, not PFA, & saw flop
            
            # CHECKRAISE%
            'checkraise_hands': 0,
            'checkraise_opportunities': 0 # Had a chance to check
        }
        
        # --- Per-hand parsed data ---
        # This is reset on each hand by parse_betting_history
        self.preflop_flags = {}
        self.action_counts_by_street = {}

    def get_street_from_cards(self, num_community_cards):
        """Helper to get the current street name."""
        if num_community_cards == 0:
            return 'PREFLOP'
        elif num_community_cards == 3:
            return 'FLOP'
        elif num_community_cards == 4:
            return 'TURN'
        elif num_community_cards == 5:
            return 'RIVER'
        return 'UNKNOWN'

    def get_previous_street(self, street):
            """Helper to get the previous street name."""
            street_order = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
            try:
                idx = street_order.index(street)
                return street_order[idx - 1] if idx > 0 else None
            except ValueError:
                return None
            
    def get_street_history(self, history, street):
        """Helper to get the slice of history for a specific street."""
        bounds = self.street_boundaries.get(street, {})
        start = bounds.get('start')
        if start is None:
            return []
        end = bounds.get('end', len(history))
        return history[start:end]

    def update_street_boundaries(self, street, history_len):
        """
        Called on every action to log the start/end of betting rounds
        in the flat history list.
        """
        if street not in self.street_boundaries:
            # This is the first action of a new street
            if street == 'PREFLOP':
                self.street_boundaries[street] = {'start': 0}
            else:
                self.street_boundaries[street] = {'start': history_len} if self.player_id != self.button else {'start': history_len - 1}
            
            # Find the previous street and set its 'end'
            prev_street = self.get_previous_street(street)
            if prev_street and prev_street in self.street_boundaries:
                if 'end' not in self.street_boundaries[prev_street]:
                    # player who acts second on each street will have one extra bet to record
                    self.street_boundaries[prev_street]['end'] = history_len if self.player_id != self.button else history_len - 1

    def _parse_street_line(self, street, street_history):
        """
        The core logic. Parses a slice of history for one street
        and returns the betting lines, aggressor, and action counts.
        """

        player_actions = {0: [], 1: []}
        action_counts = {0: Counter(), 1: Counter()}
        player_investment = {0: 0, 1: 0}
        current_bet_level = 0
        last_aggressor = None
        
        # --- Preflop Only State ---
        num_raises = 0
        preflop_flags = {
            'p0_faced_2bet': False, 'p1_faced_2bet': False,
            'p0_3bet': False, 'p1_3bet': False,
        }

        if street == 'PREFLOP':
            # player_investment = {0: 1, 1: 2} # SB=1, BB=2
            player_investment[self.button] = 1
            player_investment[1 - self.button] = 2
            current_bet_level = 2
            last_aggressor = 1 - self.button # BB is initial aggressor
            num_raises += 1
        
        first_to_act_post_flop = 1 - self.button
        is_donk_bet_opportunity = (
            street != 'PREFLOP' 
            # and
            # self.preflop_aggressor is not None and
            # first_to_act_post_flop != self.preflop_aggressor
        )
        actions_this_street = 0

        for (pos, bet, fold) in street_history:
            amount_to_call = current_bet_level - player_investment[pos]
            action_str = ""

            if fold:
                action_str = "FOLD"
            elif bet > current_bet_level:
                # This is a Bet or a Raise
                if amount_to_call == 0:
                    action_str = "BET"
                    num_raises += 1
                    if (is_donk_bet_opportunity and 
                        pos == first_to_act_post_flop and 
                        actions_this_street == 0):
                        action_str = "DONK_BET"
                else:
                    action_str = "RAISE"
                    # if street == 'PREFLOP':
                    num_raises += 1
                    if num_raises == 2: # This is a 2-bet
                        pass
                    elif num_raises == 3: # This is a 3-bet
                        action_str = "THREE_BET"
                        if street == 'PREFLOP':
                            preflop_flags[f'p{pos}_3bet'] = True
                            preflop_flags[f'p{1-pos}_faced_2bet'] = True
                    elif num_raises == 4: # This is a 4-bet
                        action_str = "FOUR_BET"

                current_bet_level = bet + player_investment[pos]
                last_aggressor = pos
            
            elif bet + player_investment[pos] == current_bet_level:
                if amount_to_call == 0:
                    action_str = "CHECK"
                else:
                    action_str = "CALL"
            
            player_investment[pos] = bet + player_investment[pos]
            player_actions[pos].append(action_str)
            if action_str == "THREE_BET" or action_str == "FOUR_BET":
                player_actions[pos] = [action_str]
            action_counts[pos][action_str] += 1
            actions_this_street += 1
        
        lines = {
            0: '-'.join(player_actions[0]),
            1: '-'.join(player_actions[1])
        }
        
        return lines, last_aggressor, preflop_flags, action_counts


    def parse_betting_history(self, history):
        """
        Orchestrates the parsing of the entire history by street.
        This is called *during* the hand by act() and *after* the hand
        by hand_complete().
        """
        # Reset per-hand state
        self.lines_by_street = {0: {}, 1: {}}
        self.preflop_aggressor = None
        self.preflop_flags = {}
        self.action_counts_by_street = {}
        
        street_slices = {}
        for street, bounds in self.street_boundaries.items():
            start = bounds['start']
            end = bounds.get('end', len(history)) 
            street_slices[street] = history[start:end]

        if 'PREFLOP' in street_slices:
            lines, agg, flags, counts = self._parse_street_line('PREFLOP', street_slices['PREFLOP'])
            self.lines_by_street[0]['PREFLOP'] = lines[0]
            self.lines_by_street[1]['PREFLOP'] = lines[1]
            self.preflop_aggressor = agg
            self.preflop_flags = flags
            self.action_counts_by_street['PREFLOP'] = counts

        for street in ['FLOP', 'TURN', 'RIVER']:
            if street in street_slices:
                lines, agg, _, counts = self._parse_street_line(street, street_slices[street])
                self.lines_by_street[0][street] = lines[0]
                self.lines_by_street[1][street] = lines[1]
                if agg is not None: # Track aggressor postflop
                    self.preflop_aggressor = agg
                self.action_counts_by_street[street] = counts
    
    def hand_complete(self, final_history, reward): #, final_observation):
        """
        !! NEW METHOD !!
        Call this at the END of each hand to update opponent stats.
        
        Args:
            final_history (list): The complete dealer.history for the hand.
            reward (float): The reward received by *this* player.
            final_observation (dict): The final observation for the hand.
        """

        # 1. Parse the final, complete history
        # This sets self.lines_by_street, self.preflop_flags, etc.
        self.parse_betting_history(final_history)
        
        # 2. Get opponent info and stats dict
        opp_id = self.opponent_id
        stats = self.opponent_stats # This is a reference, so we modify it directly
        
        stats['num_hands'] += 1
        
        # 3. Get key hand facts for the OPPONENT
        opponent_won_hand = reward[self.player_id] < 0
        opp_line_preflop = self.lines_by_street[opp_id].get('PREFLOP', '')
        player_line_preflop = self.lines_by_street[self.player_id].get('PREFLOP', '')
        opp_line_flop = self.lines_by_street[opp_id].get('FLOP', '')
        
        opponent_folded_preflop = 'FOLD' in opp_line_preflop or '' == opp_line_preflop or 'FOLD' in player_line_preflop or '' == player_line_preflop
        opponent_saw_flop = not opponent_folded_preflop
        
        final_street = self.get_street_from_cards(len(self.final_observation.get('community_cards', [])))
        last_action_was_fold = final_history and final_history[-1][2]

        # 4. Update stats
        
        # --- WTSD / W$SD ---
        if final_street == 'RIVER' and not last_action_was_fold:
            stats['wtsd_hands'] += 1
            if opponent_won_hand:
                stats['wtsd_win_hands'] += 1
        
        # --- WWSF ---
        if opponent_saw_flop:
            stats['saw_flop_hands'] += 1
            if opponent_won_hand:
                stats['wwsf_hands'] += 1
        
        # --- VPIP ---
        # VPIP = Voluntarily put in money preflop.
        # For SB: Any CALL or RAISE.
        # For BB: Any CALL or RAISE (CHECK is not voluntary).
        opp_vpip = False
        if 'BET' in opp_line_preflop or 'RAISE' in opp_line_preflop \
            or 'CALL' in opp_line_preflop or 'THREE_BET' in opp_line_preflop \
                or 'FOUR_BET' in opp_line_preflop:
            opp_vpip = True
        # if 'FOLD' in opp_line_preflop: pass
        # elif '' == opp_line_preflop: pass
        # elif opp_line_preflop == 'CHECK': pass # BB checked
        # elif opp_line_preflop == 'CHECK-FOLD': pass
        # else:
        #     opp_vpip = True # Any other line (CALL, RAISE, CALL-RAISE, etc.)
            
        if opp_vpip:
            stats['vpip_hands'] += 1
            
        # --- PFR ---
        if 'RAISE' in opp_line_preflop or 'THREE_BET' in opp_line_preflop or 'FOUR_BET' in opp_line_preflop:
            stats['pfr_hands'] += 1
            
        # --- 3BET ---
        if self.preflop_flags.get(f'p{opp_id}_faced_2bet', False):
            stats['3bet_opportunities'] += 1
        if self.preflop_flags.get(f'p{opp_id}_3bet', False):
            stats['3bet_hands'] += 1
            
        # --- AF, CBET, DONK, CHECKRAISE (Postflop) ---
        opp_was_pfa = self.preflop_aggressor == opp_id
        
        for street in ['FLOP', 'TURN', 'RIVER']:
            opp_actions = self.action_counts_by_street.get(street, {}).get(opp_id, Counter())
            if not opp_actions: # Hand ended before this street
                break
                
            # AF (Aggression Factor)
            stats['agg_bets'] += opp_actions['BET'] + opp_actions['DONK_BET']
            stats['agg_raises'] += opp_actions['RAISE'] + opp_actions['THREE_BET'] + opp_actions['FOUR_BET']
            stats['agg_calls'] += opp_actions['CALL']
            
            # Check-Raise
            if opp_actions['CHECK'] > 0:
                stats['checkraise_opportunities'] += 1
                if opp_actions['RAISE'] > 0:
                    stats['checkraise_hands'] += 1
            
            # CBET / DONK (Flop only)
            if street == 'FLOP' and opponent_saw_flop:
                if opp_was_pfa:
                    stats['cbet_opportunities'] += 1
                    if opp_actions['BET'] > 0 or opp_actions['DONK_BET'] > 0:
                        stats['cbet_hands'] += 1
                else:
                    opp_is_oop = opp_id != self.button # BB is OOP postflop
                    if opp_is_oop:
                        stats['donk_opportunities'] += 1
                        if opp_actions['DONK_BET'] > 0:
                            stats['donk_hands'] += 1
            
            if opp_actions['FOLD'] > 0:
                break # Opponent folded, stop processing further streets
                
        # 5. Reset street boundaries for the next hand
        self.street_boundaries = {}

    def get_opponent_stats(self):
        """
        !! NEW METHOD !!
        Calculates percentages from raw counts to be fed to the agent.
        """
        s = self.opponent_stats
        
        def safe_div(num, den):
            return (num / den) if den > 0 else 0.0
            
        return {
            'num_hands': s['num_hands'],
            'VPIP': safe_div(s['vpip_hands'], s['num_hands']),
            'PFR': safe_div(s['pfr_hands'], s['num_hands']),
            '3BET': safe_div(s['3bet_hands'], s['3bet_opportunities']),
            'WTSD': safe_div(s['wtsd_hands'], s['num_hands']),
            'W$SD': safe_div(s['wtsd_win_hands'], s.get('wtsd_hands', 0)), # Use .get for safety
            'WWSF': safe_div(s['wwsf_hands'], s['saw_flop_hands']),
            'AF': safe_div(s['agg_bets'] + s['agg_raises'], s['agg_calls']),
            'CBET%': safe_div(s['cbet_hands'], s['cbet_opportunities']),
            'DONK%': safe_div(s['donk_hands'], s['donk_opportunities']),
            'CHECKRAISE%': safe_div(s['checkraise_hands'], s['checkraise_opportunities']),
        }

    # Value agent bets according to their hand strength
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        # collect ast inputs (pot_size, button, stack_size, opponent_stack_size, amount_to_call, hand_strength, hand_class, street)
        # mvp basics
        hole_cards = obs['hole_cards']
        community_cards = obs['community_cards']

        pot_size = obs['pot']
        button = self.dealer.button == self.player_id
        stack_size = obs['stacks'][self.player_id]
        opponent_stack_size = obs['stacks'][self.player_id - 1]
        amount_to_call = obs['call']
        hand_strength = self.dealer.evaluator.evaluate(hole_cards, community_cards) / 7462 # divide by max hand ranks
        hands_dict = self.dealer.evaluator.table.hand_dict
        
        if hand_strength < hands_dict['straight flush']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-1]
        elif hand_strength < hands_dict['four of a kind']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-2]
        elif hand_strength < hands_dict['full house']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-3]
        elif hand_strength < hands_dict['flush']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-4]
        elif hand_strength < hands_dict['straight']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-5]
        elif hand_strength < hands_dict['three of a kind']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-6]
        elif hand_strength < hands_dict['two pair']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-7]
        elif hand_strength < hands_dict['pair']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-8]
        else: # High Card
            hand_class = HAND_CLASSES[-9]

        if len(obs['community_cards']) == 0:
            street = STREETS[0]
        elif len(obs['community_cards']) == 3:
            street = STREETS[1]
        elif len(obs['community_cards']) == 4:
            street = STREETS[2]
        else: # River - 5 community cards
            street = STREETS[3]

        # board texture
        board_highest_card = find_highest_card(community_cards=community_cards)
        board_num_same_suits = num_suited(community_cards=community_cards)
        board_num_pairs, board_num_trips, board_num_quads = num_pairs_trips_quads(community_cards=community_cards)
        board_smallest_3_card_span = find_smallest_3_card_span(community_cards=community_cards)

        # player's hand
        hole_suited = hole_cards[0].suit == hole_cards[1].suit
        hole_highest_card = max(RANK_ORDER[hole_cards[0].rank],RANK_ORDER[hole_cards[1].rank])
        hole_paired = hole_cards[0].rank == hole_cards[1].rank
        hole_flush_draw = has_flush_draw(hole_cards=hole_cards, community_cards=community_cards)
        open_straight_draws, gutshot_straight_draws = count_straight_draws(hole_cards=hole_cards, community_cards=community_cards)
        hole_open_straight_draw = open_straight_draws >= 1 or gutshot_straight_draws >= 2
        hole_gutshot_straight_draw = gutshot_straight_draws >= 1

        # betting line
        history = self.dealer.history
        if len(history) == 0:
            # we're preflop in a new hand on the button
            self.button = self.player_id
        elif len(history) == 1:
            # we're preflop in a new hand in the big blind
            self.button = self.opponent_id

        self.update_street_boundaries(street, len(history))
        
        # 2. Re-parse the entire history on every action
        # This ensures our state is always up-to-date.
        self.parse_betting_history(history)
        
        # 3. Get the opponent's line for the current street
        preflop_opponent_line = self.lines_by_street[self.opponent_id].get("PREFLOP", "NONE")
        flop_opponent_line = self.lines_by_street[self.opponent_id].get("FLOP", "NONE")
        turn_opponent_line = self.lines_by_street[self.opponent_id].get("TURN", "NONE")
        river_opponent_line = self.lines_by_street[self.opponent_id].get("RIVER", "NONE")
        
        # opponent's statistics
        # These stats are as-of the *end of the last hand*.

        self.final_observation = copy.deepcopy(obs)
        historical_stats = self.get_opponent_stats()


        # execute AST logic
        action = self.ast(
            pot_size, 
            button, 
            stack_size, 
            opponent_stack_size, 
            amount_to_call, 
            hand_strength, 
            hand_class, 
            street,

            board_highest_card,
            board_num_pairs,
            board_num_trips, 
            board_num_quads,
            board_num_same_suits,
            board_smallest_3_card_span,

            hole_suited,
            hole_highest_card,
            hole_paired,
            hole_flush_draw,
            hole_open_straight_draw,
            hole_gutshot_straight_draw,

            preflop_opponent_line,
            flop_opponent_line,
            turn_opponent_line,
            river_opponent_line,

            historical_stats['num_hands'],
            historical_stats['VPIP'],
            historical_stats['PFR'],
            historical_stats['3BET'],
            historical_stats['WTSD'],
            historical_stats['W$SD'],
            historical_stats['WWSF'],
            historical_stats['AF'],
            historical_stats['CBET%'],
            historical_stats['DONK%'],
            historical_stats['CHECKRAISE%'],

        )
        bet = action * pot_size

        return bet


def evaluate_agents(agent1_logic, agent2_logic, max_hands=500):
    """
    Simulates a heads-up poker match between two compiled agents.
    Returns the final winnings for each agent.
    """
    
    winnings = [0.0, 0.0]
    num_hands = 0

    # Instantiate heads up poker env
    env = PokerEnv(
        num_players=2,
        num_streets=4,
        blinds=[1,2],
        antes=0,
        raise_sizes='inf',
        num_raises=float('inf'),
        num_suits=4,
        num_ranks=13,
        num_hole_cards=2,
        num_community_cards=[0, 3, 1, 1],
        num_cards_for_hand=5,
        mandatory_num_hole_cards=0,
        start_stack=500,
        low_end_straight=True
    )

    # Instantiate agents
    player1 = agent1_logic
    player2 = agent2_logic
    if inspect.isclass(agent1_logic):
        player1 = agent1_logic(dealer=env.dealer, seat_id=0)
    else:
        player1 = ASTAgent(dealer=env.dealer, seat_id=0, ast=agent1_logic)
        
    if inspect.isclass(agent2_logic):
        player2 = agent2_logic(dealer=env.dealer, seat_id=1)
    else:
        player2 = ASTAgent(dealer=env.dealer, seat_id=1, ast=agent2_logic)

    env.register_agents([player1, player2])
    obs = env.reset()

    done = [False]
    game_over = False

    # Simulate poker game
    while True and num_hands < max_hands:

        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        if all(done):
            game_over = 0 in env.dealer.stacks
            num_hands += 1
            winnings[0] += rewards[0]
            winnings[1] += rewards[1]

            if isinstance(player1, ASTAgent):
                player1.hand_complete(env.dealer.history, rewards)
            if isinstance(player2, ASTAgent):
                player2.hand_complete(env.dealer.history, rewards)
            
            if game_over:
                break
            obs = env.reset()

    return winnings[0], winnings[1], num_hands


# Register the evaluation, selection, crossover, and mutation operators
toolbox.register("evaluate", evaluate_agents)
toolbox.register("select", tools.selBest) # Select the best X individuals
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=INITIAL_MIN_TREE_HEIGHT, max_=INITIAL_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorate crossover and mutation to prevent code bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))       # can we increase limit?
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))


# --- 4. The Main Evolutionary Loop ---

def main():
    random.seed(42)
    
    POP_SIZE = 100      # TIP: divisible by 4
    N_GEN = 500
    MAX_HANDS = 200
    EVAL_WITH_LEGACY_INDIVIDUALS = False
    SAVE_EVERY_X_GEN = 100
    VERSION_NUM = 0.3

    pop = toolbox.population(n=POP_SIZE)
    fossil_record = {}
    
    # Statistics to track
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + mstats.fields

    # print("--- Starting Evolution ---")

    # for gen in range(N_GEN):
    #     # --- Evaluate the entire population ---
    #     # Each individual plays against every other individual (round-robin)
    #     winnings_map = {i: 0 for i in range(len(pop))}
    #     num_hands_map = {i: 0 for i in range(len(pop))}
        
    #     for i in range(len(pop)):
    #         for j in range(i + 1, len(pop)):
    #             agent1_logic = toolbox.compile(expr=pop[i])
    #             agent2_logic = toolbox.compile(expr=pop[j])

    #             winnings1, winnings2, num_hands = toolbox.evaluate(agent1_logic, agent2_logic, max_hands=MAX_HANDS)
                
    #             winnings_map[i] += winnings1
    #             winnings_map[j] += winnings2
    #             num_hands_map[i] += num_hands
    #             num_hands_map[j] += num_hands

    #     # Each individual plays against our bench of simple and legacy agents
    #     bench = [RandomAgent, RandomAgent2, StationAgent, ManiacAgent, SimpleValueAgent]
    #     if EVAL_WITH_LEGACY_INDIVIDUALS:
    #         bench += [gen['individual'] for gen in fossil_record.values()]
    #     bench_size = len(bench)

    #     for i in range(len(pop)):
    #         for opponent in bench:
    #             # ready opponent
    #             if inspect.isclass(opponent):
    #                 winnings, opp_winnings, num_hands = toolbox.evaluate(agent1_logic, opponent, max_hands=MAX_HANDS)
    #             else:
    #                 opponent_logic = toolbox.compile(expr=opponent)
    #                 winnings, opp_winnings, num_hands = toolbox.evaluate(agent1_logic, opponent_logic, max_hands=MAX_HANDS)

    #             winnings_map[i] += winnings
    #             num_hands_map[i] += num_hands

    #     # Assign fitness based on total winnings
    #     for i, ind in enumerate(pop):
    #         # # Assign total winnings as fitness
    #         # ind.fitness.values = (winnings_map[i],)
    #         # Assign win rate as fitness
    #         ind.fitness.values = (winnings_map[i] / num_hands_map[i] ,)
    # --- Inside your main() function ---

    print("--- Starting Evolution ---")
    for gen in range(N_GEN):
        # # PROFILING MP CODE
        # import cProfile
        # import pstats
        # import io

        # profiler = cProfile.Profile()

        # # Enable profiling for a specific section
        # profiler.enable()


        print(f"\n--- Generation {gen}/{N_GEN} ---")
        
        # --- 1. Prepare all evaluation tasks ---
        tasks = []
        
        # --- Task Group 1: Round-Robin (Pop vs. Pop) ---
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                # We send the raw trees (pop[i]) and indices (i, j)
                tasks.append((list(pop[i]), list(pop[j]), MAX_HANDS, i, j))
                
        # --- Task Group 2: Benchmark (Pop vs. Bench) ---
        bench = [RandomAgent, RandomAgent2, StationAgent, ManiacAgent, SimpleValueAgent]
        if EVAL_WITH_LEGACY_INDIVIDUALS:
            bench += [gen_data['individual'] for gen_data in fossil_record.values()]
        bench_size = len(bench)
        
        for i in range(len(pop)):
            for opponent in bench:
                # We send the raw tree (pop[i]) and the opponent (class or tree)
                # We use 'None' as the 'j' index to mark it as a bench match
                tasks.append((pop[i], opponent, MAX_HANDS, i, None))
                
        
        print(f"Evaluating {len(tasks)} matches across {multiprocessing.cpu_count()} cores...")
        
        # --- 2. Run all tasks in parallel ---
        # pool.map distributes the 'tasks' list to the 'run_evaluation' function.
        # It waits until all tasks are complete.
        import time
        import sys
        start_time = time.time()
        results_iterator = pool.imap_unordered(run_evaluation, tasks)
        
        # # --- 3. Process results ---
        # print("Processing results...")
        winnings_map = {i: 0 for i in range(len(pop))}
        num_hands_map = {i: 0 for i in range(len(pop))}
        
        # for res in results:
        #     i, j, w1, w2, n_hands = res
            
        #     winnings_map[i] += w1
        #     num_hands_map[i] += n_hands
            
        #     if j is not None:
        #         # This was a pop vs. pop match, so update agent j
        #         winnings_map[j] += w2
        #         num_hands_map[j] += n_hands
        # --- 3. Process results from the iterator ---
        for task_count, res in enumerate(results_iterator):
            # This loop receives results as soon as a worker finishes one task
            i, j, w1, w2, n_hands = res
            
            # Update scores for individual i
            winnings_map[i] += w1
            num_hands_map[i] += n_hands
            
            # Update scores for individual j (if it was a pop vs. pop match)
            if j is not None:
                winnings_map[j] += w2
                num_hands_map[j] += n_hands
            
            # Progress update
            if (task_count + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (task_count + 1) / elapsed
                sys.stdout.write(f"\r  -> Completed {task_count + 1}/{len(tasks)} tasks | Rate: {rate:.2f} tasks/sec")
                sys.stdout.flush()

        print(f"\nAll {len(tasks)} tasks processed in {time.time() - start_time:.2f} seconds.")
                
        # --- 4. Assign fitness (your code, slightly modified) ---
        print("Assigning fitness...")
        for i, ind in enumerate(pop):
            if num_hands_map[i] > 0:
                ind.fitness.values = (winnings_map[i] / num_hands_map[i],)
            else:
                ind.fitness.values = (0,) # Avoid division by zero
                
        # ... (rest of your generation loop: selection, crossover, etc.) ...

        # --- Log statistics ---
        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop)-1 + bench_size, **record)
        print(logbook.stream)

        
        # # END PROFILE
        # profiler.disable()
        # s = io.StringIO()
        # sortby = pstats.SortKey.CUMULATIVE
        # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print("\n--- Profiling Results ---")
        # print(s.getvalue())
        # # profile_file = f"profile_{5}"
        # # with open(profile_file, 'w'):

        # --- Save the best of the generation ---
        best_ind = tools.selBest(pop, 1)[0]
        fossil_record[gen] = {
            'fitness': best_ind.fitness.values[0],
            'individual': best_ind
        }

        # --- Visualize generation's best individual ---
        visualize = False
        if visualize:
            print('Best Individual of Generation:')
            print(str(best_ind))

        # --- Selection ---
        num_survivors = POP_SIZE // 4
        survivors = toolbox.select(pop, k=num_survivors)
        
        # --- Create the next generation ---
        offspring1 = [toolbox.clone(ind) for ind in survivors]
        offspring2 = [toolbox.clone(ind) for ind in survivors]
        offspring3 = [toolbox.clone(ind) for ind in survivors]
        
        # Apply crossover
        from itertools import zip_longest

        for child1, child2 in zip_longest(offspring2[::2], offspring2[1::2], fillvalue=None):
            if child1 is None: 
                toolbox.mutate(child2)
                del child2.fitness.values
            elif child2 is None:
                toolbox.mutate(child1)
                del child1.fitness.values
            else:
                # if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring2:
            # if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Introduce new organisms
        offspring3 = [toolbox.individual() for _ in range(len(survivors))]

        # The new population is the survivors and their offspring
        pop[:] = survivors + offspring1 + offspring2 + offspring3

        # Save
        if gen % SAVE_EVERY_X_GEN == 0:
            # Save fossil record
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_id = f"fossils_v{VERSION_NUM}_gen{gen}_{cur_time}"
            save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
            with open(save_path, 'wb') as f:
                pickle.dump(fossil_record, f)

    print("--- Evolution Finished ---")

    # --- Save, Print, and Plot Results ---
    # Save fossil record
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_id = f"fossils_v{VERSION_NUM}_gen{gen_nums}_{cur_time}"
    save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
    with open(save_path, 'wb') as f:
        pickle.dump(fossil_record, f)

    print("\n--- Fossil Record (Best of Each Generation) ---")
    for gen, data in fossil_record.items():
        print(f"Gen {gen}: Fitness = {data['fitness']:.2f}")
        print(f"  Code: {str(data['individual'])}") # Uncomment to see the evolved code

    # Plotting
    gen_nums = logbook.select("gen")
    avg_fitness = logbook.chapters["fitness"].select("avg")
    max_fitness = logbook.chapters["fitness"].select("max")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen_nums, avg_fitness, "b-", label="Average Fitness")
    line2 = ax1.plot(gen_nums, max_fitness, "r-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Total Winnings (Fitness)")
    ax1.set_title("Poker Agent Fitness Over Generations")
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    
    plt.grid(True)
    # plt.show()
    fig_save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.png" # Windows file path format
    plt.savefig(fig_save_path)


if __name__ == "__main__":
    # Create the pool once, globally.
    # This will use all available CPU cores.
    NUM_PROCS = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=NUM_PROCS)
    
    # Make the pool and toolbox globally accessible to the wrapper function
    # Note: This often happens by default, but it's good to be explicit
    # if you run into issues.
    globals()['pool'] = pool
    globals()['toolbox'] = toolbox # Ensure toolbox is global

    try:
        main()
    except KeyboardInterrupt:
        print("Evolution stopped by user.")
    finally:
        # Clean up the worker processes
        pool.close()
        pool.join()
        print("--- Evolution Complete ---")
    # main()
