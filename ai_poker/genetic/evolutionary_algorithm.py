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


# --- 1. Define Primitives and Terminals ---

INITAL_MAX_TREE_HEIGHT = 5
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


)

# Add float constants
FLOAT_CONSTANTS = [0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
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
LINES = ['FOLD', "DONK_BET", "BET", "RAISE", "CHECK", "CALL", "NONE"]
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=INITAL_MAX_TREE_HEIGHT)

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
        self.hand_count = 0


        self.player_id = seat_id
        self.opponent_id = 1 - self.player_id
        
        # This will store the history indices for each street
        # e.g., {'PREFLOP': {'start': 0, 'end': 4}, 'FLOP': {'start': 4, 'end': 6}}
        self.street_boundaries = {}
        
        # This will store the final parsed lines for both players
        # e.g., {0: {'FLOP': 'CHECK-RAISE'}, 1: {'FLOP': 'BET-CALL'}}
        self.lines_by_street = {0: {}, 1: {}}
        
        # Who was the last aggressor in the preflop action?
        self.preflop_aggressor = None

    def get_previous_street(self, street):
            """Helper to get the previous street name."""
            street_order = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
            try:
                idx = street_order.index(street)
                return street_order[idx - 1] if idx > 0 else None
            except ValueError:
                return None
            
    def update_street_boundaries(self, street, history_len):
        """
        Called on every action to log the start/end of betting rounds
        in the flat history list.
        """
        if street not in self.street_boundaries:
            # This is the first action of a new street
            self.street_boundaries[street] = {'start': history_len}
            
            # Find the previous street and set its 'end'
            prev_street = self.get_previous_street(street)
            if prev_street and prev_street in self.street_boundaries:
                if 'end' not in self.street_boundaries[prev_street]:
                    self.street_boundaries[prev_street]['end'] = history_len

    def _parse_street_line(self, street, street_history):
        """
        The core logic. Parses a slice of history for one street
        and returns the betting lines for both players.
        """
        player_actions = {0: [], 1: []}
        player_investment = {0: 0, 1: 0}
        current_bet_level = 0
        last_aggressor = None
        
        # Preflop is special: it has blinds.
        if street == 'PREFLOP':
            player_investment = {0: 1, 1: 2} # SB=1, BB=2
            current_bet_level = 2
            last_aggressor = 1 # The Big Blind is the initial "aggressor"
        
        # Who is first to act on this street?
        # Postflop, SB (pos 0) is always first.
        # Preflop, SB (pos 0) is also first.
        first_to_act = 0
        
        # Check for a donk-bet opportunity
        # A donk-bet is when an out-of-position player
        # (who was NOT the preflop aggressor) bets first.
        is_donk_bet_opportunity = (
            street != 'PREFLOP' and
            self.preflop_aggressor is not None and
            first_to_act != self.preflop_aggressor
        )
        
        actions_this_street = 0

        for (pos, bet, fold) in street_history:
            amount_to_call = current_bet_level - player_investment[pos]
            action_str = "None"

            if fold:
                action_str = "FOLD"
            elif bet > current_bet_level:
                # This is a Bet or a Raise
                if amount_to_call == 0:
                    # No bet to call, so this is a "BET"
                    if (is_donk_bet_opportunity and 
                        pos == first_to_act and 
                        actions_this_street == 0):
                        action_str = "DONK_BET"
                    else:
                        action_str = "BET"
                else:
                    # There was a bet to call, so this is a "RAISE"
                    action_str = "RAISE"
                
                current_bet_level = bet
                last_aggressor = pos
            
            elif bet == current_bet_level:
                # This is a Check or a Call
                if amount_to_call == 0:
                    action_str = "CHECK"
                else:
                    action_str = "CALL"
            
            player_investment[pos] = bet
            player_actions[pos].append(action_str)
            actions_this_street += 1
        
        # Combine the action lists into a single string line
        lines = {
            0: '-'.join(player_actions[0]),
            1: '-'.join(player_actions[1])
        }
        
        return lines, last_aggressor

    def parse_betting_history(self, history):
        """
        Orchestrates the parsing of the entire history by street.
        """
        # Reset state
        self.lines_by_street = {0: {}, 1: {}}
        self.preflop_aggressor = None
        
        # Slice the flat history list into a dict of lists by street
        street_slices = {}
        for street, bounds in self.street_boundaries.items():
            start = bounds['start']
            # .get('end', len(history)) ensures we parse the current, in-progress street
            end = bounds.get('end', len(history)) 
            street_slices[street] = history[start:end]

        # Parse Preflop first to find the aggressor
        if 'PREFLOP' in street_slices:
            lines, aggressor = self._parse_street_line('PREFLOP', street_slices['PREFLOP'])
            self.lines_by_street[0]['PREFLOP'] = lines[0]
            self.lines_by_street[1]['PREFLOP'] = lines[1]
            self.preflop_aggressor = aggressor # Store for postflop parsing

        # Parse Postflop streets
        for street in ['FLOP', 'TURN', 'RIVER']:
            if street in street_slices:
                lines, _ = self._parse_street_line(street, street_slices[street])
                self.lines_by_street[0][street] = lines[0]
                self.lines_by_street[1][street] = lines[1]
    

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
        if len(history) <= 1:
            # we're preflop in a new hand
            self.hand_count += 1
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
            if game_over:
                break
            obs = env.reset()

    return winnings[0], winnings[1], num_hands


# Register the evaluation, selection, crossover, and mutation operators
toolbox.register("evaluate", evaluate_agents)
toolbox.register("select", tools.selBest) # Select the best X individuals
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=INITAL_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorate crossover and mutation to prevent code bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))       # can we increase limit?
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))


# --- 4. The Main Evolutionary Loop ---

def main():
    random.seed(42)
    
    POP_SIZE = 100
    N_GEN = 100
    CXPB, MUTPB = 0.7, 0.2
    
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

    print("--- Starting Evolution ---")

    for gen in range(N_GEN):
        # --- Evaluate the entire population ---
        # Each individual plays against every other individual (round-robin)
        winnings_map = {i: 0 for i in range(len(pop))}
        num_hands_map = {i: 0 for i in range(len(pop))}
        
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                agent1_logic = toolbox.compile(expr=pop[i])
                agent2_logic = toolbox.compile(expr=pop[j])

                winnings1, winnings2, num_hands = toolbox.evaluate(agent1_logic, agent2_logic, max_hands=100)
                
                winnings_map[i] += winnings1
                winnings_map[j] += winnings2
                num_hands_map[i] += num_hands
                num_hands_map[j] += num_hands

        # Each individual plays against our bench of simple and legacy agents
        bench = [RandomAgent, RandomAgent2, StationAgent, ManiacAgent, SimpleValueAgent]
        bench += [gen['individual'] for gen in fossil_record.values()]
        bench_size = len(bench)

        for i in range(len(pop)):
            for opponent in bench:
                # ready opponent
                if inspect.isclass(opponent):
                    winnings, opp_winnings, num_hands = toolbox.evaluate(agent1_logic, opponent, max_hands=100)
                else:
                    opponent_logic = toolbox.compile(expr=opponent)
                    winnings, opp_winnings, num_hands = toolbox.evaluate(agent1_logic, opponent_logic, max_hands=100)

                winnings_map[i] += winnings
                num_hands_map[i] += num_hands

        # Assign fitness based on total winnings
        for i, ind in enumerate(pop):
            # # Assign total winnings as fitness
            # ind.fitness.values = (winnings_map[i],)
            # Assign win rate as fitness
            ind.fitness.values = (winnings_map[i] / num_hands_map[i] ,)

        # --- Log statistics ---
        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop)-1 + bench_size, **record)
        print(logbook.stream)

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
        offspring3 = [gp.genHalfAndHalf(pset, min_=0, max_=INITAL_MAX_TREE_HEIGHT*2)]*len(survivors)

        # The new population is the survivors and their offspring
        pop[:] = survivors + offspring1 + offspring2 + offspring3

    print("--- Evolution Finished ---")

    # --- Save, Print, and Plot Results ---
    # Save fossil record
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_id = f"fossils_v0.1_{cur_time}"
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
    main()
