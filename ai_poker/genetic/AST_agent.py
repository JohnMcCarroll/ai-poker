from clubs_gym.agent.base import BaseAgent
from typing import Any
import clubs
from collections import Counter
from itertools import combinations
import copy
from ai_poker.genetic.constants import RANK_ORDER, HAND_CLASSES, STREETS


# DEFINE HELPER FUNCTIONS
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
            player_investment[self.button] = 1
            player_investment[1 - self.button] = 2
            current_bet_level = 2
            last_aggressor = 1 - self.button # BB is initial aggressor
            num_raises += 1
        
        first_to_act_post_flop = 1 - self.button
        is_donk_bet_opportunity = (
            street != 'PREFLOP' 
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
        hole_high_card = max(RANK_ORDER[hole_cards[0].rank],RANK_ORDER[hole_cards[1].rank])
        hole_low_card = min(RANK_ORDER[hole_cards[0].rank],RANK_ORDER[hole_cards[1].rank])
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
            hole_high_card,
            hole_low_card,
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
