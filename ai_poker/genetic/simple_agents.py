from clubs_gym.agent.base import BaseAgent
from typing import Any
import clubs
import random


class ManiacAgent(BaseAgent):

    def __init__(self, **kwargs):
        pass

    # Maniac applies maximum pressure
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        max_bet = obs['max_raise']
        bet = int(max_bet * prob)
        return bet


class StationAgent(BaseAgent):

    def __init__(self, **kwargs):
        pass

    # Station plays extremely passive
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        bet = obs['call']
        return bet


class SimpleValueAgent(BaseAgent):
    def __init__(self, dealer: Any, seat_id: int, **kwargs):
        self.dealer = dealer
        self.position = seat_id

    # Value agent bets according to their hand strength
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        community_cards = obs['community_cards']
        hole_cards = obs['hole_cards']

        hands_dict = self.dealer.evaluator.table.hand_dict
        hand_strength = self.dealer.evaluator.evaluate(hole_cards, community_cards)
        pot = obs['pot']
        call = obs['call']

        # bet according to value
        if hand_strength < hands_dict['two pair']['cumulative unsuited']:
            return pot
        elif hand_strength < hands_dict['pair']['cumulative unsuited']:
            return max(0, call)
        else:
            return 0

