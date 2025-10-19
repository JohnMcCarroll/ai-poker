from clubs_gym.agent.base import BaseAgent
import clubs
import random


class ManiacAgent(BaseAgent):
    # Maniac applies maximum pressure
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.3334:
            return 0
        elif prob < 0.6667:
            return 10
        else:
            return 100


class StationAgent(BaseAgent):
    # Station plays extremely passive
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.3334:
            return 0
        elif prob < 0.6667:
            return 10
        else:
            return 100


class SimpleValueAgent(BaseAgent):
    # Value agent bets according to their hand strength
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.3334:
            return 0
        elif prob < 0.6667:
            return 10
        else:
            return 100
