from clubs_gym.agent.base import BaseAgent
import clubs
import random


class RandomAgent(BaseAgent):
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.3334:
            return 0
        elif prob < 0.6667:
            return 10
        else:
            return 100


class RandomAgent2(BaseAgent):
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.5:
            return 0
        elif prob < 0.75:
            return 10
        else:
            return 100