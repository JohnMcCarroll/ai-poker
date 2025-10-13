from clubs_gym.agent.base import BaseAgent
import clubs
import random
from clubs_gym.envs import ClubsEnv
import time


# Define agent
class RandomAgent(BaseAgent):
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.3334:
            return 0
        elif prob < 0.6667:
            return 10
        else:
            return 100


if __name__ == '__main__':
    # Instantiate heads up poker env
    env = ClubsEnv(
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
    env.register_agents([RandomAgent()] * 2)
    obs = env.reset()
    env.render(sleep=10)

    # Simulate poker game
    while True:
        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        # env.render(sleep=5)
        time.sleep(5)
        
        # print("OBS:")
        # print(obs)
        # print()
        # print("ACTION")
        # print(bet)
        # print()
        # print("REWARDS")
        # print(rewards)
        # print()
        # print("INFO:")
        # print(info)
        # print()
        # print("DONE:")
        
        # print(done)

        if all(done):
            break

    # print(rewards)
    env.close()