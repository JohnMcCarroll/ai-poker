import time
from ai_poker.mvp.poker_env import PokerEnv
from ai_poker.mvp.visualization import CommandLineViewer
from ai_poker.mvp.agents import RandomAgent, RandomAgent2


if __name__ == '__main__':
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
    env.register_agents([RandomAgent(), RandomAgent2()])
    obs = env.reset()
    viz = CommandLineViewer(env.dealer, num_players=2)
    viz.update()
    viz.display()
    time.sleep(3)

    done = [False]
    game_over = False

    # Simulate poker game
    while True:

        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        hand_result = ""
        if all(done):
            winner = 1 if rewards[0] > 0 else 2
            hand_result = f"Player {winner} won {rewards[winner-1]}"
            game_over = 0 in env.dealer.stacks

        viz.update(hand_result=hand_result)
        viz.display()

        if game_over:
            break

        time.sleep(3)

        if all(done):
            obs = env.reset()


    # detect and declare winner of heads up match
    print("\n--- GAME OVER ---")
    print(f"Final Stacks: Player 1: {env.dealer.stacks[0]}, Player 2: {env.dealer.stacks[1]}")
