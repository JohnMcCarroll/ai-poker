from ai_poker.mvp.poker_env import PokerEnv
from ai_poker.mvp.visualization import CommandLineViewer
from ai_poker.genetic.AST_agent import ASTAgent
from clubs_gym.agent.base import BaseAgent
import clubs
import os
import pickle
from ai_poker.genetic.evolutionary_algorithm import toolbox, creator


class HumanCommandLineViewer(CommandLineViewer):
    def display(self):
        """Prints the current game state to the console."""
        os.system('cls' if os.name == 'nt' else 'clear') 
        print("="*70)
        print("POKER GAME STATE VIEWER (Command Line) | Clubs Environment")
        print("="*70)
        
        # Pot and Community Cards
        print(f"Pot Size: ${self.current_state['pot']:,.2f}")
        comm_cards = self._format_cards(self.current_state['community_cards'])
        print(f"Community Cards ({len(self.current_state['community_cards'])}): {comm_cards}")
        print("-"*70)
        
        # Player Data
        dealer_pos = self.current_state.get('button', -1) # Get the dealer button position

        for i in range(self.num_players):
            # Hole cards are retrieved from the persistent state
            hole_cards = self._format_cards(self.current_state['hole_cards'][i])
            
            # Safely retrieve stack and bet data
            stack = self.current_state['stacks'][i] if i < len(self.current_state['stacks']) else 0.0
            bet = self.current_state['bets'][i] if i < len(self.current_state['bets']) else 0.0
            is_active = self.current_state['acting_player'] == i
            
            # --- MODIFICATION: Build player status string ---
            status_markers = []
            if i == dealer_pos:
                status_markers.append("(D)") # Add Dealer Button Icon
            if is_active:
                status_markers.append("<--- ACTING")
                
            active_marker = " ".join(status_markers)
            # --- END MODIFICATION ---

            print(f"PLAYER {i + 1} {active_marker}") # Display markers
            print(f"  Stack:      ${stack:,.2f}")
            print(f"  Commitment: ${bet:,.2f} (This Betting Street)")
            if i == 0:
                print(f"  Hole Cards: {hole_cards}")
            elif i == 1 and self.current_state['hand_result']:
                print(f"  Hole Cards: {hole_cards}")
            else:
                print(f"  Hole Cards: ?  ?")
            print("-" * 35)

        # Hand Winner Display
        if self.current_state['hand_result']:
            print("\n<<< HAND RESULT >>>")
            print(self.current_state['hand_result'])
            print("<<< HAND RESULT >>>")
            
        print("="*70)


class HumanAgent(BaseAgent):

    def __init__(self, **kwargs):
        pass

    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        call_amount = obs.get('call', 0)
        min_raise = obs.get('min_raise', call_amount * 2)

        while True:
            try:
                # Prompt the user
                user_input = input(f"\nEnter your total bet amount (Call={call_amount}, Min Raise={min_raise}): ")
                bet = int(user_input.strip())
                return bet

            except ValueError:
                print("Invalid input. Please enter a whole number (integer) for the bet amount.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Please try again.")


if __name__ == '__main__':
    # Load fossil
    file_name = "G:\\poker_bot\\ai-poker\\ai_poker\\genetic\\fossils\\evo_ckpt_v1.4_gen3_2025-11-19_18-58-00.pkl"
    generation_index = None

    with open(file_name, 'rb') as f:
        fossil_record = pickle.load(f)
        fossil_record = fossil_record['fossil_record']
        if generation_index is None:
            generation_index = max(fossil_record.keys())

    latest_ast = fossil_record[generation_index]['individual']
    
    deap_ind = creator.Individual(latest_ast)
    ast_code = toolbox.compile(expr=deap_ind)

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
    opponent = ASTAgent(env.dealer, 1, ast_code)
    env.register_agents([HumanAgent(), opponent])
    obs = env.reset()
    viz = HumanCommandLineViewer(env.dealer, num_players=2)
    viz.update()
    viz.display()

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

        if all(done):
            obs = env.reset()
            
            # --- MODIFICATION: Replace time.sleep with input() ---
            # time.sleep(5)
            input("\nHand complete. Press Enter to start the next hand...")
            # --- END MODIFICATION ---
            
            viz.update()
            viz.display()

    # detect and declare winner of heads up match
    print("\n--- GAME OVER ---")
    print(f"Final Stacks: Player 1: {env.dealer.stacks[0]}, Player 2: {env.dealer.stacks[1]}")
