from clubs_gym.agent.base import BaseAgent
import clubs
import random
from clubs_gym.envs import ClubsEnv
import time


# Visualization code
import os
from typing import Dict, List, Any

class CommandLineViewer:
    """
    A simple command-line viewer for a poker environment, designed to
    extract and display critical game state information from the observation.
    
    This version is specifically tailored to use the 'clubs' environment keys: 
    'stacks', 'pot', 'community_cards', 'hole_cards', and 'street_commits' (for bets).
    """
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        # Initialize persistent state storage
        self.current_state = {
            'stacks': [0] * num_players,
            'bets': [0.0] * num_players,
            'pot': 0.0,
            # Use list of lists for persistent card storage
            'hole_cards': [[] for _ in range(num_players)], 
            'community_cards': []
        }

    def update(self, observation: Dict[str, Any]):
        """
        Updates the internal state using the latest observation data keys.
        
        Args:
            observation (Dict): The full observation dictionary returned by env.step().
        """
        try:
            # --- REQUIRED DATA PARSING (Using confirmed keys) ---
            
            # Stacks (e.g., [498, 489])
            self.current_state['stacks'] = observation.get('stacks', self.current_state['stacks'])

            # Pot Size (e.g., 13)
            self.current_state['pot'] = observation.get('pot', self.current_state['pot'])

            # Current Bets (from 'street_commits')
            # 'street_commits' contains the total amount bet by each player in the current betting street.
            self.current_state['bets'] = observation.get('street_commits', self.current_state['bets'])

            # Community Cards
            self.current_state['community_cards'] = observation.get('community_cards', self.current_state['community_cards'])
            
            # Hole Cards (Agent-centric update)
            # The 'hole_cards' key provides the cards for the player whose observation this is (index 'action').
            if 'hole_cards' in observation:
                player_index = observation.get('action') 
                
                if player_index is not None and 0 <= player_index < self.num_players:
                    # Store the cards for the acting player persistently
                    cards_data = observation['hole_cards']
                    
                    # Ensure we store the cards as a list of objects
                    if isinstance(cards_data, list):
                        self.current_state['hole_cards'][player_index] = cards_data
                    elif cards_data is not None:
                        self.current_state['hole_cards'][player_index] = [cards_data]


        except Exception as e:
            print(f"ERROR: An error occurred during observation parsing: {e}")
            
    def display(self):
        """Prints the current game state to the console."""
        # Clear screen is important for a dynamic display
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
        for i in range(self.num_players):
            # Hole cards are retrieved from the persistent state
            hole_cards = self._format_cards(self.current_state['hole_cards'][i])
            
            # Safely retrieve stack and bet data
            stack = self.current_state['stacks'][i] if i < len(self.current_state['stacks']) else 0.0
            bet = self.current_state['bets'][i] if i < len(self.current_state['bets']) else 0.0

            # Assuming the agent is P1 (index 0) and opponent is P2 (index 1) for the display
            print(f"PLAYER {i + 1}")
            print(f"  Stack:      ${stack:,.2f}")
            print(f"  Commitment: ${bet:,.2f} (This Betting Street)")
            print(f"  Hole Cards: {hole_cards}")
            print("-" * 35)
            
        print("="*70)

    def _format_cards(self, cards: List[Any]) -> str:
        """Helper to format card representations, assuming Card objects use str() for display."""
        if not cards:
            return "[--]"
        
        # The Card objects in your environment look like 'Card (...): 7♠'. 
        # This logic strips the extra object metadata for a cleaner look.
        formatted_cards = []
        for c in cards:
            s = str(c)
            # Find the actual card representation (e.g., '7♠') after the colon and space.
            if ': ' in s:
                formatted_cards.append(s.split(': ', 1)[-1])
            else:
                formatted_cards.append(s) # Fallback
                
        return " ".join(formatted_cards) if formatted_cards else "[--]"


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
    # env.render(sleep=5)
    viz = CommandLineViewer(num_players=2)
    viz.update(obs)
    viz.display()
    time.sleep(2)

    # Simulate poker game
    while True:
        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        viz.update(obs)
        viz.display()

        # # env.render(sleep=5)
        # print('step')
        time.sleep(2)
        
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