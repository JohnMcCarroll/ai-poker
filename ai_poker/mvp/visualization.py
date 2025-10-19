import os
from typing import List, Any


class CommandLineViewer:
    """
    A command-line viewer for a poker environment, updated to pull all necessary 
    information directly from the Dealer object.
    """
    def __init__(self, dealer: Any, num_players: int = 2):
        self.num_players = num_players
        self.dealer = dealer # Store the Dealer object reference
        
        # Initialize persistent state storage
        self.current_state = {
            'stacks': [0] * num_players,
            'bets': [0.0] * num_players,
            'pot': 0.0,
            'hole_cards': [[] for _ in range(num_players)], 
            'community_cards': [],
            'acting_player': 0,
            'hand_result': None
        }
        # Attempt to pull initial state from dealer
        self._update_dealer_info()


    def _update_dealer_info(self):
        """Pulls necessary game state attributes directly from the Dealer object."""
        try:
            # We assume the Dealer object has 'stacks' as this is crucial state info.
            # If this fails, the user must fetch stacks from the environment and pass them to update().
            self.current_state['stacks'] = getattr(self.dealer, 'stacks', self.current_state['stacks'])
            
            self.current_state['pot'] = getattr(self.dealer, 'pot', self.current_state['pot'])
            
            # 'street_commits' is the total bet by each player in the current street.
            self.current_state['bets'] = getattr(self.dealer, 'street_commits', self.current_state['bets'])

            # Hole cards are now accessible for all players from the Dealer
            self.current_state['hole_cards'] = getattr(self.dealer, 'hole_cards', self.current_state['hole_cards'])

            self.current_state['community_cards'] = getattr(self.dealer, 'community_cards', self.current_state['community_cards'])

            # Active players list (e.g., [False, True] means Player 2 is acting)
            self.current_state['acting_player'] = getattr(self.dealer, 'action', self.current_state['acting_player'])
            
        except AttributeError as e:
            print(f"ERROR: Dealer object is missing an expected attribute: {e}. Check dealer structure.")
        
    def update(self, hand_result: str = None):
        """
        Updates the internal state by pulling the latest data from the Dealer.

        Args:
            hand_result (str): Optional string describing the outcome (e.g., "Player 1 wins $500 with a Straight").
                               Should only be passed after the final step.
        """
        self._update_dealer_info()
        self.current_state['hand_result'] = hand_result
            
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
        for i in range(self.num_players):
            # Hole cards are retrieved from the persistent state
            hole_cards = self._format_cards(self.current_state['hole_cards'][i])
            
            # Safely retrieve stack and bet data
            stack = self.current_state['stacks'][i] if i < len(self.current_state['stacks']) else 0.0
            bet = self.current_state['bets'][i] if i < len(self.current_state['bets']) else 0.0
            is_active = self.current_state['acting_player'] == i
            
            active_marker = " <--- ACTING" if is_active else ""

            print(f"PLAYER {i + 1}{active_marker}")
            print(f"  Stack:      ${stack:,.2f}")
            print(f"  Commitment: ${bet:,.2f} (This Betting Street)")
            print(f"  Hole Cards: {hole_cards}")
            print("-" * 35)

        # Hand Winner Display
        if self.current_state['hand_result']:
            print("\n<<< HAND RESULT >>>")
            print(self.current_state['hand_result'])
            print("<<< HAND RESULT >>>")
            
        print("="*70)

    def _format_cards(self, cards: List[Any]) -> str:
        """Helper to format card representations, stripping object metadata."""
        if not cards:
            return "[--]"
        
        formatted_cards = []
        for c in cards:
            s = str(c)
            # Strips 'Card (3040834013072): ' from the string representation
            if ': ' in s:
                formatted_cards.append(s.split(': ', 1)[-1])
            else:
                formatted_cards.append(s) # Fallback
                
        return " ".join(formatted_cards) if formatted_cards else "[--]"
