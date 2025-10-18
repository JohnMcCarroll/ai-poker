from clubs_gym.agent.base import BaseAgent
import clubs
import random
from clubs_gym.envs import ClubsEnv
import time
import os
from typing import Dict, List, Any, Union, Literal, Optional, Tuple
from clubs.poker.engine import Dealer


class PokerDealer(Dealer):
    def _bet_sizes(self) -> Tuple[int, int, int]:
        # call difference between commit and maximum commit
        call = max(self.street_commits) - self.street_commits[self.action]
        # min raise at least largest previous raise
        # if limit game min and max raise equal to raise size
        raise_size = self.raise_sizes[self.street]
        if isinstance(raise_size, int):
            max_raise = min_raise = raise_size + call
        else:
            # TODO: add nuance to this - we want min raise to be big_blind on each street, but reraises should consider last bet size
            # min_raise = max(self.big_blind, self.largest_raise + call)
            min_raise = self.big_blind
            if raise_size == "pot":
                max_raise = self.pot + call * 2
            elif raise_size == float("inf"):
                max_raise = self.stacks[self.action]
        # if maximum number of raises in street
        # was reached cap raise at 0
        if self.street_raises >= self.num_raises[self.street]:
            min_raise = max_raise = 0
        # if last full raise was done by active player
        # (another player has raised less than minimum raise amount)
        # cap active players raise size to 0
        if self.street_raises and call < self.largest_raise:
            min_raise = max_raise = 0
        # clip bets to stack size
        call = min(call, self.stacks[self.action])
        min_raise = min(min_raise, self.stacks[self.action])
        max_raise = min(max_raise, self.stacks[self.action])
        return call, min_raise, max_raise


class PokerEnv(ClubsEnv):
    def __init__(
        self,
        num_players: int,
        num_streets: int,
        blinds: Union[int, List[int]],
        antes: Union[int, List[int]],
        raise_sizes: Union[
            int, Literal["pot", "inf"], List[Union[int, Literal["pot", "inf"]]]
        ],
        num_raises: Union[int, Literal["inf"], List[Union[int, Literal["inf"]]]],
        num_suits: int,
        num_ranks: int,
        num_hole_cards: int,
        num_community_cards: Union[int, List[int]],
        num_cards_for_hand: int,
        mandatory_num_hole_cards: int,
        start_stack: int,
        low_end_straight: bool = True,
        order: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            num_players=num_players,
            num_streets=num_streets,
            blinds=blinds,
            antes=antes,
            raise_sizes=raise_sizes,
            num_raises=num_raises,
            num_suits=num_suits,
            num_ranks=num_ranks,
            num_hole_cards=num_hole_cards,
            num_community_cards=num_community_cards,
            num_cards_for_hand=num_cards_for_hand,
            mandatory_num_hole_cards=mandatory_num_hole_cards,
            start_stack=start_stack,
            low_end_straight=low_end_straight
        )
        self.dealer = PokerDealer(
            num_players,
            num_streets,
            blinds,
            antes,
            raise_sizes,
            num_raises,
            num_suits,
            num_ranks,
            num_hole_cards,
            num_community_cards,
            num_cards_for_hand,
            mandatory_num_hole_cards,
            start_stack,
            low_end_straight,
            order,
        )


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
            print("<<< /HAND RESULT >>>")
            
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


class RandomAgent2(BaseAgent):
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        prob = random.random()
        if prob < 0.5:
            return 0
        elif prob < 0.75:
            return 10
        else:
            return 100


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

    # Simulate poker game
    while True:

        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        hand_result = ""
        if all(done):
            winner = 1 if rewards[0] > 0 else 2
            hand_result = f"Player {winner} won {rewards[winner-1]}"
            game_over = 0 in env.dealer.stacks
            obs = env.reset()

        viz.update(hand_result=hand_result)
        viz.display()

        if game_over:
            break

        time.sleep(3)
        
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


    # TODO: detect and declare winner of heads up match
    print("\n--- GAME OVER ---")
    print(f"Final Stacks: Player 1: {env.dealer.stacks[0]}, Player 2: {env.dealer.stacks[1]}")
    # time.sleep(10)