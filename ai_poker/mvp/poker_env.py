from typing import List, Union, Literal, Optional, Tuple
from clubs.poker.engine import Dealer
from clubs_gym.envs import ClubsEnv


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
