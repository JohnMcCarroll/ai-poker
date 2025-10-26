from typing import List, Union, Literal, Optional, Tuple
from clubs.poker.engine import Dealer, ObservationDict
from clubs_gym.envs import ClubsEnv
from clubs import error, poker, render


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
            min_raise = self.big_blind + call
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
    
    def step(self, bet: float) -> Tuple[ObservationDict, List[int], List[bool]]:
        """Advances poker game to next player. If the bet is 0, it is
        either considered a check or fold, depending on the previous
        action. The given bet is always rounded to the closest valid bet
        size. When it is the same distance from two valid bet sizes
        the smaller bet size is used, e.g. if the min raise is 10 and
        the bet is 5, it is rounded down to 0.

        Parameters
        ----------
        bet : int
            number of chips bet by player currently active

        Returns
        -------
        Tuple[ObservationDict, List[int], List[bool]]
            observation dictionary, payouts for every player, boolean value for every
            player showing if that player is still active in the round

        Examples
        --------

        >>> dealer = Dealer(**configs.LEDUC_TWO_PLAYER)
        >>> obs = dealer.reset()
        >>> dealer.step(0)
        ... ({'action': 0,
        ...  'active': [True, True],
        ...  'button': 1,
        ...  'call': 0,
        ...  'community_cards': [],
        ...  'hole_cards': [[Card (139879188163600): A♥], [Card (139879188163504): A♠]],
        ...  'max_raise': 2,
        ...  'min_raise': 2,
        ...  'pot': 2,
        ...  'stacks': [9, 9],
        ...  'street_commits': [0, 0]},
        ...  [0, 0],
        ...  [False, False])
        """
        if self.action == -1:
            if any(self.active):
                done = self._done()
                payouts = self._payouts()
                observation = self._observation(all(done))
                return observation, payouts, done
            raise error.TableResetError("call reset() before calling first step()")

        fold = bet < 0
        bet = round(bet)

        call, min_raise, max_raise = self._bet_sizes()
        # round bet to nearest sizing
        bet = self._clean_bet(bet, call, min_raise, max_raise)

        # only fold if player cannot check
        if call and ((bet < call) or fold):
            self.active[self.action] = False
            bet = 0
            fold = True     # Added: UPDATE FOLD BOOL!

        # if bet is full raise record as largest raise
        if bet and (bet - call) >= self.largest_raise:
            self.largest_raise = bet - call
            self.street_raises += 1

        self._collect_bet(bet)

        self.history.append((self.action, int(bet), bool(fold)))

        self.street_option[self.action] = True
        self._move_action()

        # if all agreed go to next street
        if self._all_agreed():
            self.action = self.button
            self._move_action()
            # if at most 1 player active and not all in turn up all
            # community cards and evaluate hand
            while True:
                self.street += 1
                full_streets = self.street >= self.num_streets
                all_in = [
                    bool(active * (stack == 0))
                    for active, stack in zip(self.active, self.stacks)
                ]
                all_all_in = sum(self.active) - sum(all_in) <= 1
                if full_streets:
                    break
                self.community_cards += self.deck.draw(
                    self.num_community_cards[self.street]
                )
                if not all_all_in:
                    break
            self.street_commits = [0] * self.num_players
            self.street_option = [not active for active in self.active]
            self.street_raises = 0

        done = self._done()
        payouts = self._payouts()
        if all(done):
            self.action = -1
            self.pot = 0
            self.stacks = [
                stack + payout + pot_commit
                for stack, payout, pot_commit in zip(
                    self.stacks, payouts, self.pot_commits
                )
            ]
        observation = self._observation(all(done))
        return observation, payouts, done


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
