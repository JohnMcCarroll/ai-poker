from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

from chipzen import Action, Bot, GameState

# from poker_env import PokerEnv  # Replace with the module containing PokerEnv & PokerDealer

from typing import List, Union, Literal, Optional, Tuple
from clubs.poker.engine import Dealer, ObservationDict
from clubs_gym.envs import ClubsEnv
from clubs import error


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



_STREET_TO_PHASE = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
}

_RANK_MAP = {
    14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
    9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2",
    "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T", "10": "T",
    "9": "9", "8": "8", "7": "7", "6": "6", "5": "5", "4": "4", "3": "3", "2": "2",
}

_SUIT_MAP = {
    "s": "s", "spades": "s", "♠": "s",
    "h": "h", "hearts": "h", "♥": "h",
    "d": "d", "diamonds": "d", "♦": "d",
    "c": "c", "clubs": "c", "♣": "c",
}


def _format_card(card: Any) -> str:
    """Convert a clubs Card (or string) into Chipzen's 2-char notation like 'As' or 'Td'."""
    if isinstance(card, str):
        return card
    rank = getattr(card, "rank", None)
    suit = getattr(card, "suit", None)
    if rank is not None and suit is not None:
        r_str = _RANK_MAP.get(rank, _RANK_MAP.get(str(rank).upper(), str(rank)))
        s_str = _SUIT_MAP.get(suit, _SUIT_MAP.get(str(suit).lower(), str(suit)))
        return f"{r_str}{s_str}"
    text = str(card).strip()
    # Fallback if str(card) contains a 2-char token or suit symbol
    if len(text) == 2:
        return text
    for sym, code in _SUIT_MAP.items():
        if sym in text and len(text) >= 2:
            r_part = _RANK_MAP.get(text[0].upper(), text[0])
            return f"{r_part}{code}"
    return text


def _build_game_state(obs: Dict[str, Any], dealer: Any) -> GameState:
    """Translate a clubs ObservationDict into a chipzen GameState."""
    seat: int = obs["action"]
    raw_hole = obs.get("hole_cards", [])
    if raw_hole and isinstance(raw_hole[0], (list, tuple)):
        seat_cards = raw_hole[seat] if seat < len(raw_hole) else []
    else:
        seat_cards = raw_hole
    hole_cards = [_format_card(c) for c in seat_cards]
    board = [_format_card(c) for c in obs.get("community_cards", [])]

    street = getattr(dealer, "street", 0)
    phase = _STREET_TO_PHASE.get(street, f"street_{street}")

    call_amt = int(obs.get("call", 0))
    min_raise = int(obs.get("min_raise", 0))
    max_raise = int(obs.get("max_raise", 0))
    pot = int(obs.get("pot", 0))
    stacks = list(obs.get("stacks", []))
    street_commits = list(obs.get("street_commits", []))
    button = int(obs.get("button", 0))

    valid_actions: List[str] = ["fold"]
    if call_amt == 0:
        valid_actions.append("check")
    else:
        valid_actions.append("call")
    if max_raise > call_amt:
        valid_actions.append("raise" if call_amt > 0 else "bet")

    candidate_fields: Dict[str, Any] = {
        "hole_cards": hole_cards,
        "board": board,
        "community_cards": board,
        "phase": phase,
        "street": street,
        "pot": pot,
        "to_call": call_amt,
        "call_amount": call_amt,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "min_bet": min_raise,
        "max_bet": max_raise,
        "stack": stacks[seat] if seat < len(stacks) else 0,
        "stacks": stacks,
        "street_commits": street_commits,
        "seat": seat,
        "player_id": seat,
        "button": button,
        "valid_actions": valid_actions,
    }

    try:
        valid_names = {f.name for f in fields(GameState)}
        kwargs = {k: v for k, v in candidate_fields.items() if k in valid_names}
        return GameState(**kwargs)
    except TypeError:
        state = GameState()  # type: ignore[call-arg]
        for k, v in candidate_fields.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


class ChipzenPokerEnv:
    """Wraps PokerEnv / PokerDealer to expose a Chipzen-compatible interface."""

    def __init__(self, env: PokerEnv) -> None:
        self.env = env
        self.dealer = env.dealer
        self._last_obs: Optional[Dict[str, Any]] = None

    def reset(self) -> Tuple[Dict[str, Any], Optional[GameState]]:
        """Reset the underlying PokerEnv/dealer and return (raw_obs, active_game_state)."""
        obs = self.env.reset()
        self._last_obs = obs
        state = self.current_game_state()
        return obs, state

    def current_game_state(self) -> Optional[GameState]:
        """Return GameState for the currently acting seat, or None if hand is terminal."""
        if self._last_obs is None:
            return None
        seat = self._last_obs.get("action", -1)
        if seat is None or seat < 0:
            return None
        return _build_game_state(self._last_obs, self.dealer)

    def action_to_bet(self, action: Action, obs: Optional[Dict[str, Any]] = None) -> float:
        """Convert a chipzen Action into a bet value accepted by PokerDealer.step(bet)."""
        current_obs = obs if obs is not None else self._last_obs
        if current_obs is None:
            raise RuntimeError("Environment has not been reset yet.")

        call_amt = int(current_obs.get("call", 0))
        min_raise = int(current_obs.get("min_raise", call_amt))
        max_raise = int(current_obs.get("max_raise", call_amt))

        kind_raw = (
            getattr(action, "kind", None)
            or getattr(action, "action", None)
            or getattr(action, "type", None)
            or str(action)
        )
        kind = str(kind_raw).lower().removeprefix("actionkind.").removeprefix("action.")

        raw_amount = getattr(action, "amount", None)
        if raw_amount is None:
            raw_amount = getattr(action, "value", None)

        if kind == "fold":
            return -1.0 if call_amt > 0 else 0.0
        if kind == "check":
            return 0.0
        if kind == "call":
            return float(call_amt)
        if kind in {"raise", "bet", "all_in", "allin"}:
            if kind in {"all_in", "allin"}:
                return float(max_raise if max_raise > 0 else call_amt)
            if raw_amount is None:
                return float(max(min_raise, call_amt))
            target = float(raw_amount)
            if max_raise > call_amt:
                return float(max(min_raise, min(max_raise, target)))
            return float(call_amt)

        raise ValueError(f"Unsupported chipzen Action: {action!r}")

    def step_action(
        self, action: Action
    ) -> Tuple[Dict[str, Any], Optional[GameState], List[int], List[bool]]:
        """Apply a chipzen Action to the dealer and return (obs, next_state, payouts, done)."""
        bet = self.action_to_bet(action, self._last_obs)
        obs, payouts, done = self.dealer.step(bet)
        self._last_obs = obs
        next_state = None if all(done) else self.current_game_state()
        return obs, next_state, payouts, done

    def play_hand(self, bots: Dict[int, Bot]) -> Tuple[List[int], Dict[str, Any]]:
        """Play a single hand from start to finish across seat -> Bot mappings."""
        obs, state = self.reset()
        done = [False] * len(obs.get("stacks", []))
        payouts = [0] * len(obs.get("stacks", []))

        while not all(done):
            seat = obs.get("action", -1)
            if seat < 0 or state is None:
                break
            bot = bots[seat]
            action = bot.decide(state)
            obs, state, payouts, done = self.step_action(action)

        return payouts, obs


def make_heads_up_holdem_env(
    start_stack: int = 1000,
    small_blind: int = 1,
    big_blind: int = 3,
) -> ChipzenPokerEnv:
    """Create a heads-up No-Limit Texas Hold'em ChipzenPokerEnv."""
    env = PokerEnv(
        num_players=2,
        num_streets=4,
        blinds=[small_blind, big_blind],
        antes=0,
        raise_sizes="inf",
        num_raises="inf",
        num_suits=4,
        num_ranks=13,
        num_hole_cards=2,
        num_community_cards=[0, 3, 1, 1],
        num_cards_for_hand=5,
        mandatory_num_hole_cards=0,
        start_stack=start_stack,
        low_end_straight=True,
    )
    return ChipzenPokerEnv(env)
