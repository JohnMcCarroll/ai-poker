from ai_poker.chipzen.training.ChipzenEnv import make_heads_up_holdem_env
from chipzen import Action, Bot, GameState


class CheckCallBot(Bot):
    def decide(self, state: GameState) -> Action:
        valid = getattr(state, "valid_actions", [])
        if "check" in valid:
            return Action.check()
        if "call" in valid:
            return Action.call()
        return Action.fold()


class AggroMinRaiseBot(Bot):
    def decide(self, state: GameState) -> Action:
        valid = getattr(state, "valid_actions", [])
        min_raise = getattr(state, "min_raise", 0)
        if ("raise" in valid or "bet" in valid) and min_raise > 0:
            if hasattr(Action, "raise_to"):
                return Action.raise_to(min_raise)
            if hasattr(Action, "raise_"):
                return Action.raise_(min_raise)
            return Action(kind="raise", amount=min_raise)  # type: ignore[call-arg]
        if "call" in valid:
            return Action.call()
        if "check" in valid:
            return Action.check()
        return Action.fold()


if __name__ == "__main__":
    cz_env = make_heads_up_holdem_env(start_stack=1000, small_blind=1, big_blind=3)
    bots = {0: CheckCallBot(), 1: AggroMinRaiseBot()}

    total_payouts = [0, 0]
    for hand_idx in range(100):
        payouts, final_obs = cz_env.play_hand(bots)
        for i, p in enumerate(payouts):
            total_payouts[i] += p

    print("100-hand payouts:", total_payouts)
