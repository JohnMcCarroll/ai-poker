import random
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from chipzen import Action, Bot, GameState

# Adjust these imports to match your project layout
from ChipzenEnv import ChipzenPokerEnv, make_heads_up_holdem_env


_PHASE_TO_NUM = {
    "preflop": 0.0,
    "flop": 0.33,
    "turn": 0.67,
    "river": 1.0,
}


def game_state_to_features(state: Optional[GameState], state_dim: int) -> np.ndarray:
    """Convert a Chipzen GameState into a fixed-size float32 feature vector."""
    if state is None:
        return np.zeros(state_dim, dtype=np.float32)

    pot = float(getattr(state, "pot", 0) or 0)
    to_call = float(
        getattr(state, "to_call", None)
        if getattr(state, "to_call", None) is not None
        else (getattr(state, "call_amount", 0) or 0)
    )
    min_raise = float(getattr(state, "min_raise", 0) or 0)
    max_raise = float(getattr(state, "max_raise", 0) or 0)
    stack = float(getattr(state, "stack", 0) or 0)
    phase = str(getattr(state, "phase", "preflop")).lower()
    phase_val = _PHASE_TO_NUM.get(phase, 0.5)

    hole = getattr(state, "hole_cards", None) or []
    board = getattr(state, "board", None) or getattr(state, "community_cards", None) or []
    denom = max(stack + pot, 1.0)

    raw_features = [
        to_call / denom,
        pot / denom,
        phase_val,
        min_raise / denom,
        max_raise / denom,
        stack / denom,
        len(hole) / 2.0,
        len(board) / 5.0,
    ]

    vec = np.zeros(state_dim, dtype=np.float32)
    n = min(state_dim, len(raw_features))
    vec[:n] = np.asarray(raw_features[:n], dtype=np.float32)
    return vec


def _make_raise_action(target_amount: int, min_raise: int, max_raise: int) -> Action:
    """Construct a raise/bet Action clamped to [min_raise, max_raise]."""
    clamped = max(min_raise, min(max_raise, int(round(target_amount))))
    if hasattr(Action, "raise_to"):
        return Action.raise_to(clamped)
    if hasattr(Action, "raise_"):
        return Action.raise_(clamped)
    return Action(kind="raise", amount=clamped)  # type: ignore[call-arg]


def action_index_to_chipzen(action_idx: int, state: GameState, action_dim: int = 3) -> Action:
    """Map a discrete policy action index onto a legal Chipzen Action for `state`."""
    valid = [str(a).lower() for a in (getattr(state, "valid_actions", None) or [])]
    to_call = int(
        getattr(state, "to_call", None)
        if getattr(state, "to_call", None) is not None
        else (getattr(state, "call_amount", 0) or 0)
    )
    min_raise = int(getattr(state, "min_raise", 0) or 0)
    max_raise = int(getattr(state, "max_raise", 0) or 0)
    pot = int(getattr(state, "pot", 0) or 0)

    can_raise = ("raise" in valid or "bet" in valid) and max_raise > to_call and min_raise > 0

    def passive_action() -> Action:
        if to_call == 0 and "check" in valid:
            return Action.check()
        if "call" in valid:
            return Action.call()
        if "check" in valid:
            return Action.check()
        return Action.fold()

    # Action 0: Fold when facing a bet; otherwise check/call passively
    if action_idx <= 0:
        if to_call > 0 and "fold" in valid:
            return Action.fold()
        return passive_action()

    # Action 1: Call / check
    if action_idx == 1:
        return passive_action()

    # Action 2+: Raise / bet sizing tiers
    if can_raise:
        if action_idx == 2 or action_dim <= 3:
            return _make_raise_action(min_raise, min_raise, max_raise)
        # Larger raise sizing for higher action indices
        target = min_raise + max(0, pot // 2)
        return _make_raise_action(target, min_raise, max_raise)

    return passive_action()


class ConfigurablePokerNet(nn.Module):
    """Feed-forward Q-network for discrete poker actions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ChipzenDQNBot(Bot):
    """Chipzen Bot backed by a ConfigurablePokerNet Q-network."""

    def __init__(
        self,
        model: nn.Module,
        state_dim: int,
        action_dim: int,
        use_epsilon: bool = False,
        epsilon: float = 0.0,
    ):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_epsilon = use_epsilon
        self.epsilon = epsilon
        self.last_state_vec: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None

    def decide_with_index(self, state: GameState) -> tuple[Action, int]:
        state_vec = game_state_to_features(state, self.state_dim)
        if self.use_epsilon and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
                q_values = self.model(state_tensor)
                action_idx = int(torch.argmax(q_values, dim=1).item())
            if was_training:
                self.model.train()

        action = action_index_to_chipzen(action_idx, state, self.action_dim)
        self.last_state_vec = state_vec
        self.last_action_idx = action_idx
        return action, action_idx

    def decide(self, state: GameState) -> Action:
        action, _ = self.decide_with_index(state)
        return action


class PokerSelfPlayTrainer:
    """Manages self-play training over ChipzenPokerEnv with replay and target snapshots."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        pool_size: int = 5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 128]

        self.current_agent = ConfigurablePokerNet(state_dim, action_dim, self.hidden_dims)
        self.target_agent = ConfigurablePokerNet(state_dim, action_dim, self.hidden_dims)
        self.update_target_network()

        self.optimizer = optim.Adam(self.current_agent.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.opponent_pool: deque[ConfigurablePokerNet] = deque(maxlen=pool_size)
        self._snapshot_to_pool()

        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def update_target_network(self) -> None:
        self.target_agent.load_state_dict(self.current_agent.state_dict())
        self.target_agent.eval()

    def _snapshot_to_pool(self) -> None:
        snapshot = ConfigurablePokerNet(self.state_dim, self.action_dim, self.hidden_dims)
        snapshot.load_state_dict(self.current_agent.state_dict())
        snapshot.eval()
        self.opponent_pool.append(snapshot)
        print(f"[Pool Update] Added model snapshot. Total pool size: {len(self.opponent_pool)}")

    def train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.from_numpy(np.asarray(next_states, dtype=np.float32))
        dones_t = torch.tensor(dones, dtype=torch.float32)

        self.current_agent.train()
        q_values = self.current_agent(states_t)
        state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_agent(next_states_t)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_values = rewards_t + (self.gamma * max_next_q_values * (1.0 - dones_t))

        loss = self.criterion(state_action_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_training_loop(
        self,
        num_episodes: int = 500,
        snapshot_interval: int = 50,
        target_update_interval: int = 25,
    ) -> None:
        cz_env: ChipzenPokerEnv = make_heads_up_holdem_env(
            start_stack=500,
            small_blind=1,
            big_blind=2,
        )

        for episode in range(1, num_episodes + 1):
            if random.random() < 0.5 or len(self.opponent_pool) == 0:
                opponent_model = self.current_agent
            else:
                opponent_model = random.choice(self.opponent_pool)

            agent_seat = random.randint(0, 1)
            opponent_seat = 1 - agent_seat

            train_bot = ChipzenDQNBot(
                model=self.current_agent,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                use_epsilon=True,
                epsilon=self.epsilon,
            )
            opp_bot = ChipzenDQNBot(
                model=opponent_model,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                use_epsilon=False,
                epsilon=0.0,
            )
            bots = {agent_seat: train_bot, opponent_seat: opp_bot}

            obs, state = cz_env.reset()
            done = [False] * len(obs.get("stacks", [True, True]))
            payouts = [0] * len(done)

            pending_state_vec: Optional[np.ndarray] = None
            pending_action_idx: Optional[int] = None

            while not all(done):
                acting_seat = obs.get("action", -1)
                if acting_seat < 0 or state is None:
                    break

                # If the training bot had a pending decision before the opponent acted,
                # close out that transition using the state observed when control returns.
                if pending_state_vec is not None and pending_action_idx is not None:
                    next_vec = game_state_to_features(state, self.state_dim)
                    self.memory.append(
                        (pending_state_vec, pending_action_idx, 0.0, next_vec, 0.0)
                    )
                    self.train_step()
                    pending_state_vec = None
                    pending_action_idx = None

                if acting_seat == agent_seat:
                    action, action_idx = train_bot.decide_with_index(state)
                    state_vec = train_bot.last_state_vec
                else:
                    action = opp_bot.decide(state)
                    action_idx = None
                    state_vec = None

                obs, next_state, payouts, done = cz_env.step_action(action)
                is_hand_done = float(all(done))

                if acting_seat == agent_seat and state_vec is not None and action_idx is not None:
                    if is_hand_done:
                        next_vec = game_state_to_features(next_state, self.state_dim)
                        reward = float(payouts[agent_seat]) if agent_seat < len(payouts) else 0.0
                        self.memory.append((state_vec, action_idx, reward, next_vec, 1.0))
                        self.train_step()
                    else:
                        pending_state_vec = state_vec
                        pending_action_idx = action_idx

                state = next_state

            # If the hand ended on the opponent's final action, finalize the training bot's last transition
            if pending_state_vec is not None and pending_action_idx is not None:
                next_vec = game_state_to_features(state, self.state_dim)
                reward = float(payouts[agent_seat]) if agent_seat < len(payouts) else 0.0
                self.memory.append((pending_state_vec, pending_action_idx, reward, next_vec, 1.0))
                self.train_step()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % target_update_interval == 0:
                self.update_target_network()

            if episode % snapshot_interval == 0:
                self._snapshot_to_pool()

            if 0 in cz_env.dealer.stacks:
                cz_env.env.reset(reset_stacks=True)


if __name__ == "__main__":
    trainer = PokerSelfPlayTrainer(state_dim=8, action_dim=3)
    trainer.run_training_loop()
