import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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

    if action_idx <= 0:
        if to_call > 0 and "fold" in valid:
            return Action.fold()
        return passive_action()

    if action_idx == 1:
        return passive_action()

    if can_raise:
        if action_idx == 2 or action_dim <= 3:
            return _make_raise_action(min_raise, min_raise, max_raise)
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


def load_poker_net(checkpoint_path: str, map_location: str = "cpu") -> Tuple[ConfigurablePokerNet, Dict[str, Any]]:
    """Load a saved ConfigurablePokerNet checkpoint and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = ConfigurablePokerNet(
        input_dim=checkpoint["state_dim"],
        output_dim=checkpoint["action_dim"],
        hidden_dims=checkpoint.get("hidden_dims", [128, 128]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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

    def decide_with_index(self, state: GameState) -> Tuple[Action, int]:
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
    """Manages self-play training over ChipzenPokerEnv with stabilized DQN updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        pool_size: int = 5,
        max_grad_norm: float = 10.0,
        reward_clip_bb: Optional[float] = 50.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 128]
        self.max_grad_norm = max_grad_norm
        self.reward_clip_bb = reward_clip_bb

        self.current_agent = ConfigurablePokerNet(state_dim, action_dim, self.hidden_dims)
        self.target_agent = ConfigurablePokerNet(state_dim, action_dim, self.hidden_dims)
        self.update_target_network()

        self.optimizer = optim.Adam(self.current_agent.parameters(), lr=0.001)
        # 1. Huber / Smooth L1 loss damps outlier TD spikes better than squared MSE
        self.criterion = nn.SmoothL1Loss()

        self.opponent_pool: deque[ConfigurablePokerNet] = deque(maxlen=pool_size)
        self._snapshot_to_pool()

        self.memory: deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.history: Dict[str, List[float]] = {
            "episode": [],
            "reward_bb": [],
            "rolling_reward_bb": [],
            "win_rate": [],
            "loss": [],
            "epsilon": [],
        }

    def normalize_reward(self, raw_payout: float, big_blind: float) -> float:
        """Convert raw chip payout into big-blind units with optional clipping."""
        bb = max(float(big_blind), 1e-6)
        reward_bb = float(raw_payout) / bb
        if self.reward_clip_bb is not None:
            reward_bb = max(-self.reward_clip_bb, min(self.reward_clip_bb, reward_bb))
        return reward_bb

    def update_target_network(self) -> None:
        self.target_agent.load_state_dict(self.current_agent.state_dict())
        self.target_agent.eval()

    def _snapshot_to_pool(self) -> None:
        snapshot = ConfigurablePokerNet(self.state_dim, self.action_dim, self.hidden_dims)
        snapshot.load_state_dict(self.current_agent.state_dict())
        snapshot.eval()
        self.opponent_pool.append(snapshot)
        print(f"[Pool Update] Added model snapshot. Total pool size: {len(self.opponent_pool)}")

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

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
        # 2. Gradient norm clipping prevents runaway updates on noisy TD targets
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def save_model(self, output_path: str = "checkpoints/poker_dqn_final.pt") -> str:
        """Save the final trained neural network and training metadata."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        payload = {
            "model_state_dict": self.current_agent.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "max_grad_norm": self.max_grad_norm,
            "reward_clip_bb": self.reward_clip_bb,
            "epsilon": self.epsilon,
            "history": self.history,
        }
        torch.save(payload, output_path)
        print(f"[Saved] Final model checkpoint written to {output_path}")
        return output_path

    def plot_progress(self, output_path: str = "checkpoints/training_progress.png") -> Optional[str]:
        """Plot rolling reward (in BB), win rate, and SmoothL1 TD loss."""
        if not self.history["episode"]:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Viz] matplotlib not installed; skipping chart generation.")
            return None

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        episodes = self.history["episode"]

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        axes[0].plot(episodes, self.history["rolling_reward_bb"], label="Rolling Mean Reward (BB, 50 ep)", linewidth=2)
        axes[0].plot(episodes, self.history["win_rate"], label="Rolling Win Rate (50 ep)", linestyle="--")
        axes[0].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        axes[0].set_ylabel("BB / Win Rate")
        axes[0].set_title("Poker Self-Play DQN Progress (Stabilized)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, self.history["loss"], label="Episode Mean SmoothL1 Loss", color="tab:orange")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Smooth L1 Loss")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[Viz] Training progress chart saved to {output_path}")
        return output_path

    def run_training_loop(
        self,
        num_episodes: int = 500,
        snapshot_interval: int = 50,
        target_update_interval: int = 25,
        log_interval: int = 25,
        small_blind: int = 50,
        big_blind: int = 100,
        save_path: str = "checkpoints/poker_dqn_final.pt",
        plot_path: str = "checkpoints/training_progress.png",
    ) -> ConfigurablePokerNet:
        """Runs the main training process across self-play episodes."""
        cz_env: ChipzenPokerEnv = make_heads_up_holdem_env(
            start_stack=10000,
            small_blind=small_blind,
            big_blind=big_blind,
        )

        recent_rewards_bb: deque[float] = deque(maxlen=50)
        recent_wins: deque[float] = deque(maxlen=50)
        recent_losses: deque[float] = deque(maxlen=50)

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

            obs, state = cz_env.reset()
            done = [False] * len(obs.get("stacks", [True, True]))
            payouts = [0] * len(done)

            pending_state_vec: Optional[np.ndarray] = None
            pending_action_idx: Optional[int] = None
            episode_reward_bb = 0.0
            episode_losses: List[float] = []

            while not all(done):
                acting_seat = obs.get("action", -1)
                if acting_seat < 0 or state is None:
                    break

                if pending_state_vec is not None and pending_action_idx is not None:
                    next_vec = game_state_to_features(state, self.state_dim)
                    self.memory.append(
                        (pending_state_vec, pending_action_idx, 0.0, next_vec, 0.0)
                    )
                    loss_val = self.train_step()
                    if loss_val is not None:
                        episode_losses.append(loss_val)
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
                        raw_payout = payouts[agent_seat] if agent_seat < len(payouts) else 0.0
                        # 3. Normalize terminal payout by big_blind before storing/updating
                        reward_bb = self.normalize_reward(raw_payout, big_blind)
                        episode_reward_bb += reward_bb
                        self.memory.append((state_vec, action_idx, reward_bb, next_vec, 1.0))
                        loss_val = self.train_step()
                        if loss_val is not None:
                            episode_losses.append(loss_val)
                    else:
                        pending_state_vec = state_vec
                        pending_action_idx = action_idx

                state = next_state

            if pending_state_vec is not None and pending_action_idx is not None:
                next_vec = game_state_to_features(state, self.state_dim)
                raw_payout = payouts[agent_seat] if agent_seat < len(payouts) else 0.0
                reward_bb = self.normalize_reward(raw_payout, big_blind)
                episode_reward_bb += reward_bb
                self.memory.append((pending_state_vec, pending_action_idx, reward_bb, next_vec, 1.0))
                loss_val = self.train_step()
                if loss_val is not None:
                    episode_losses.append(loss_val)

            mean_ep_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            recent_rewards_bb.append(episode_reward_bb)
            recent_wins.append(1.0 if episode_reward_bb > 0 else 0.0)
            if episode_losses:
                recent_losses.append(mean_ep_loss)

            rolling_reward_bb = float(np.mean(recent_rewards_bb))
            rolling_win_rate = float(np.mean(recent_wins))
            rolling_loss = float(np.mean(recent_losses)) if recent_losses else 0.0

            self.history["episode"].append(float(episode))
            self.history["reward_bb"].append(episode_reward_bb)
            self.history["rolling_reward_bb"].append(rolling_reward_bb)
            self.history["win_rate"].append(rolling_win_rate)
            self.history["loss"].append(rolling_loss)
            self.history["epsilon"].append(self.epsilon)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % target_update_interval == 0:
                self.update_target_network()

            if episode % snapshot_interval == 0:
                self._snapshot_to_pool()

            if episode % log_interval == 0 or episode == 1:
                print(
                    f"[Ep {episode:4d}/{num_episodes}] "
                    f"roll_reward={rolling_reward_bb:+6.2f}BB | "
                    f"win_rate={rolling_win_rate:4.2f} | "
                    f"loss={rolling_loss:7.3f} | "
                    f"eps={self.epsilon:.3f} | "
                    f"buf={len(self.memory)}"
                )

            if 0 in cz_env.dealer.stacks:
                cz_env.env.reset(reset_stacks=True)

        self.save_model(save_path)
        self.plot_progress(plot_path)
        return self.current_agent


if __name__ == "__main__":
    trainer = PokerSelfPlayTrainer(state_dim=8, action_dim=3, max_grad_norm=10.0, reward_clip_bb=50.0)
    trainer.run_training_loop(num_episodes=5000, snapshot_interval=200, log_interval=50)