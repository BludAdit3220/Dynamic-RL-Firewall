import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DynamicFirewallEnv(gym.Env):
    """
    OpenAI Gym-style environment for a reinforcement-learning firewall.

    This environment is intentionally abstracted from real packets. It expects
    preprocessed traffic samples with features and a binary/ternary label
    indicating whether the connection is benign or malicious (and optionally
    attack severity).

    You can train offline on recorded traffic and then deploy the learned
    policy for live traffic, where the features are computed in real time.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        traffic_samples: np.ndarray,
        labels: np.ndarray,
        benign_reward: float = 1.0,
        block_attack_reward: float = 2.0,
        false_positive_penalty: float = -1.0,
        false_negative_penalty: float = -3.0,
        rate_limit_penalty: float = -0.3,
        seed: int | None = None,
    ):
        """
        Args:
            traffic_samples: np.ndarray of shape (N, F) with precomputed features.
            labels: np.ndarray of shape (N,) with 0=benign, 1=attack (extendable).
            benign_reward: reward for allowing benign traffic.
            block_attack_reward: reward for blocking malicious traffic.
            false_positive_penalty: penalty for blocking benign traffic.
            false_negative_penalty: penalty for allowing malicious traffic.
            rate_limit_penalty: small penalty for rate-limiting (latency cost).
        """
        super().__init__()

        assert traffic_samples.shape[0] == labels.shape[0], "Mismatched samples and labels"

        self.traffic_samples = traffic_samples.astype(np.float32)
        self.labels = labels.astype(np.int32)

        self.num_samples, self.num_features = self.traffic_samples.shape

        # Observation: feature vector for current flow/packet.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32
        )

        # Actions: 0=allow, 1=block, 2=rate-limit
        self.action_space = spaces.Discrete(3)

        self.benign_reward = benign_reward
        self.block_attack_reward = block_attack_reward
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty
        self.rate_limit_penalty = rate_limit_penalty

        self.current_index = 0

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _compute_reward(self, action: int, label: int) -> float:
        """Reward function based on action and ground-truth label."""
        if label == 0:  # benign
            if action == 0:  # allow
                return self.benign_reward
            elif action == 1:  # block (false positive)
                return self.false_positive_penalty
            elif action == 2:  # rate-limit benign
                return self.benign_reward + self.rate_limit_penalty
        else:  # attack
            if action == 1:  # block
                return self.block_attack_reward
            elif action == 0:  # allow (false negative)
                return self.false_negative_penalty
            elif action == 2:  # rate-limit attack
                return self.block_attack_reward + self.rate_limit_penalty

        return 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Shuffle samples each episode for stochasticity.
        indices = self.np_random.permutation(self.num_samples)
        self.traffic_samples = self.traffic_samples[indices]
        self.labels = self.labels[indices]

        self.current_index = 0
        obs = self.traffic_samples[self.current_index]
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        label = int(self.labels[self.current_index])
        obs = self.traffic_samples[self.current_index]
        reward = self._compute_reward(action, label)

        self.current_index += 1
        terminated = self.current_index >= self.num_samples
        truncated = False

        if terminated:
            next_obs = np.zeros_like(obs)
        else:
            next_obs = self.traffic_samples[self.current_index]

        info = {
            "label": label,
            "action": action,
            "reward": reward,
        }
        return next_obs, reward, terminated, truncated, info

    def render(self):
        # For now, this environment is numeric only. Integrate with a dashboard instead.
        pass

