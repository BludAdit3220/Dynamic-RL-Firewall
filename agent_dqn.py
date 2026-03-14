from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import tensorflow as tf
from keras import Model, layers, optimizers


@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000
    tau: float = 0.005  # soft target update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states.astype(np.float32),
            actions.astype(np.int32),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def build_q_network(state_dim: int, num_actions: int) -> Model:
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    # Ensure Keras gets a plain Python int for units
    outputs = layers.Dense(int(num_actions), activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


class DQNAgent:
    """
    Simple DQN agent using Keras for the RL firewall.

    - Uses an epsilon-greedy exploration schedule.
    - Maintains online and target Q networks.
    - Supports soft target updates via tau.
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.online_network = build_q_network(config.state_dim, config.num_actions)
        self.target_network = build_q_network(config.state_dim, config.num_actions)
        self.target_network.set_weights(self.online_network.get_weights())

        self.optimizer = optimizers.Adam(learning_rate=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        self.epsilon = config.epsilon_start
        self.global_step = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.num_actions)

        q_values = self.online_network(np.expand_dims(state, axis=0), training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def _update_epsilon(self):
        frac = min(1.0, self.global_step / self.config.epsilon_decay_steps)
        self.epsilon = self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function
    def _train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        gamma = self.config.gamma
        num_actions = self.config.num_actions

        with tf.GradientTape() as tape:
            q_values = self.online_network(states, training=True)
            q_values = tf.gather(
                q_values, tf.cast(actions, tf.int32), batch_dims=1
            )  # shape (batch,)

            next_q_values_target = self.target_network(next_states, training=False)
            max_next_q = tf.reduce_max(next_q_values_target, axis=1)

            targets = rewards + gamma * max_next_q * (1.0 - dones)

            loss = tf.reduce_mean(tf.square(targets - q_values))

        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.online_network.trainable_variables)
        )
        return loss

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.config.batch_size)

        loss = self._train_step(states, actions, rewards, next_states, dones)
        self._soft_update_target()

        self.global_step += 1
        self._update_epsilon()
        return float(loss.numpy())

    def _soft_update_target(self):
        tau = self.config.tau
        online_weights = self.online_network.get_weights()
        target_weights = self.target_network.get_weights()
        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            new_weights.append(tau * ow + (1.0 - tau) * tw)
        self.target_network.set_weights(new_weights)

    def save(self, path: str):
        self.online_network.save(path)

    def load(self, path: str):
        self.online_network = tf.keras.models.load_model(path)
        self.target_network.set_weights(self.online_network.get_weights())
