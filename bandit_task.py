"""
bandit_task.py
==============
Core simulation and modeling code for the multi-armed bandit task.
Simulates human-like behavior and fits RL models to behavioral data.

Usage:
    from bandit_task import BanditEnvironment, simulate_agent, QLearningModel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. ENVIRONMENT
# ─────────────────────────────────────────────

class BanditEnvironment:
    """
    Multi-armed bandit environment.
    Each arm has a fixed (but hidden) reward probability.
    """

    def __init__(self, n_arms: int = 4, reward_probs: List[float] = None, seed: int = 42):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

        if reward_probs is not None:
            assert len(reward_probs) == n_arms
            self.reward_probs = np.array(reward_probs)
        else:
            self.reward_probs = self.rng.uniform(0.1, 0.9, size=n_arms)

        self.best_arm = np.argmax(self.reward_probs)

    def step(self, action: int) -> float:
        """Pull an arm and receive a reward (0 or 1)."""
        return float(self.rng.random() < self.reward_probs[action])

    def __repr__(self):
        probs = ", ".join([f"Arm {i}: {p:.2f}" for i, p in enumerate(self.reward_probs)])
        return f"BanditEnvironment({probs})"


# ─────────────────────────────────────────────
# 2. AGENTS
# ─────────────────────────────────────────────

def simulate_random_agent(env: BanditEnvironment, n_trials: int = 100) -> dict:
    """Baseline: chooses arms uniformly at random."""
    rng = np.random.default_rng(0)
    choices, rewards = [], []

    for _ in range(n_trials):
        action = rng.integers(env.n_arms)
        reward = env.step(action)
        choices.append(action)
        rewards.append(reward)

    return {"choices": np.array(choices), "rewards": np.array(rewards)}


def simulate_greedy_agent(env: BanditEnvironment, n_trials: int = 100) -> dict:
    """
    Greedy agent: always picks the arm with the highest estimated reward.
    With epsilon=0.1 exploration at the start.
    """
    rng = np.random.default_rng(1)
    q_values = np.zeros(env.n_arms)   # estimated reward per arm
    n_chosen = np.zeros(env.n_arms)   # times each arm was chosen
    choices, rewards = [], []

    for t in range(n_trials):
        # Explore first few trials, then be greedy
        if t < env.n_arms:
            action = t  # try each arm once first
        else:
            action = np.argmax(q_values)

        reward = env.step(action)
        n_chosen[action] += 1
        # Incremental mean update
        q_values[action] += (reward - q_values[action]) / n_chosen[action]

        choices.append(action)
        rewards.append(reward)

    return {"choices": np.array(choices), "rewards": np.array(rewards)}


def simulate_qlearning_agent(
    env: BanditEnvironment,
    alpha: float = 0.3,
    beta: float = 5.0,
    n_trials: int = 100,
    seed: int = 2
) -> dict:
    """
    Q-learning agent with softmax action selection.

    Parameters
    ----------
    alpha : float
        Learning rate (0–1). How fast Q-values update.
    beta : float
        Inverse temperature. Higher = more exploitative.
    """
    rng = np.random.default_rng(seed)
    q_values = np.zeros(env.n_arms)
    choices, rewards = [], []

    for _ in range(n_trials):
        # Softmax policy
        logits = beta * q_values
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()
        action = rng.choice(env.n_arms, p=probs)

        reward = env.step(action)

        # Q-value update (delta rule)
        prediction_error = reward - q_values[action]
        q_values[action] += alpha * prediction_error

        choices.append(action)
        rewards.append(reward)

    return {
        "choices": np.array(choices),
        "rewards": np.array(rewards),
        "q_values_final": q_values.copy()
    }


# ─────────────────────────────────────────────
# 3. MODEL FITTING
# ─────────────────────────────────────────────

class QLearningModel:
    """
    Fits a Q-learning model to behavioral data (choices + rewards).
    Uses maximum likelihood estimation to recover alpha and beta.
    """

    def __init__(self, n_arms: int = 4):
        self.n_arms = n_arms
        self.fitted_params = None

    def negative_log_likelihood(
        self, params: np.ndarray, choices: np.ndarray, rewards: np.ndarray
    ) -> float:
        """
        Compute -log likelihood of observed choices under Q-learning model.
        params = [alpha, beta]
        """
        alpha, beta = params

        # Parameter bounds check
        if not (0 < alpha < 1) or not (0 < beta < 30):
            return 1e6

        q_values = np.zeros(self.n_arms)
        log_likelihood = 0.0

        for t in range(len(choices)):
            action = choices[t]
            reward = rewards[t]

            # Softmax probability of chosen action
            logits = beta * q_values
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()

            # Accumulate log likelihood
            log_likelihood += np.log(probs[action] + 1e-10)

            # Update Q-value
            q_values[action] += alpha * (reward - q_values[action])

        return -log_likelihood

    def fit(self, choices: np.ndarray, rewards: np.ndarray, n_restarts: int = 10) -> dict:
        """
        Fit model to data using multiple random restarts.
        Returns best-fitting parameters.
        """
        rng = np.random.default_rng(42)
        best_result = None

        for _ in range(n_restarts):
            # Random starting point
            x0 = [rng.uniform(0.01, 0.99), rng.uniform(0.1, 15)]

            result = minimize(
                self.negative_log_likelihood,
                x0=x0,
                args=(choices, rewards),
                method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-4}
            )

            if best_result is None or result.fun < best_result.fun:
                best_result = result

        self.fitted_params = {
            "alpha": best_result.x[0],
            "beta": best_result.x[1],
            "neg_log_likelihood": best_result.fun,
        }
        return self.fitted_params


# ─────────────────────────────────────────────
# 4. PLOTTING UTILITIES
# ─────────────────────────────────────────────

def plot_cumulative_rewards(results_dict: dict, title: str = "Agent Comparison") -> None:
    """
    Plot cumulative rewards over trials for multiple agents.

    Parameters
    ----------
    results_dict : dict
        Keys = agent names, values = result dicts from simulate_* functions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    for i, (name, result) in enumerate(results_dict.items()):
        rewards = result["rewards"]
        cumulative = np.cumsum(rewards)
        running_mean = np.convolve(rewards, np.ones(10)/10, mode='valid')

        axes[0].plot(cumulative, label=name, color=colors[i % len(colors)], linewidth=2)
        axes[1].plot(running_mean, label=name, color=colors[i % len(colors)], linewidth=2)

    axes[0].set_title("Cumulative Reward", fontsize=13)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Running Average Reward (window=10)", fontsize=13)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Avg Reward")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_arm_selection(choices: np.ndarray, n_arms: int, title: str = "Arm Selection") -> None:
    """Visualize which arm was chosen over time."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(choices)), choices, alpha=0.4, s=15, c=choices, cmap="tab10")
    ax.set_yticks(range(n_arms))
    ax.set_yticklabels([f"Arm {i}" for i in range(n_arms)])
    ax.set_xlabel("Trial")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_parameter_recovery(
    true_alphas: List[float],
    true_betas: List[float],
    recovered_alphas: List[float],
    recovered_betas: List[float]
) -> None:
    """Check how well the model recovers known parameters (parameter recovery test)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(true_alphas, recovered_alphas, alpha=0.7, color="#457B9D", edgecolors="white")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("True α (learning rate)")
    axes[0].set_ylabel("Recovered α")
    axes[0].set_title("Parameter Recovery: Learning Rate")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(true_betas, recovered_betas, alpha=0.7, color="#2A9D8F", edgecolors="white")
    diag = [min(true_betas), max(true_betas)]
    axes[1].plot(diag, diag, "k--", alpha=0.4)
    axes[1].set_xlabel("True β (inverse temperature)")
    axes[1].set_ylabel("Recovered β")
    axes[1].set_title("Parameter Recovery: Exploration")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
