"""main.py – confronto DQN vs Q‑table

Partiamo dal tuo *vecchio* main ed estendiamo il flusso per:
1. allentare **separatamente** un agente DQN e un agente Q‑Learning tabulare;
2. confrontarli in inference greedy (senza MCTS) sullo stesso ambiente;
3. stampare le metriche di successo di entrambi.

Sono mantenuti tutti i parametri originali (NUM_EPISODES, ecc.) e la compatibilità
con Gym ≥0.26 (nuova API) e <0.26.
"""

from __future__ import annotations

import gym
import random
import numpy as np
import torch
import logging
import time
from collections import deque
from typing import Tuple, Union

from config import (
    MAX_EPISODE_STEPS,
    NUM_EPISODES,
    SEED,
    DQN_MIN_REPLAY,
    DQN_TARGET_UPDATE,
)
from deep_q_learning import DQNAgent
from q_learning import QLearningAgent
from frozen_lake import generate_random_map
from mcts_module import mcts, ROLLOUT_DEPTH, NUM_SIMULATIONS

# -------------------------------------------------------------
# Logging di base
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("main.log", "w", "utf-8")],
    force=True,
)
logger = logging.getLogger(__name__)
logger.info(f"num episodes = {NUM_EPISODES}, num simulations = {NUM_SIMULATIONS}")
# -------------------------------------------------------------
# Helpers compat Gym vecchia/nuova
# -------------------------------------------------------------

def _reset(env: gym.Env):
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret


def _step(env: gym.Env, action):
    ret = env.step(action)
    if len(ret) == 5:  # new Gym
        s, r, term, trunc, info = ret
        return s, r, term or trunc, info
    s, r, done, info = ret
    return s, r, done, info


# -------------------------------------------------------------
# Evaluation (pure greedy, NO MCTS)
# -------------------------------------------------------------

def evaluate_greedy(agent, env, n_episodes: int = 100) -> float:
    """Valuta senza MCTS e senza esplorazione (ε=0)."""
    wins = 0

    # Funzione greedy
    if hasattr(agent, "greedy_action"):
        greedy_fn = agent.greedy_action
    else:  # fallback per agenti custom
        def greedy_fn(state: int) -> int:
            if isinstance(state, (int, np.integer)):
                state_vec = np.eye(agent.state_size, dtype=np.float32)[state]
            else:
                state_vec = state
            with torch.no_grad():
                x = torch.from_numpy(state_vec).float().unsqueeze(0).to(agent.device)
                return int(agent.policy_net(x).argmax(dim=1).item())

    # Se l'agente ha epsilon lo portiamo a 0 temporaneamente
    if hasattr(agent, "epsilon"):
        old_eps, agent.epsilon = agent.epsilon, 0.0

    for _ in range(n_episodes):
        state = _reset(env)
        for _ in range(MAX_EPISODE_STEPS):
            action = greedy_fn(state)
            state, reward, done, _ = _step(env, action)
            if done:
                if reward > 0:
                    wins += 1
                break

    if hasattr(agent, "epsilon"):
        agent.epsilon = old_eps

    return wins / n_episodes


# -------------------------------------------------------------
# Training loops
# -------------------------------------------------------------

def train_dqn(env: gym.Env) -> DQNAgent:
    agent = DQNAgent(env.observation_space.n, env.action_space.n, seed=SEED)
    rewards, successes = deque(maxlen=100), deque(maxlen=100)

    for ep in range(1, NUM_EPISODES + 1):
        state = _reset(env)
        total_reward, done = 0.0, False
        for _ in range(MAX_EPISODE_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = _step(env, action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        successes.append(int(total_reward > 0))
        if ep % 100 == 0:
            logger.info(
                "DQN ep=%4d | avgR=%.3f | succ=%.1f%%",
                ep,
                np.mean(rewards),
                100 * sum(successes) / len(successes),
            )
    return agent


def train_qtable(env: gym.Env) -> QLearningAgent:
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, seed=SEED)
    # compat: alias per API uniforme
    agent.select_action = agent.choose_action
    # per inference greedy
    agent.greedy_action = lambda s: int(agent.q_table[s].argmax())

    rewards, successes = deque(maxlen=100), deque(maxlen=100)

    for ep in range(1, NUM_EPISODES + 1):
        state = _reset(env)
        total_reward, done = 0.0, False
        for _ in range(MAX_EPISODE_STEPS):
            # ε‑greedy sull’agente tabulare
            action = agent.select_action(state)
            next_state, reward, done, _ = _step(env, action)
            best_next = np.max(agent.q_table[next_state])
            td_target = reward + agent.gamma * best_next
            agent.q_table[state, action] += agent.alpha * (
                td_target - agent.q_table[state, action]
            )
            state = next_state
            total_reward += reward
            if done:
                break
        # decay ε
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        rewards.append(total_reward)
        successes.append(int(total_reward > 0))
        if ep % 100 == 0:
            logger.info(
                "Q‑table ep=%4d | avgR=%.3f | succ=%.1f%% | ε=%.2f",
                ep,
                np.mean(rewards),
                100 * sum(successes) / len(successes),
                agent.epsilon,
            )
    return agent


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    t0 = time.time()
    # seed global
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # mappa fissa
    fixed_map = generate_random_map(size=6, p=0.8, seed=SEED)

    # env separati per non mischiare stati
    env_dqn = gym.make("CustomFrozenLake-v1", desc=fixed_map, is_slippery=False)
    env_ql  = gym.make("CustomFrozenLake-v1", desc=fixed_map, is_slippery=False)
    env_eval = gym.make("CustomFrozenLake-v1", desc=fixed_map, is_slippery=False)

    logger.info("Training DQN…")
    dqn_agent = train_dqn(env_dqn)
    env_dqn.close()

    logger.info("Training Q‑table…")
    ql_agent = train_qtable(env_ql)
    env_ql.close()

    logger.info("Evaluating greedy agents (100 episodes)…")
    dqn_rate = evaluate_greedy(dqn_agent, env_eval)
    ql_rate = evaluate_greedy(ql_agent, env_eval)
    env_eval.close()

    logger.info("Success DQN = %.1f%%", dqn_rate * 100)
    logger.info("Success Q‑table = %.1f%%", ql_rate * 100)
    logger.info("Completed in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
