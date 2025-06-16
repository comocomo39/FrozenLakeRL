# new_main.py – versione 3 completa (con delay)
# --------------------------------------------------
"""
Visualizzazione MCTS + DQN su FrozenLake 4×4 con ritardo tra i frame.

Novità:
* **DELAY_SEC** rende l’animazione comprensibile (default 0.2 s).
* Bugfix precedenti: grid dtype corretto, NameError in UCT risolto, duplicati rimossi.

Riduci `DELAY_SEC` per accelerare o portalo a 0 per disattivare completamente il delay.
"""

import random
import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from config import (
    MAX_EPISODE_STEPS,
    NUM_EPISODES,
    ROLLOUT_DEPTH,
    SEED,
    C_PARAM,
)
from deep_q_learning import DQNAgent
from frozen_lake import generate_random_map
from mcts_module import mcts

# Ritardo visivo (secondi)
DELAY_SEC = 0.1
MAX_EPISODE_STEPS = 10

# ────────────────────────────────
#  Struttura ausiliaria per l’albero
# ────────────────────────────────
class _VizNode:
    def __init__(self, parent=None, state=None, action_from_parent=None):
        self.parent = parent
        self.state = state
        self.action_from_parent = action_from_parent
        self.children: dict[int, "_VizNode"] = {}
        self.N = 0
        self.Q = 0.0


def _extract_snapshot(root: "_VizNode") -> dict[tuple[int, ...], int]:
    """Ritorna {path_tuple: N_visits} per disegnare l’albero."""
    snap: dict[tuple[int, ...], int] = {}
    stack = [(root, [])]
    while stack:
        node, path = stack.pop()
        snap[tuple(path)] = node.N
        for a, child in node.children.items():
            stack.append((child, path + [a]))
    return snap


def _mcts_for_visualization(
    root_state: int,
    transitions: list[list[list[tuple[float, int, float, bool]]]],
    nA: int,
    rollout_agent: DQNAgent | None = None,
    num_simulations: int = 20,
    rollout_depth: int = ROLLOUT_DEPTH,
    c: float = C_PARAM,
):
    """MCTS minimale per mostrare un albero istantaneo (no back‑prop nel modello principale)."""
    root = _VizNode(state=root_state)
    gamma = 0.99

    for _ in range(num_simulations):
        node, s, depth = root, root_state, 0
        path: list[int] = []

        # Selection + Expansion
        while node.N > 0:
            uct: list[float] = []
            for a in range(nA):
                if a in node.children and node.children[a].N > 0:
                    child = node.children[a]
                    uct.append((child.Q / child.N) + c * np.sqrt(np.log(node.N) / child.N))
                else:
                    uct.append(float("inf"))
            a_best = int(np.argmax(uct))
            path.append(a_best)
            if a_best not in node.children:
                node.children[a_best] = _VizNode(parent=node, action_from_parent=a_best)
            node = node.children[a_best]
            probs, next_states, rews, dones = zip(*transitions[s][a_best])
            idx = np.random.choice(len(probs), p=probs)
            s = next_states[idx]
            depth += 1
            if dones[idx]:
                break

        node.state = s

        # Rollout
        reward = 0.0
        s_roll = s
        for t in range(rollout_depth - depth):
            a_roll = rollout_agent.greedy_action(s_roll) if rollout_agent else np.random.randint(nA)
            probs, next_states, rews, dones = zip(*transitions[s_roll][a_roll])
            idx = np.random.choice(len(probs), p=probs)
            s_roll = next_states[idx]
            reward += (gamma ** t) * rews[idx]
            if dones[idx]:
                break

        # Backup
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += reward
            cur = cur.parent

    return _extract_snapshot(root)


def _draw_tree_snapshot(ax, snapshot, *, leaf_color_fn=None):
    """Disegna l’albero su Axes pulito."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])

    # Raggruppa per profondità
    levels: dict[int, list[tuple[int, ...]]] = {}
    for path in snapshot.keys():
        levels.setdefault(len(path), []).append(path)

    positions: dict[tuple[int, ...], tuple[float, float]] = {}
    for depth, paths in levels.items():
        paths_sorted = sorted(paths)
        n_nodes = len(paths_sorted)
        for i, path in enumerate(paths_sorted):
            x = i - (n_nodes - 1) / 2
            y = -depth
            positions[path] = (x, y)
            color = leaf_color_fn(path) if leaf_color_fn else "skyblue"
            ax.scatter(x, y, s=80, c=color, edgecolors="k", zorder=2)

    # Aggiungi archi
    for path in snapshot.keys():
        if not path:
            continue
        parent_path = tuple(path[:-1])
        if parent_path in positions:
            x1, y1 = positions[parent_path]
            x2, y2 = positions[path]
            ax.plot([x1, x2], [y1, y2], c="gray", zorder=1)


def _make_leaf_color_fn(root: _VizNode, fixed_map: list[str], N: int):
    tile_codes = {"G": "green", "H": "red"}

    def fn(path: tuple[int, ...]):
        node = root
        for a in path:
            node = node.children.get(a)
            if node is None:
                return "skyblue"
        if node.children:
            return "skyblue"
        r, c = divmod(node.state, N)
        tile = fixed_map[r][c]
        return tile_codes.get(tile, "skyblue")

    return fn

# ────────────────────────────────
#  Funzione di valutazione greedy
# ────────────────────────────────

def _evaluate(agent: DQNAgent, env: gym.Env, n_episodes: int = 50) -> float:
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            successes += 1
    return successes / n_episodes

# ────────────────────────────────
#  MAIN
# ────────────────────────────────

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Mappa deterministica
    fixed_map = generate_random_map(size=4, p=0.8, seed=SEED)
    env = gym.make("CustomFrozenLake-v1", desc=fixed_map, is_slippery=True, render_mode="rgb_array")

    nS, nA = env.observation_space.n, env.action_space.n
    transitions = [[list(env.P[s][a]) for a in range(nA)] for s in range(nS)]

    # Grid numerica per imshow
    tile_codes = {"F": 0, "H": 1, "G": 2, "S": 3}
    grid = np.array([[tile_codes[ch] for ch in row] for row in fixed_map], dtype=int)
    N = len(fixed_map)

    agent = DQNAgent(nS, nA, seed=SEED)
    reward_window, success_window = deque(maxlen=100), deque(maxlen=100)

    # Setup figure
    plt.ion()
    fig, (ax_map, ax_tree_temp, ax_tree_real) = plt.subplots(
        1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    cmap = colors.ListedColormap(["white", "black", "green", "lightblue"])
    ax_map.imshow(grid, cmap=cmap, norm=colors.BoundaryNorm([0, 1, 2, 3, 4], 4))
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    agent_dot = ax_map.scatter([], [], s=120, c="red", marker="o", zorder=2)

    ax_tree_temp.set_title("MCTS (20 simulazioni)")
    ax_tree_temp.set_xticks([])
    ax_tree_temp.set_yticks([])

    ax_tree_real.set_title("Albero mosse reali")
    ax_tree_real.set_xticks([])
    ax_tree_real.set_yticks([])

    fig.canvas.manager.set_window_title("FrozenLake – MCTS + DQN")

    state, _ = env.reset()
    root_real = _VizNode(state=state)

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        current_real = root_real

        r0, c0 = divmod(state, N)
        agent_dot.set_offsets((c0, r0))

        for _ in range(MAX_EPISODE_STEPS):
            # Azione tramite MCTS guidato
            action, _ = mcts(
                env,
                state,
                rollout_agent=agent,
                num_simulations=10,
                rollout_depth=ROLLOUT_DEPTH,
                c=C_PARAM,
            )

            # Snapshot albero temporaneo
            snap_temp = _mcts_for_visualization(
                state,
                transitions,
                nA,
                rollout_agent=agent,
                num_simulations=20,
                rollout_depth=ROLLOUT_DEPTH,
                c=C_PARAM,
            )
            _draw_tree_snapshot(ax_tree_temp, snap_temp)

            # Esecuzione azione
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Aggiornamenti visuali
            r, c = divmod(next_state, N)
            agent_dot.set_offsets((c, r))

            # Aggiorna albero reale
            if action not in current_real.children:
                current_real.children[action] = _VizNode(parent=current_real, state=next_state, action_from_parent=action)
            current_real = current_real.children[action]
            current_real.N += 1

            leaf_fn = _make_leaf_color_fn(root_real, fixed_map, N)
            _draw_tree_snapshot(ax_tree_real, _extract_snapshot(root_real), leaf_color_fn=leaf_fn)

            # Aggiornamento DQN
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            if done:
                break

            # Refresh figure & delay
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(DELAY_SEC)

        reward_window.append(total_reward)
        success_window.append(int(total_reward > 0))

        if ep % 10 == 0:
            avg_r = np.mean(reward_window)
            succ_rate = 100 * np.mean(success_window)
            print(f"Episodio {ep}: reward medio(100)={avg_r:.2f} | success={succ_rate:.1f}%")

    # Valutazione finale greedy
    success_final = _evaluate(agent, env) * 100
    print(f"Success rate finale greedy: {success_final:.1f}%")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
