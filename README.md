# FrozenLake‑RL

Reference implementation of **tabular Q‑Learning**, **Deep Q‑Network (DQN)** and optional **Monte‑Carlo Tree Search (MCTS)** on a custom‑reward variant of *Frozen Lake* (Gym).

## ✨ Highlights

* Custom environment (`frozen_lake.py`) with shaped rewards and deterministic/slippery maps.
* **Two agents**:

  * `QLearningAgent` – classic table‑based Q‑Learning.
  * `DQNAgent` – linear Q‑Network with replay buffer & target network.
* Lightweight **MCTS module** (`mcts_module.py`) that can wrap any agent and plan ahead.
* Single‑command benchmark: trains both agents and prints success rates.
* Extra script (`new_main.py`) that **visualises** MCTS trees + agent path in real time.

## Quick Start

| Task                                      | Command                     |
| ----------------------------------------- | --------------------------- |
| Train **DQN** *and* Q‑table, then compare | `python main.py`            |
| Visualise MCTS vs DQN (4×4 map)           | `python new_main.py`        |

All scripts honour the hyper‑parameters in `config.py` (episodes, γ, ε‑decay, DQN buffer size, etc.).

## Reinforcement‑Learning Workflow

1. **Interaction** – the agent explores the lake for `NUM_EPISODES`, collecting `(s, a, r, s′)` pairs.
2. **Update**
   • *Q‑table*: tabular Bellman update `Q ← Q + α (r + γ max Q′ − Q)`
   • *DQN*: minibatch TD‑error on replay buffer + periodic target sync.
3. **Planning (optional)** – at inference time you can call `mcts(env, state, rollout_agent=agent, …)` to run *N* simulated playouts that use the **agent’s greedy policy** for rollouts.
4. **Evaluation** – greedy success rate (ε = 0) is reported every 100 episodes and at the end.

## Repository Layout

```text
frozen_lake.py        # custom Gym env with shaped rewards
q_learning.py         # tabular agent
deep_q_learning.py    # DQN agent & replay buffer
mcts_module.py        # lightweight MCTS (planning only)
main.py               # train & compare DQN vs Q‑table
new_main.py           # visual demo (MCTS + DQN)
config.py             # shared hyper‑parameters
```

## License

MIT License. Feel free to use and cite!
