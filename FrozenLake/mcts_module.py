import numpy as np
from config import NUM_SIMULATIONS, ROLLOUT_DEPTH, QL_GAMMA, C_PARAM

def _compile_model(env):
    """
    Compila i modelli di transizione e reward dall'ambiente.
    Usa tutte le probabilità di transizione in env.P
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    # Matrici: nxt[s, a] è lista di (prob, next_state), rew[s, a] è lista dei reward corrispondenti
    transitions = [[[] for _ in range(nA)] for _ in range(nS)]
    for s in range(nS):
        for a in range(nA):
            for prob, next_s, rew, done in env.P[s][a]:
                transitions[s][a].append((prob, next_s, rew, done))
    return transitions


def mcts(env, root_state, rollout_agent=None, num_simulations=NUM_SIMULATIONS, rollout_depth=ROLLOUT_DEPTH, c= C_PARAM):
    """
    MCTS che usa tutte le transizioni pesate, con backprop comprensiva della radice.
    """
    
    nS = env.observation_space.n
    nA = env.action_space.n
    # Pre-compila modello
    transitions = _compile_model(env)
    # Conteggi e valori
    Ns = np.zeros(nS, dtype=int)
    Nsa = np.zeros((nS, nA), dtype=int)
    Q = np.zeros((nS, nA), dtype=float)

    # Simulazioni
    for _ in range(num_simulations):
        s = root_state
        path = [(s, None)]  # include sempre la radice
        reward = 0.0
        depth = 0
        # Selection & Expansion
        while True:
            if Ns[s] == 0:  # nodo non visitato
                break
            # UCT
            uct_values = np.zeros(nA)
            for a in range(nA):
                if Nsa[s, a] > 0:
                    uct_values[a] = Q[s, a] / Nsa[s, a] + c * np.sqrt(np.log(Ns[s]) / Nsa[s, a])
                else:
                    uct_values[a] = np.inf
            a = int(np.argmax(uct_values))
            path.append((s, a))
            # Espandi tramite campionamento della transizione
            probs, next_states, rews, dones = zip(*transitions[s][a])
            idx = np.random.choice(len(probs), p=probs)
            s = next_states[idx]
            reward += (QL_GAMMA ** depth) * rews[idx]
            depth += 1
            if dones[idx]:
                break

        # Rollout
        # ----- ESERCIZIO ROLLOUT 2 -------------------------------------------
        
        rollout_s = s
        rollout_reward = 0.0
        for t in range(rollout_depth):
            if rollout_agent is not None:
                a_roll = rollout_agent.greedy_action(rollout_s)
            else:
                a_roll = np.random.randint(nA)
            probs, next_states, rews, dones = zip(*transitions[rollout_s][a_roll])
            idx = np.random.choice(len(probs), p=probs)
            rollout_reward += (QL_GAMMA ** (depth + t)) * rews[idx]
            rollout_s = next_states[idx]
            if dones[idx]:
                break
        reward += rollout_reward
        
        # -------------------------------------------------------------------

        # Backpropagation
        for (state, action) in reversed(path):
            Ns[state] += 1
            if action is not None:
                Nsa[state, action] += 1
                Q[state, action] += (reward - Q[state, action]) / Nsa[state, action]

    # Politica finale per la radice
    counts = Nsa[root_state]
    if counts.sum() == 0:
        counts = np.ones_like(counts) * 1e-9
    pi = counts / counts.sum()
    best_action = int(np.argmax(counts))
    return best_action, pi