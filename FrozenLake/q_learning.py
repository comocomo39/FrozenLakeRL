"""
Module implementing a Q-Learning agent with epsilon-greedy policy,
epsilon decay, and reproducible behavior via seeding.
Assume che l'env restituisca già le ricompense corrette (STEP_PENALTY, HOLE_PENALTY, GOAL_REWARD).
"""
from typing import Optional, List
import numpy as np
import gym
from config import MAX_EPISODE_STEPS, NUM_EPISODES, SEED
from config import QL_ALPHA, QL_GAMMA, QL_EPSILON, QL_EPSILON_MIN, QL_EPSILON_DECAY

class QLearningAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float = QL_ALPHA,
        gamma: float = QL_GAMMA,
        epsilon: float = QL_EPSILON,
        epsilon_min: float = QL_EPSILON_MIN,
        epsilon_decay: float = QL_EPSILON_DECAY,
        seed: Optional[int] = None
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size), dtype=float)
        self.rng = np.random.default_rng(seed=SEED)

    def reset(self) -> None:
        """Ripristina Q-table e ε al valore iniziale."""
        self.q_table.fill(0.0)
        self.epsilon = self.epsilon_start
        
    
    def greedy_action(self, state):
        return int(self.Q[state].argmax())


    def choose_action(self, state: int) -> int:
        """Epsilon-greedy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return int(np.argmax(self.q_table[state]))

    def learn(
        self,
        env: gym.Env,
        num_episodes: int = NUM_EPISODES,
        max_steps_per_episode: int = MAX_EPISODE_STEPS
    ) -> List[float]:
        """
        Addestra l’agent su `env`, restituendo la reward cumulata per episodio.

        Ora usa **direttamente** `reward` restituito da env.step(), senza ulteriori aggiustamenti.
        """
        rewards_per_episode: List[float] = []
        for ep in range(1, num_episodes + 1):
            reset_result = env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                action = self.choose_action(state)
                result = env.step(action)
                # gestisce API Gym 4- o 5-tuple
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result

                # update Q-learning con reward **diretta**:
                
                # ---- ESERCIZIO 1 ----------------------------------
                
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error
               
                # ---------------------------------------------------
                
                state = next_state
                total_reward += reward
                if done:
                    break

            # decay ε
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

        return rewards_per_episode

    def save(self, filepath: str) -> None:
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        self.q_table = np.load(filepath)
